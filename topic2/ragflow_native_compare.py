"""Compare raw RAGFlow, native Wiki retrieval, and guarded original evidence."""

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256 as digest
import json
from pathlib import Path
import re
from statistics import median
from time import perf_counter
import unicodedata
from urllib.request import ProxyHandler, Request, build_opener

from baseline import sentence_blocks
from cloud_experiment import BudgetTokenizer, TOKENIZER
from cloud_reader import CloudReader, MODEL
from dependency_development import SOURCE
from external_compression import append, dump, read_question
from intent_evidence_experiment import sha256
from lightweight_wiki_experiment import chain_check, route
from paired import official_answer, support_sets
from prepare import read_jsonl
from ragflow_native_setup import RELEASE, authenticate, documents
from semantic_bridge_experiment import verify_freeze
from sweep_compare import difference, measure
from wiki_bridge import guarded_bridges, span_text, title_surface


FULL_BUDGET = 2560
SHORT_BUDGET = 1024
SEARCH_SETTINGS = {"page_size": 64, "similarity_threshold": 0.0, "keywords_similarity_weight": 0.7,
                   "knn_top_k": 128, "knn_num_candidates": 256, "rerank_candidates_count": 64,
                   "knowledge_compilation_scope": "dataset"}
ARMS = ("A_raw", "B_native_wiki", "C_guarded")


def verify_ready(client, state):
    dataset = state["dataset_id"]
    statuses = Counter(row["ingestion_status"] for row in documents(client, dataset))
    compile_status = client.call("GET", f"/datasets/{dataset}/compilation/status")
    artifacts = client.call("GET", f"/datasets/{dataset}/artifacts", params={"page_size": 1})["total"]
    if dict(statuses) != {"COMPLETED": len(state["documents"])} or compile_status["inflight"] or compile_status["backlog"]:
        raise RuntimeError(f"Index still preparing: {dict(statuses)}, {compile_status['state']}, {artifacts} Wiki artifacts")
    if not artifacts:
        raise RuntimeError("No native Wiki artifacts found in the completed dataset")
    return artifacts


class Lineage:
    def __init__(self, client, dataset):
        tenant = client.call("GET", f"/datasets/{dataset}")["tenant_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", tenant):
            raise ValueError("Unexpected Elasticsearch tenant index name")
        self.url = f"http://127.0.0.1:19200/ragflow_{tenant}/_mget"
        env = dict(line.split("=", 1) for line in (RELEASE / "docker/.env").read_text().splitlines()
                   if line.startswith("ELASTIC_PASSWORD="))
        self.headers = {"Authorization": "Basic " + base64.b64encode(
            ("elastic:" + env["ELASTIC_PASSWORD"]).encode()).decode(), "Content-Type": "application/json"}
        self.opener = build_opener(ProxyHandler({}))

    def for_chunks(self, ids):
        if not ids:
            return {}
        body = {"docs": [{"_id": chunk_id, "_source": [
            "source_doc_ids", "source_doc_ids_kwd", "meta", "meta_json", "metadata"
        ]} for chunk_id in ids]}
        request = Request(self.url, json.dumps(body).encode(), self.headers, method="POST")
        with self.opener.open(request, timeout=30) as response:
            rows = json.load(response)["docs"]
        results = {}
        for row in rows:
            if not row["found"]:
                continue
            source = row["_source"]
            meta = source.get("meta") or source.get("meta_json") or source.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            raw_ids = source.get("source_doc_ids") or source.get("source_doc_ids_kwd") or meta.get("source_doc_ids") or []
            results[row["_id"]] = [raw_ids] if isinstance(raw_ids, str) else raw_ids
        return results


def retrieve(client, lineage, state, question, wiki):
    id_to_local = {state["documents"][c["id"]]: i for i, c in enumerate(question["candidates"])}
    request = {**SEARCH_SETTINGS, "dataset_ids": [state["dataset_id"]], "document_ids": list(id_to_local),
               "question": question["question"], "include_knowledge_compilation": wiki}
    start = perf_counter()
    result = client.call("POST", "/retrieval", request)
    search_ms = (perf_counter() - start) * 1000
    source_map = lineage.for_chunks([r["id"] for r in result["chunks"] if r.get("compile_kwd")])
    kept, dropped = [], []
    for row in result["chunks"]:
        ids = source_map.get(row["id"], []) if row.get("compile_kwd") else [row.get("document_id")]
        if not ids or set(ids) - id_to_local.keys():
            dropped.append({"id": row["id"], "compile_kwd": row.get("compile_kwd"), "source_docs": ids})
            continue
        if row.get("content"):
            kept.append({**row, "source_indices": sorted(id_to_local[doc] for doc in set(ids))})
    return {"chunks": kept[:20], "dropped": dropped, "search_ms": search_ms,
            "capture_ms": (perf_counter() - start) * 1000}


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def relay_records(output):
    return [record for path in sorted(output.glob("ragflow-model-usage*.jsonl")) for record in read_jsonl(path)]


def source_spans(chunks, docs):
    matched = defaultdict(set)
    for row in chunks:
        text = canonical(row["content"])
        for i in row["source_indices"]:
            for span in docs[i]["spans"]:
                j = span["index"]
                original = canonical(span_text(docs[i], j))
                if original and original in text:
                    matched[i].add(j)
    return matched


def native_pack(tokenizer, retrieved, docs):
    chosen, text, count = [], "", 0
    for row in retrieved["chunks"]:
        candidate = "\n\n".join([text, row["content"]]).strip()
        tokens = len(tokenizer.tokenize(candidate))
        if tokens <= FULL_BUDGET:
            chosen.append(row)
            text, count = candidate, tokens
    return {"evidence": text, "evidence_tokens": count, "selected": source_spans(chosen, docs),
            "chunks": [row["id"] for row in chosen],
            "wiki_chunks": sum(bool(row.get("compile_kwd")) for row in chosen)}


def guarded_pack(tokenizer, b, captured, question, docs):
    allowed = source_spans(captured["chunks"], docs)
    provenance = source_spans([r for r in captured["chunks"] if r.get("compile_kwd")], docs)
    if not provenance:
        return {**b, "route": "fallback_no_verified_wiki"}
    titles = {d["id"]: title_surface(d["title"]) for d in docs}
    selected, _, _, events, guard = guarded_bridges(tokenizer, question, docs, titles, SHORT_BUDGET)
    selected = {i: set(s) & allowed[i] for i, s in selected.items() if set(s) & allowed[i]}
    pairs = {(i, j) for i, spans in selected.items() for j in spans}
    chain = chain_check(question, docs)
    evidence = sentence_blocks(docs, selected)
    count = len(tokenizer.tokenize(evidence))
    candidate = {"evidence_tokens": count, "guard": guard,
                 "selected_blocks": [[i, sorted(indices)] for i, indices in sorted(selected.items())]}
    decision = route(candidate, b, chain)
    if not chain or not chain <= pairs or not any(j in provenance[i] for i, j in chain):
        return {**b, "route": "fallback_no_supported_wiki_chain"}
    if set(guard["direct_pages"]) - selected.keys():
        return {**b, "route": "fallback_missing_named_page"}
    if decision != "short_checked_chain":
        return {**b, "route": decision}
    return {"evidence": evidence, "evidence_tokens": count, "selected": selected,
            "chunks": [], "wiki_chunks": 0, "route": decision, "bridge_events": events,
            "wiki_verified_chain": sorted(chain)}


def answer(reader, rows, query, docs, label):
    for row in rows:
        row.update(dataset=query["dataset"], query_id=query["id"], family_id=query["family_id"],
                   question=query["question"])
    for row in read_question(reader, rows, {}):
        selected = row.pop("selected")
        ref, strict, touched = support_sets(query["dataset"], query, docs, label, selected)
        em, f1 = official_answer(query["dataset"], row["answer"], label) if row["status"] == "ok" else (0.0, 0.0)
        row.update(em=em, f1=f1, selected_support_complete=ref <= strict,
                   all_support_touched=ref <= touched, spans=[[i, sorted(s)] for i, s in sorted(selected.items())])
    return rows


def summarize(rows):
    summary = {}
    for scope in ("all", "hotpot", "musique"):
        groups = {arm: {r["query_id"]: r for r in rows if r["method"] == arm
                        and (scope == "all" or r["dataset"] == scope)} for arm in ARMS}
        raw = groups[ARMS[0]]
        if not raw:
            continue
        ids = sorted(raw)
        output = {}
        for arm, group in groups.items():
            if group.keys() != raw.keys() or any(r["status"] != "ok" for r in group.values()):
                raise RuntimeError(f"Incomplete reader results for {scope}/{arm}")
            output[arm] = {**measure(group, ids), "n": len(ids),
                           "wiki_chunks": sum(group[q]["wiki_chunks"] for q in ids),
                           "output_tokens": sum(group[q]["api_completion_tokens"] for q in ids),
                           "search_ms_p50": median(group[q]["search_ms"] for q in ids),
                           "captured_plus_assembly_ms_p50": median(
                               group[q]["capture_ms"] + group[q].get("assembly_ms", 0) for q in ids),
                           "observed_pipeline_ms_p50": median(
                               group[q]["capture_ms"] + group[q].get("assembly_ms", 0) + group[q]["api_ms"] for q in ids),
                           "route": dict(Counter(group[q]["route"] for q in ids if "route" in group[q])),
                           "vs_A": difference(raw, group, ids)}
        output[ARMS[2]]["vs_B"] = difference(groups[ARMS[1]], groups[ARMS[2]], ids)
        summary[scope] = output
    return summary


def run(output, scope):
    _, corpus, queries = verify_freeze(SOURCE)
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique")
              for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    client = authenticate()
    state = json.loads((output / f"native-{scope}.json").read_text(encoding="utf-8"))
    artifacts = verify_ready(client, state)
    lineage = Lineage(client, state["dataset_id"])
    reader = CloudReader()
    path = output / f"native-{scope}-scored.jsonl"
    previous = read_jsonl(path) if path.exists() else []
    counts = Counter(r["query_id"] for r in previous)
    if set(counts.values()) - {len(ARMS)}:
        raise ValueError("Interrupted per-question row group")
    selected = [q for q in queries if q["id"] in state["questions"] and q["id"] not in counts]
    start = datetime.now(timezone.utc).isoformat()

    def evaluate(q):
        docs = [corpus[c["id"]] for c in q["candidates"]]
        include_first = bool(digest(q["id"].encode()).digest()[0] & 1)
        captures = {include: retrieve(client, lineage, state, q, include) for include in (include_first, not include_first)}
        a, b = captures[False], captures[True]
        tokenizer = BudgetTokenizer()
        begin = perf_counter()
        a_pack = native_pack(tokenizer, a, docs)
        a_ms = (perf_counter() - begin) * 1000
        begin = perf_counter()
        b_pack = native_pack(tokenizer, b, docs)
        b_ms = (perf_counter() - begin) * 1000
        begin = perf_counter()
        c_pack = guarded_pack(tokenizer, b_pack, b, q["question"], docs)
        c_ms = b_ms + (perf_counter() - begin) * 1000
        rows = [{**pack, "method": arm, "search_ms": retrieval["search_ms"], "capture_ms": retrieval["capture_ms"],
                 "assembly_ms": assembly}
                for pack, arm, retrieval, assembly in zip((a_pack, b_pack, c_pack), ARMS, (a, b, b),
                                                          (a_ms, b_ms, c_ms), strict=True)]
        return {"query_id": q["id"], "raw": a, "wiki": b}, answer(reader, rows, q, docs, labels[q["dataset"], q["id"]])

    with path.open("a", encoding="utf-8") as scored, (
            output / f"native-{scope}-retrieval.jsonl").open("a", encoding="utf-8") as captured, ThreadPoolExecutor(8) as pool:
        for retrieval, rows in pool.map(evaluate, selected):
            if any(r["status"] != "ok" for r in rows):
                raise RuntimeError(f"Reader failed for {retrieval['query_id']}")
            append(captured, retrieval)
            scored.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            scored.flush()
            previous.extend(rows)
            if len(previous) % 30 == 0 or len(previous) == len(state["questions"]) * len(ARMS):
                print(f"RAGFlow questions: {len(previous) // len(ARMS)}/{len(state['questions'])}", flush=True)
    end = datetime.now(timezone.utc).isoformat()
    usage = [r for r in relay_records(output) if start <= r["at"] <= end]
    embed = [r for r in usage if r["route"] == "/v1/embeddings" and r["status"] == 200]
    query_usage = {"successful_embeddings": len(embed), "paired_embedding_inputs": sum(r["usage"]["prompt_tokens"] for r in embed),
                   "other_calls": len(usage) - len(embed),
                   "complete_full_run": not counts and len(embed) == len(state["questions"]) * 2 and len(usage) == len(embed)}
    dump(output / f"native-{scope}-summary.json", {
        "start_utc": start, "end_utc": end, "wiki_artifacts": artifacts,
        "model": MODEL, "search": SEARCH_SETTINGS, "full_budget": FULL_BUDGET, "short_budget": SHORT_BUDGET,
        "script_sha256": sha256(Path(__file__)), "tokenizer_sha256": sha256(TOKENIZER),
        "results": summarize(previous), "online_query_embeddings": query_usage,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("scope", choices=("pilot", "full"))
    args = parser.parse_args()
    run(args.output, args.scope)
