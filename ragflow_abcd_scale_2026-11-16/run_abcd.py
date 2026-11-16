"""Compare one frozen RAGFlow retrieval snapshot with and without evidence compilation."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import platform
from statistics import mean
from time import sleep
import unicodedata

from baseline import SYSTEM, sentence_blocks
from cloud_experiment import BudgetTokenizer, TOKENIZER_REVISION
from cloud_reader import CloudReader, MODEL, ENDPOINT, MAX_OUTPUT
from external_compression import read_question
from paired import support_sets
from prepare import jsonl, read_jsonl
from raw_relation_locator import locate
from relation_gate import verified_facts
from score import official_answer
from scripts.check_paired import official_scores
from semantic_bridge_experiment import verify_freeze
from sweep_compare import difference
from wiki_bridge import span_text


DATA = Path("data")
RESULTS = Path("results")
CAPTURE = DATA / "native-full-retrieval.jsonl"
ARMS = ("A", "B", "C", "D")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def source_documents(query: dict, corpus: dict) -> list[dict]:
    return [corpus[row["id"]] for row in query["candidates"]]


def native_wiki_sources(capture: dict, docs: list[dict]) -> dict[str, set[str]]:
    return {row["id"]: {docs[i]["id"] for i in row["source_indices"]}
            for row in capture["wiki"]["chunks"] if row.get("compile_kwd")}


def build_index(captures: dict, queries: list[dict], corpus: dict, compiled: dict) -> dict:
    wiki_sources = {}
    for query in queries:
        docs = source_documents(query, corpus)
        for wiki_id, sources in native_wiki_sources(captures[query["id"]], docs).items():
            prior = wiki_sources.setdefault(wiki_id, sources)
            if prior != sources:
                raise ValueError("Native Wiki chunk has inconsistent provenance")
    index = {}
    for wiki_id, sources in sorted(wiki_sources.items()):
        index[wiki_id] = [{"doc_id": docid, "source_version": corpus[docid]["version"], "llm": facts}
                          for docid in sorted(sources) if (facts := verified_facts(corpus[docid], compiled))]
    jsonl(RESULTS / "wiki-evidence-index.jsonl", (
        {"wiki_chunk_id": wiki_id, "source_doc_ids": sorted(wiki_sources[wiki_id]), "sources": rows}
        for wiki_id, rows in index.items()
    ))
    return index


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def raw_sentences(chunks: list[dict], docs: list[dict]) -> dict[int, set[int]]:
    selected = defaultdict(set)
    for chunk in chunks:
        text = canonical(chunk["content"])
        for i in chunk["source_indices"]:
            for span in docs[i]["spans"]:
                j = span["index"]
                original = canonical(span_text(docs[i], j))
                if original and original in text:
                    selected[i].add(j)
    return dict(selected)


def baseline_pack(tokenizer: BudgetTokenizer, chunks: list[dict], docs: list[dict]) -> dict:
    evidence = "\n\n".join(row["content"] for row in chunks)
    return {"evidence": evidence, "evidence_tokens": len(tokenizer.tokenize(evidence)),
            "selected": raw_sentences(chunks, docs), "route": "baseline"}


def component_pack(tokenizer: BudgetTokenizer, query: dict, docs: list[dict], allowed: set, b: dict,
                   compiled: dict) -> dict:
    reason, proposed = locate(query["question"], docs, compiled)
    paths = {frozenset(p["spans"]) for p in proposed if p["spans"] <= allowed}
    if len(paths) != 1:
        return {**b, "route": "fallback", "locator": reason,
                "decision": "ambiguous" if len(paths) > 1 else "no_verified_path",
                "proposed_paths": len({frozenset(p["spans"]) for p in proposed})}
    path = next(iter(paths))
    selected = defaultdict(set)
    for i, j in path:
        selected[i].add(j)
    evidence = sentence_blocks(docs, selected)
    count = len(tokenizer.tokenize(evidence))
    if count >= b["evidence_tokens"]:
        return {**b, "route": "fallback", "locator": reason, "decision": "no_token_gain"}
    return {"evidence": evidence, "evidence_tokens": count, "selected": selected,
            "route": "selected", "locator": reason, "verified_spans": sorted(path)}


def plan_question(tokenizer: BudgetTokenizer, query: dict, docs: list[dict], capture: dict, index: dict) -> list[dict]:
    wiki_chunks = capture["wiki"]["chunks"]
    a = baseline_pack(tokenizer, capture["raw"]["chunks"], docs)
    b = baseline_pack(tokenizer, wiki_chunks, docs)
    wiki_sources = native_wiki_sources(capture, docs)
    matched_source_ids = set().union(*wiki_sources.values())
    allowed_indices = raw_sentences(wiki_chunks, docs)
    allowed = {(i, j) for i, indices in allowed_indices.items() for j in indices}
    # C and D can open the same source pages referenced by the native Wiki results.
    allowed.update((i, span["index"]) for i, doc in enumerate(docs) if doc["id"] in matched_source_ids
                   for span in doc["spans"])
    compiled = {row["doc_id"]: row for wiki_id in wiki_sources for row in index[wiki_id]}
    c = component_pack(tokenizer, query, docs, allowed, b, {})
    d = component_pack(tokenizer, query, docs, allowed, b, compiled)
    results = []
    for arm, pack in zip(ARMS, (a, b, c, d), strict=True):
        selected = pack["selected"]
        results.append({**pack, "selected": [[i, sorted(indices)] for i, indices in sorted(selected.items())],
                        "method": arm, "dataset": query["dataset"], "query_id": query["id"],
                        "family_id": query["family_id"], "question": query["question"],
                        "retrieval_chunk_ids": [row["id"] for row in (capture["raw"]["chunks"] if arm == "A" else wiki_chunks)],
                        "wiki_index_hits": len(compiled) if arm == "D" else 0})
    return results


def prepare() -> None:
    freeze, corpus, queries = verify_freeze(DATA)
    captured_rows = read_jsonl(CAPTURE)
    captures = {r["query_id"]: r for r in captured_rows}
    if len(captured_rows) != len(queries) or set(captures) != {q["id"] for q in queries}:
        raise ValueError("Retrieval snapshot differs from the frozen cohort")
    compiled = {r["doc_id"]: r for r in read_jsonl(DATA / "compiled.jsonl")}
    index = build_index(captures, queries, corpus, compiled)
    tokenizer = BudgetTokenizer()
    jsonl(RESULTS / "requests.jsonl", (
        request for q in queries
        for request in plan_question(tokenizer, q, source_documents(q, corpus), captures[q["id"]], index)
    ))
    inputs = [*(DATA / name for name in freeze["input_sha256"]), DATA / "freeze.json", DATA / "compile-source.py",
              CAPTURE, DATA / "compiled.jsonl", DATA / "native-full-summary.json", DATA / "native-full.json",
              DATA / "native-pipeline.json", *sorted(Path(".").glob("*.py")), RESULTS / "requests.jsonl",
              RESULTS / "wiki-evidence-index.jsonl", Path("scripts/check_paired.py"),
              Path("vendor/tokenizer/qwen3.5.json")]
    write_json(RESULTS / "config.json", {
        "questions": len(queries), "retrieval": "RAGFlow v1.0.0 frozen final index replay",
        "short_budget": None, "baseline_evidence": "all captured chunks in retrieval order",
        "model": MODEL, "endpoint": ENDPOINT, "system": SYSTEM, "max_output": MAX_OUTPUT,
        "temperature": 0, "thinking": False, "tokenizer_revision": TOKENIZER_REVISION, "python": platform.python_version(),
        "wiki_index_chunks": len(index), "wiki_index_with_facts": sum(bool(rows) for rows in index.values()),
        "wiki_index_doc_bindings": sum(len(rows) for rows in index.values()),
        "sha256": {path.as_posix(): digest(path) for path in inputs if path.is_file()},
    })


def answer_group(rows: list[dict], query: dict, docs: list[dict], label: dict) -> list[dict]:
    answered = read_question(CloudReader(), [dict(row) for row in rows], {})
    for row in answered:
        selected = {i: set(indices) for i, indices in row["selected"]}
        ref, strict, touched = support_sets(row["dataset"], query, docs, label, selected)
        row["em"], row["f1"] = official_answer(row["dataset"], row["answer"], label) if row["status"] == "ok" else (0, 0)
        row["strict_support"] = ref <= strict
        row["support_touched"] = ref <= touched
    return answered


def score_all(queries: list[dict], corpus: dict, labels: dict) -> list[dict]:
    plans = defaultdict(list)
    for row in read_jsonl(RESULTS / "requests.jsonl"):
        plans[row["query_id"]].append(row)
    file = RESULTS / "scored.jsonl"
    prior = read_jsonl(file) if file.exists() else []
    complete = defaultdict(set)
    for row in prior:
        if row["status"] == "ok":
            complete[row["query_id"]].add(row["method"])
    prior = [row for row in prior if complete[row["query_id"]] == set(ARMS)]
    checkpoint = file.with_suffix(".tmp")
    if checkpoint.exists():
        checkpoint.unlink()
    jsonl(checkpoint, prior)
    checkpoint.replace(file)
    pending = [q for q in queries if complete[q["id"]] != set(ARMS)]
    def evaluate(q):
        return answer_group(plans[q["id"]], q, source_documents(q, corpus), labels[q["dataset"]][q["id"]])
    with ThreadPoolExecutor(max_workers=6) as pool, file.open("a", encoding="utf-8") as log:
        for rows in pool.map(evaluate, pending):
            prior.extend(rows)
            log.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            log.flush()
            if len(prior) % 40 == 0:
                print(f"{len(prior)}/{len(queries) * len(ARMS)} scored", flush=True)
    return prior


def summarize(rows: list[dict], labels: dict, ids: set[str], retrieval_tokens: int) -> dict:
    if len(rows) != len(ids) * len(ARMS) or {(r["query_id"], r["method"]) for r in rows} != {
            (qid, arm) for qid in ids for arm in ARMS} or any(r["status"] != "ok" for r in rows):
        raise ValueError("Expected one successful score per frozen question and arm")
    groups = {arm: {r["query_id"]: r for r in rows if r["method"] == arm} for arm in ARMS}
    for qid in ids:
        candidate = groups["B"][qid]["retrieval_chunk_ids"]
        if any(groups[arm][qid]["retrieval_chunk_ids"] != candidate for arm in ("C", "D")):
            raise ValueError(f"B/C/D candidate ordering differs for {qid}")
    results = {}
    for arm, group in groups.items():
        arm_rows = list(group.values())
        for ds, ds_labels in labels.items():
            subset = [r for r in arm_rows if r["dataset"] == ds]
            subset_labels = {r["query_id"]: ds_labels[r["query_id"]] for r in subset}
            for key, official in zip(("em", "f1"), official_scores(ds, subset, subset_labels), strict=True):
                score = mean(r[key] for r in subset)
                if not isclose(official, round(score, 3) if ds == "musique" else score, abs_tol=1e-12):
                    raise ValueError(f"Official {ds}/{arm}/{key} mismatch")
        prompt = sum(r["api_prompt_tokens"] for r in arm_rows)
        output = sum(r["api_completion_tokens"] for r in arm_rows)
        results[arm] = {"f1": mean(r["f1"] for r in arm_rows), "em": mean(r["em"] for r in arm_rows),
                        "strict_support": sum(r["strict_support"] for r in arm_rows),
                        "support_touched": sum(r["support_touched"] for r in arm_rows),
                        "reader_input": prompt, "reader_output": output,
                        "by_dataset": {ds: {"questions": len(part), "f1": mean(r["f1"] for r in part),
                            "em": mean(r["em"] for r in part), "reader_input": sum(r["api_prompt_tokens"] for r in part)}
                            for ds in labels if (part := [r for r in arm_rows if r["dataset"] == ds])},
                        "retrieval_input_archived": retrieval_tokens, "total_tokens": retrieval_tokens + prompt + output,
                        "routes": dict(Counter(r["route"] for r in arm_rows))}
    comparisons = {}
    for a, b in (("B", "A"), ("C", "B"), ("D", "C"), ("D", "B"), ("C", "A"), ("D", "A")):
        common = sorted(groups[a])
        diff = results[a]["total_tokens"] - results[b]["total_tokens"]
        comparisons[f"{a}_vs_{b}"] = {
            **difference(groups[b], groups[a], common), "total_token_delta": diff,
            "total_token_delta_rate": diff / results[b]["total_tokens"],
            "em_gained": sum(groups[a][q]["em"] > groups[b][q]["em"] for q in common),
            "em_lost": sum(groups[a][q]["em"] < groups[b][q]["em"] for q in common),
            "changed_query_ids": [q for q in common if groups[a][q]["evidence"] != groups[b][q]["evidence"]],
        }
    actual = [r for r in rows if not r["shared_api_result"]]
    return {"arms": results, "comparisons": comparisons, "actual_reader_calls": len(actual),
            "actual_reader_attempts": sum(r["api_attempts"] for r in actual),
            "actual_input": sum(r["api_prompt_tokens"] for r in actual),
            "actual_output": sum(r["api_completion_tokens"] for r in actual)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    if parser.parse_args().stage == "prepare":
        prepare()
        return
    config = json.loads((RESULTS / "config.json").read_text(encoding="utf-8"))
    for path, expected in config["sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    _, corpus, queries = verify_freeze(DATA)
    labels = {ds: {r["id"]: r for r in read_jsonl(DATA / f"labels-{ds}.jsonl")} for ds in ("hotpot", "musique")}
    for attempt in range(3):
        scores = score_all(queries, corpus, labels)
        if all(r["status"] == "ok" for r in scores):
            break
        if attempt < 2:
            print("Retrying incomplete reader groups from the scored checkpoint", flush=True)
            sleep(20)
    cohorts = json.loads((DATA / "cohorts.json").read_text(encoding="utf-8"))
    snapshot = json.loads((DATA / "native-full-summary.json").read_text(encoding="utf-8"))
    counts = snapshot["online_query_embeddings"]["by_cohort"]
    cohorts["500"] = [q["id"] for q in queries]
    counts["500"] = {"paired_embedding_inputs": sum(r["paired_embedding_inputs"] for r in counts.values())}
    report = {}
    for cohort, members in cohorts.items():
        ids = set(members)
        tokens, remainder = divmod(counts[cohort]["paired_embedding_inputs"], 2)
        if remainder:
            raise ValueError(f"Paired query embedding token count is uneven for {cohort}")
        report[cohort] = summarize([r for r in scores if r["query_id"] in ids], labels, ids, tokens)
    write_json(RESULTS / "summary.json", report)


if __name__ == "__main__":
    main()
