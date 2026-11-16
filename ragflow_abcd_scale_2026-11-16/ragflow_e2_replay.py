"""Evaluate source-compiled E2-style short evidence on the frozen development pool."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

from baseline import words
from cloud_experiment import BudgetTokenizer, TOKENIZER
from cloud_reader import CloudReader, MODEL, ReaderError
from dependency_development import SOURCE
from external_compression import append, dump, pack_ranked
from intent_evidence_experiment import sha256
from paired import score_row
from prepare import read_jsonl
from semantic_bridge_experiment import verify_freeze
from sweep_compare import difference, measure


BUDGET = 1024
WORKERS = 10
MAX_OUTPUT = 192
SYSTEM = """Decide whether the supplied evidence fully supports a short answer to the question.
For multi-step questions, check each necessary relation and entity identity, including qualifiers.
Use only the supplied evidence. When a required relation or qualifier is missing, set answerable to false.
Return JSON only: {"answerable":true,"answer":"short answer"} or {"answerable":false,"answer":""}."""


def prepare_packs(corpus, queries, claims):
    tokenizer = BudgetTokenizer()
    for query in queries:
        terms = words(query["question"])
        docs = [corpus[c["id"]] for c in query["candidates"]]
        candidates, ranks = [], []
        for index, doc in enumerate(docs):
            title_match = len(terms & words(doc["title"]))
            by_span = {}
            for fact in claims[doc["id"]]:
                quote_match = len(terms & words(fact["quote"]))
                claim_match = len(terms & words(fact["claim"]))
                if not (quote_match or claim_match or title_match):
                    continue
                key = (-max(quote_match, claim_match), -title_match, -quote_match)
                span = fact["sentence"]
                if span not in by_span or key < by_span[span][0]:
                    by_span[span] = key, fact["quote"]
            for span, (key, quote) in sorted(by_span.items()):
                ranks.append((*key, index, span, len(candidates)))
                candidates.append((index, span, quote))
        selected, evidence, tokens, _ = pack_ranked(
            tokenizer, docs, candidates, [rank[-1] for rank in sorted(ranks)], BUDGET
        )
        yield {"dataset": query["dataset"], "query_id": query["id"], "family_id": query["family_id"],
               "question": query["question"], "method": "e2_short", "budget": BUDGET,
               "candidate_ids": [d["id"] for d in docs], "evidence": evidence, "evidence_tokens": tokens,
               "selected_blocks": [[i, sorted(spans)] for i, spans in sorted(selected.items())]}


def execute(reader, pack, query, corpus, label, raw):
    api = {}
    route = "short"
    if pack["evidence"]:
        try:
            api = reader.generate([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Evidence:\n{pack['evidence']}\n\nQuestion: {pack['question']}"},
            ], max_output=MAX_OUTPUT, json_object=True)
            decision = json.loads(api["answer"])
            if type(decision["answerable"]) is not bool or not isinstance(decision["answer"], str):
                route = "invalid_gate"
            elif not decision["answerable"]:
                route = "insufficient"
            elif not decision["answer"].strip():
                route = "invalid_gate"
        except ReaderError as error:
            api = {"error": str(error)}
            route = "api_error"
        except (KeyError, TypeError, ValueError):
            route = "invalid_gate"
    else:
        route = "empty_short"
    short = score_row({**pack, "answer": decision["answer"].strip() if route == "short" else "",
                       "status": "ok" if route == "short" else route}, query, corpus, label)
    if route == "short":
        final = {**short, **api, "answer": short["answer"]}
    else:
        final = dict(raw)
        for name in ("api_prompt_tokens", "api_completion_tokens", "list_price_cny"):
            final[name] += api.get(name, 0)
    final.update(method="e2_gated", route=route, logical_calls=int(bool(pack["evidence"])) + int(route != "short"),
                 short=short, gate_api=api)
    return final


def metrics(raw, results):
    output = {}
    for scope in ("all", "hotpot", "musique"):
        selected = {q: r for q, r in results.items() if scope == "all" or r["dataset"] == scope}
        if not selected:
            continue
        keys = sorted(selected)
        reference = measure(raw, keys)
        treatment = measure(selected, keys)
        output[scope] = {
            "questions": len(keys), "raw": reference, "e2": treatment,
            "raw_completion_total": sum(raw[q]["api_completion_tokens"] for q in keys),
            "e2_completion_total": sum(selected[q]["api_completion_tokens"] for q in keys),
            "raw_price_cny": sum(raw[q]["list_price_cny"] for q in keys),
            "e2_price_cny": sum(selected[q]["list_price_cny"] for q in keys),
            "short_full_support": sum(r["short"]["selected_support_complete"] for r in selected.values()),
            "short_f1": mean(r["short"]["f1"] for r in selected.values()),
            "routes": dict(Counter(r["route"] for r in selected.values())),
            "e2_logical_calls": sum(r["logical_calls"] for r in selected.values()),
            "input_saving": 1 - treatment["api_prompt_total"] / reference["api_prompt_total"],
            "paired_f1": difference(raw, selected, keys),
        }
    return output


def run(output: Path):
    _, corpus, queries = verify_freeze(SOURCE)
    source = read_jsonl(output / "compile-api.jsonl")
    claims = defaultdict(list)
    for batch in source:
        if batch["status"] != "ok":
            continue
        for doc in batch["documents"]:
            key = doc["doc_id"]
            if corpus[key]["version"] != doc["source_version"] or key in claims:
                raise ValueError("Duplicate source page or inconsistent version")
            claims[key] = []
        for claim in batch["claims"]:
            claims[claim["doc_id"]].append(claim)
    if claims.keys() != corpus.keys():
        raise ValueError(f"Compilation incomplete: {len(claims)}/{len(corpus)} pages")
    manifest = output / "e2-config.json"
    config = {"model": MODEL, "budget": BUDGET, "system": SYSTEM, "max_output": MAX_OUTPUT,
              "question_scope": "previously_evaluated_development", "questions": len(queries),
              "compile_sha256": sha256(output / "compile-api.jsonl"), "tokenizer_sha256": sha256(TOKENIZER),
              "source_freeze_sha256": sha256(SOURCE / "freeze.json"),
              "archive_sha256": sha256(SOURCE / "scored.jsonl"), "script_sha256": sha256(Path(__file__))}
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != config:
        raise ValueError("Replay configuration changed")
    dump(manifest, config)
    raw = {r["query_id"]: r for r in read_jsonl(SOURCE / "scored.jsonl") if r["method"] == "raw_full"}
    assert len(raw) == len(queries) == 100
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique")
              for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    path = output / "e2-scored.jsonl"
    finished = {r["query_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    pending = [q for q in queries if q["id"] not in finished]
    reader = CloudReader()
    packs = list(prepare_packs(corpus, pending, claims))
    with path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(execute, reader, p, q, corpus, labels[q["dataset"], q["id"]], raw[q["id"]])
                   for p, q in zip(packs, pending, strict=True)]
        for future in futures:
            row = future.result()
            append(log, row)
            finished[row["query_id"]] = row
            if len(finished) % 10 == 0:
                print(f"E2 questions {len(finished)}/{len(queries)}", flush=True)
    dump(output / "e2-results.json", {"end_utc": datetime.now(timezone.utc).isoformat(),
                                     "comparison": metrics(raw, finished)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
