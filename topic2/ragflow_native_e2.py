"""Evaluate the E2 adaptation on the raw chunks localized by RAGFlow."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from cloud_reader import CloudReader
from dependency_development import SOURCE
from external_compression import append, dump
from intent_evidence_experiment import sha256
from prepare import read_jsonl
from ragflow_e2_replay import execute, metrics, prepare_packs
from ragflow_native_compare import source_spans
from semantic_bridge_experiment import verify_freeze


def run(output, scope):
    _, corpus, queries = verify_freeze(SOURCE)
    compiled = defaultdict(list)
    for batch in read_jsonl(output / "compile-api.jsonl"):
        if batch["status"] != "ok":
            continue
        for claim in batch["claims"]:
            if corpus[claim["doc_id"]]["version"] != claim["source_version"]:
                raise ValueError("E2 claim version does not match its source")
            compiled[claim["doc_id"]].append(claim)
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique")
              for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    raw_path = output / f"native-{scope}-scored.jsonl"
    retrieval_path = output / f"native-{scope}-retrieval.jsonl"
    raw = {r["query_id"]: r for r in read_jsonl(raw_path) if r["method"] == "A_raw"}
    captures = {r["query_id"]: r["raw"]["chunks"] for r in read_jsonl(retrieval_path)}
    if set(raw) != set(captures) or not all(r["status"] == "ok" for r in raw.values()):
        raise ValueError("Missing paired RAGFlow A inputs or responses")
    path = output / f"native-{scope}-e2-scored.jsonl"
    finished = {r["query_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    reader = CloudReader()

    def evaluate(q):
        qid = q["id"]
        docs = [corpus[c["id"]] for c in q["candidates"]]
        hits = source_spans(captures[qid], docs)
        scoped = {d["id"]: [r for r in compiled.get(d["id"], []) if r["sentence"] in hits[i]]
                  for i, d in enumerate(docs)}
        pack = next(prepare_packs(corpus, [q], scoped))
        return execute(reader, pack, q, corpus, labels[q["dataset"], qid], raw[qid])

    pending = [q for q in queries if q["id"] in raw and q["id"] not in finished]
    with path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(8) as pool:
        for row in pool.map(evaluate, pending):
            append(log, row)
            finished[row["query_id"]] = row
            if len(finished) % 10 == 0 or len(finished) == len(raw):
                print(f"RAGFlow E2 questions: {len(finished)}/{len(raw)}", flush=True)
    if finished.keys() != raw.keys():
        raise ValueError("Incomplete paired RAGFlow E2 result")
    dump(output / f"native-{scope}-e2-summary.json", {
        "end_utc": datetime.now(timezone.utc).isoformat(), "comparison": metrics(raw, finished),
        "sha256": {p.name: sha256(p) for p in (raw_path, retrieval_path, output / "compile-api.jsonl", Path(__file__))},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("scope", choices=("pilot", "full"))
    args = parser.parse_args()
    run(args.output, args.scope)
