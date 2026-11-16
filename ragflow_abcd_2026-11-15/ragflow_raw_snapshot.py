"""Measure the native raw-only arm once all source chunks are indexed."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from time import sleep

from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader
from dependency_development import SOURCE
from external_compression import append, dump
from prepare import read_jsonl
from ragflow_native_compare import Lineage, answer, native_pack, relay_records, retrieve
from ragflow_native_e2 import run as run_e2
from ragflow_native_setup import authenticate, documents
from semantic_bridge_experiment import verify_freeze
from sweep_compare import measure


def run(output):
    _, corpus, queries = verify_freeze(SOURCE)
    state = json.loads((output / "native-full.json").read_text(encoding="utf-8"))
    client = authenticate()
    dataset = state["dataset_id"]
    while True:
        rows = list(documents(client, dataset))
        if {r["id"] for r in rows} != set(state["documents"].values()):
            raise ValueError("Source index document list does not match the frozen upload")
        statuses = Counter(r["ingestion_status"] for r in rows)
        if statuses == {"COMPLETED": len(rows)}:
            break
        sleep(20)
    before = client.call("GET", f"/datasets/{dataset}/compilation/status")
    start = datetime.now(timezone.utc).isoformat()
    lineage = Lineage(client, dataset)
    reader = CloudReader()
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique")
              for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    base = output / "native-raw-snapshot"
    scored_path = base.with_name(base.name + "-scored.jsonl")
    retrieved_path = base.with_name(base.name + "-retrieval.jsonl")
    finished = {r["query_id"]: r for r in read_jsonl(scored_path)} if scored_path.exists() else {}

    def evaluate(query):
        docs = [corpus[c["id"]] for c in query["candidates"]]
        retrieval = retrieve(client, lineage, state, query, False)
        row = {**native_pack(BudgetTokenizer(), retrieval, docs), "method": "A_raw",
               "search_ms": retrieval["search_ms"], "capture_ms": retrieval["capture_ms"]}
        scored = answer(reader, [row], query, docs, labels[query["dataset"], query["id"]])[0]
        if scored["status"] != "ok":
            raise RuntimeError(f"Reader failed for {query['id']}")
        return retrieval, scored

    pending = [q for q in queries if q["id"] not in finished]
    with scored_path.open("a", encoding="utf-8") as scored, retrieved_path.open(
            "a", encoding="utf-8") as captured, ThreadPoolExecutor(8) as pool:
        for retrieval, row in pool.map(evaluate, pending):
            append(captured, {"query_id": row["query_id"], "raw": retrieval})
            append(scored, row)
            finished[row["query_id"]] = row
    end = datetime.now(timezone.utc).isoformat()
    after = client.call("GET", f"/datasets/{dataset}/compilation/status")
    measures = {}
    for scope in ("all", "hotpot", "musique"):
        ids = sorted(q for q, row in finished.items() if scope in ("all", row["dataset"]))
        measures[scope] = {**measure(finished, ids), "questions": len(ids),
                           "search_ms_p50": median(finished[q]["search_ms"] for q in ids)}
    usage = [r for r in relay_records(output) if start <= r["at"] <= end]
    dump(base.with_name(base.name + "-summary.json"),
         {"start_utc": start, "end_utc": end, "wiki_merge_start": before["state"], "wiki_merge_end": after["state"],
          "model_relay_background_calls": len(usage), "results": measures})
    run_e2(output, "raw-snapshot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
