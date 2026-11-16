"""Score the four arms on a stable intermediate Wiki index, with a separate report."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import sleep
from urllib.request import Request

from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader
from dependency_development import SOURCE
from external_compression import append, dump
from prepare import jsonl, read_jsonl
from ragflow_native_compare import ARMS, Lineage, answer, guarded_pack, native_pack, retrieve, summarize
from ragflow_native_e2 import run as compare_e2
from ragflow_native_setup import authenticate
from semantic_bridge_experiment import verify_freeze


def index_generation(lineage):
    url = lineage.url.rsplit("/", 1)[0] + "/_stats?level=shards&filter_path=indices.*.shards.*.seq_no"
    request = Request(url, headers=lineage.headers)
    with lineage.opener.open(request, timeout=15) as response:
        return json.load(response)["indices"]


def capture(client, lineage, state, queries):
    dataset = state["dataset_id"]
    status_path = f"/datasets/{dataset}/compilation/status"
    artifacts_path = f"/datasets/{dataset}/artifacts"

    def one(q):
        return {"query_id": q["id"], "raw": retrieve(client, lineage, state, q, False),
                "wiki": retrieve(client, lineage, state, q, True)}

    for attempt in range(4):
        stage = client.call("GET", status_path)
        if stage["current_phase"] in ("filtering_candidates", "completed"):
            start = datetime.now(timezone.utc).isoformat()
            artifacts = client.call("GET", artifacts_path, params={"page_size": 1})["total"]
            first = index_generation(lineage)
            with ThreadPoolExecutor(8) as pool:
                rows = list(pool.map(one, queries))
            last = index_generation(lineage)
            if first == last:
                return rows, {"retrieval_start_utc": start, "retrieval_end_utc": datetime.now(timezone.utc).isoformat(),
                              "wiki_artifacts": artifacts, "wiki_merge_status": stage, "index_generation": first}
        if attempt < 3:
            sleep(20)
    raise RuntimeError("No stable intermediate Wiki index across all retrievals")


def run(output):
    _, corpus, queries = verify_freeze(SOURCE)
    state = json.loads((output / "native-full.json").read_text(encoding="utf-8"))
    client = authenticate()
    lineage = Lineage(client, state["dataset_id"])
    records, config = capture(client, lineage, state, queries)
    dump(output / "native-checkpoint.json", state)
    dump(output / "native-checkpoint-config.json", config)
    jsonl(output / "native-checkpoint-retrieval.jsonl", records)
    print(f"Captured 100 questions on {config['wiki_artifacts']} Wiki artifacts", flush=True)
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique")
              for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    reader = CloudReader()
    rows = []

    def score(inputs):
        query, captured = inputs
        assert query["id"] == captured["query_id"]
        docs = [corpus[c["id"]] for c in query["candidates"]]
        a, b = captured["raw"], captured["wiki"]
        tokenizer = BudgetTokenizer()
        raw, wiki = (native_pack(tokenizer, r, docs) for r in (a, b))
        guarded = guarded_pack(tokenizer, wiki, b, query["question"], docs)
        requests = [{**pack, "method": name, "search_ms": retrieval["search_ms"], "capture_ms": retrieval["capture_ms"]}
                    for pack, name, retrieval in zip((raw, wiki, guarded), ARMS, (a, b, b), strict=True)]
        return answer(reader, requests, query, docs, labels[query["dataset"], query["id"]])

    with (output / "native-checkpoint-scored.jsonl").open("w", encoding="utf-8") as log, ThreadPoolExecutor(8) as pool:
        for group in pool.map(score, zip(queries, records, strict=True)):
            for row in group:
                append(log, row)
            rows.extend(group)
    dump(output / "native-checkpoint-summary.json", {**config, "results": summarize(rows)})
    compare_e2(output, "checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
