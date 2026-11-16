"""Compile sources and capture the paired retrievals for the frozen cohorts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from external_compression import append
from freeze_scale import DATA, compiler
from prepare import read_jsonl
from run_abcd import write_json
from semantic_bridge_experiment import verify_freeze


LOGS = Path("logs")
USAGE = LOGS / "ragflow-model-usage.jsonl"


def final_relation_response(api):
    # The frozen compiler already emits empty records for unavailable responses.
    return api.get("status") == "ok" or api.get("error") == "HTTP 400 data_inspection_failed"


def compile_sources():
    config, corpus, _ = verify_freeze(DATA)
    cc = compiler()
    pages = sorted(corpus.values(), key=lambda doc: doc["id"])
    batches = [pages[i:i + cc.BATCH] for i in range(0, len(pages), cc.BATCH)]
    journal = LOGS / "relation-compilation.jsonl"
    records = read_jsonl(journal) if journal.exists() else []
    completed = {tuple(r["ids"]): r["rows"] for r in records if final_relation_response(r["api"])}
    pending = [batch for batch in batches if tuple(d["id"] for d in batch) not in completed]
    reader = cc.CloudReader()
    failures = 0
    with ThreadPoolExecutor(max_workers=20) as pool, journal.open("a", encoding="utf-8") as log:
        for i, (api, rows) in enumerate(pool.map(lambda docs: cc.compile_batch(reader, docs), pending), 1):
            ids = tuple(row["doc_id"] for row in rows)
            append(log, {"ids": ids, "api": api, "rows": rows})
            if final_relation_response(api):
                completed[ids] = rows
            else:
                failures += 1
            if i % 25 == 0 or i == len(pending):
                print(f"Offline relation batches {len(completed)}/{len(batches)}; failed this run: {failures}", flush=True)
            if failures >= 10:
                raise RuntimeError("Offline relation compilation encountered repeated API errors; inspect the journal")
    if failures or len(completed) != len(batches):
        raise RuntimeError("Offline relation compilation has unfinished batches; run compile again to resume")
    rows = [row for batch in batches for row in completed[tuple(d["id"] for d in batch)]]
    if len(rows) != config["pages"]:
        raise ValueError("Offline compiler did not cover the frozen corpus")
    path = DATA / "compiled.jsonl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            append(handle, row)
    tmp.replace(path)
    print(f"Relation compiler parsed {sum(row['parse'] == 'ok' for row in rows)}/{len(rows)} pages", flush=True)


def native_stage(relay):
    import ragflow_native_setup as setup
    from ragflow_native_wait import wait_for_index

    setup.SOURCE = DATA
    setup.NAMES = {**setup.NAMES, "full": "topic2-native-wiki-scale-500-20261116"}
    client = setup.authenticate()
    ids = setup.models(client, relay)
    setup.ingest(client, DATA, "full", setup.provision(client, DATA, ids))
    wait_for_index(DATA)


def capture():
    from ragflow_native_compare import Lineage, SEARCH_SETTINGS, retrieve, verify_ready
    from ragflow_native_setup import authenticate

    _, _, queries = verify_freeze(DATA)
    state = json.loads((DATA / "native-full.json").read_text(encoding="utf-8"))
    cohorts = json.loads((DATA / "cohorts.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in queries}
    if set(state["questions"]) != by_id.keys():
        raise ValueError("Native dataset and frozen sample differ")
    client = authenticate()
    artifacts = verify_ready(client, state)
    lineage = Lineage(client, state["dataset_id"])

    def capture_question(qid):
        q = by_id[qid]
        first = bool(sha256(qid.encode()).digest()[0] & 1)
        results = {flag: retrieve(client, lineage, state, q, flag) for flag in (first, not first)}
        return {"query_id": qid, "raw": results[False], "wiki": results[True]}

    snapshots, usage_by_cohort = [], {}
    for cohort, ids in cohorts.items():
        part = LOGS / f"capture-{cohort}.json"
        if part.exists():
            record = json.loads(part.read_text(encoding="utf-8"))
            if record["dataset_id"] != state["dataset_id"] or record["wiki_artifacts"] != artifacts:
                raise ValueError("Completed cohort was captured against a different native index")
        else:
            started = datetime.now(timezone.utc).isoformat()
            rows = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                for row in pool.map(capture_question, ids):
                    rows.append(row)
                    if len(rows) % 20 == 0 or len(rows) == len(ids):
                        print(f"Retrieval {cohort}: {len(rows)}/{len(ids)}", flush=True)
            ended = datetime.now(timezone.utc).isoformat()
            usage = [r for r in read_jsonl(USAGE) if started <= r["at"] <= ended]
            embeds = [r for r in usage if r["route"] == "/v1/embeddings" and r["status"] == 200]
            if len(embeds) != 2 * len(ids) or len(usage) != len(embeds):
                raise ValueError(f"Expected exactly {2 * len(ids)} query embeddings for {cohort}; got {len(embeds)}")
            record = {"dataset_id": state["dataset_id"], "wiki_artifacts": artifacts, "rows": rows,
                      "usage": {"successful_embeddings": len(embeds), "start_utc": started, "end_utc": ended,
                                "paired_embedding_inputs": sum(r["usage"]["prompt_tokens"] for r in embeds)}}
            tmp = part.with_suffix(".tmp")
            write_json(tmp, record)
            tmp.replace(part)
        if [r["query_id"] for r in record["rows"]] != ids:
            raise ValueError(f"Capture {cohort} differs from the frozen IDs")
        snapshots.extend(record["rows"])
        usage_by_cohort[cohort] = record["usage"]
    path = DATA / "native-full-retrieval.jsonl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in snapshots:
            append(handle, row)
    tmp.replace(path)
    write_json(DATA / "native-full-summary.json", {"wiki_artifacts": artifacts, "search": SEARCH_SETTINGS,
        "online_query_embeddings": {"paired_embedding_inputs": sum(r["paired_embedding_inputs"]
                                     for r in usage_by_cohort.values()), "by_cohort": usage_by_cohort}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("compile", "native", "capture"))
    parser.add_argument("--relay", default="http://172.18.96.1:17892/v1")
    args = parser.parse_args()
    if args.stage == "native":
        native_stage(args.relay)
    else:
        {"compile": compile_sources, "capture": capture}[args.stage]()
