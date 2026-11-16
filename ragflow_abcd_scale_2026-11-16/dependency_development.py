"""Replay the dependency packer on the 100 previously evaluated questions."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean

from cloud_experiment import BudgetTokenizer, TOKENIZER
from cloud_reader import CloudReader, MODEL
from external_compression import dump, read_question
from paired import score_row
from prepare import jsonl, read_jsonl
from semantic_bridge_experiment import BUDGETS, SOURCE_FILES, packs, verify_freeze
from wiki_bridge import span_text


SOURCE = Path("runs/semantic-bridge-heldout100")


def spans(row: dict) -> set[tuple[int, int]]:
    return {(doc, index) for doc, indices in row["selected_blocks"] for index in indices}


def describe(docs: list[dict], items: set[tuple[int, int]]) -> list[dict]:
    return [{"page": docs[doc]["title"], "version": docs[doc]["version"], "span": index,
             "text": span_text(docs[doc], index)} for doc, index in sorted(items)]


def prepare(output: Path) -> None:
    frozen, corpus, queries = verify_freeze(SOURCE)
    previous = read_jsonl(SOURCE / "scored.jsonl")
    old = {(r["dataset"], r["query_id"], r["method"], r["budget"]): r for r in previous}
    ranks = {(r["dataset"], r["query_id"]): r for r in read_jsonl(SOURCE / "rankings.jsonl")}
    compiled = {r["doc_id"]: r for r in read_jsonl(SOURCE / "compiled.jsonl")}
    assert compiled.keys() == corpus.keys()
    assert all(row["source_version"] == corpus[doc_id]["version"] for doc_id, row in compiled.items())

    output.mkdir()
    tokenizer, rows, changes = BudgetTokenizer(), [], []
    for query in queries:
        key = query["dataset"], query["id"]
        docs = [corpus[c["id"]] for c in query["candidates"]]
        facts = [compiled[d["id"]]["llm"] for d in docs]
        baseline, _ = packs(tokenizer, query, docs, facts, archived_rank=ranks[key])
        treatment, _ = packs(tokenizer, query, docs, facts, dependency_extension=True, archived_rank=ranks[key])
        for condition, row in baseline.items():
            assert row["evidence"] == old[key + condition]["evidence"], (key, condition)
            if condition[0] != "semantic_wiki":
                assert row == treatment[condition]
        for budget in BUDGETS:
            before = baseline["semantic_wiki", budget]
            after = treatment["semantic_wiki", budget]
            rows.append(after)
            events = [event for event in after["wiki_edges"] if event["kind"] == "dependency"]
            if events:
                changes.append({"dataset": key[0], "query_id": key[1], "question": query["question"], "budget": budget,
                                "previous_tokens": before["evidence_tokens"], "new_tokens": after["evidence_tokens"],
                                "changed": before["evidence"] != after["evidence"], "events": events,
                                "added": describe(docs, spans(after) - spans(before)),
                                "removed": describe(docs, spans(before) - spans(after))})
    jsonl(output / "packs.jsonl", rows)
    jsonl(output / "changes.jsonl", changes)

    snapshots = output / "source"
    paths = {"dependency_development.py", *SOURCE_FILES}
    for path in sorted(paths):
        destination = snapshots / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    source_paths = [SOURCE / name for name in ("compiled.jsonl", "rankings.jsonl", "scored.jsonl", "freeze.json")]
    hashed = [*source_paths, output / "packs.jsonl", output / "changes.jsonl", TOKENIZER,
              *(snapshots / path for path in sorted(paths))]
    dump(output / "config.json", {"scope": "development_replay", "questions": len(queries), "model": MODEL,
                                 "budgets": BUDGETS, "archived_rerank_model": "qwen3-rerank",
                                 "prepared_utc": datetime.now(timezone.utc).isoformat(),
                                 "input_hashes": frozen["input_sha256"],
                                 "sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in hashed},
                                 "baseline_conditions_reproduced": len(old),
                                 "changed_packs": sum(row["changed"] for row in changes),
                                 "decisions": dict(Counter(event["decision"] for row in changes for event in row["events"]))})


def run(output: Path) -> None:
    if (output / "scored.jsonl").exists():
        raise FileExistsError(output / "scored.jsonl")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for name, digest in config["sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Development replay input changed: {name}")
    _, corpus, queries = verify_freeze(SOURCE)
    old = read_jsonl(SOURCE / "scored.jsonl")
    by_query, planned = defaultdict(list), defaultdict(list)
    for row in old:
        by_query[row["dataset"], row["query_id"]].append(row)
    for row in read_jsonl(output / "packs.jsonl"):
        planned[row["dataset"], row["query_id"]].append(row)
    reader = CloudReader()
    scored, results = [], defaultdict(list)
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique") for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    for query in queries:
        key = query["dataset"], query["id"]
        previous = by_query[key]
        prior = {r["evidence"]: {k: v for k, v in r.items()
                                  if k in {"answer", "status", "finish_reason", "list_price_cny"} or k.startswith("api_")}
                 for r in previous if r["status"] == "ok"}
        answered = read_question(reader, planned[key], prior)
        for row in previous:
            results[row["dataset"], row["budget"], row["method"]].append(row)
        for row in answered:
            result = score_row({**row, "method": "semantic_dependency"}, query, corpus, labels[key])
            scored.append(result)
            results[row["dataset"], row["budget"], result["method"]].append(result)
    jsonl(output / "scored.jsonl", scored)
    summary = {f"{ds}/{budget}/{method}": {
        "n": len(rows), "ok": sum(r["status"] == "ok" for r in rows), "em": mean(r["em"] for r in rows),
        "f1": mean(r["f1"] for r in rows), "all_support_touched": sum(r["all_support_touched"] for r in rows),
        "all_support_complete": sum(r["selected_support_complete"] for r in rows),
        "evidence_tokens": sum(r["evidence_tokens"] for r in rows),
    } for (ds, budget, method), rows in sorted(results.items())}
    dump(output / "comparison.json", {"development_only": True, "methods": summary,
                                      "new_reader_requests": sum(not r["shared_api_result"] for r in scored),
                                      "new_reader_cost_cny": sum(r.get("list_price_cny", 0) for r in scored),
                                      "end_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)
