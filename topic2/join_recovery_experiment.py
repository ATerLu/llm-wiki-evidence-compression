"""Paired 100-question development replay of cross-page evidence connection recovery."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from statistics import mean

from baseline import SYSTEM as READER_SYSTEM
from cloud_experiment import BudgetTokenizer, FULL_LIMIT, TOKENIZER
from cloud_reader import CloudReader, MAX_OUTPUT as READER_LIMIT, MODEL
from dependency_development import SOURCE, describe, spans
from evidence_join_recovery import recover
from external_compression import append, dump, pack_ranked, read_question, sentence_candidates
from intent_evidence_experiment import MAX_OUTPUT, SYSTEM, select, selector_input, sha256
from mechanism_development import INTENT, ORACLE, packs
from paired import score_row
from prepare import jsonl, read_jsonl
from scripts.probe_question_intent import MAX_OUTPUT as PLAN_LIMIT, SYSTEM as PLAN_SYSTEM, query_plan
from semantic_bridge_experiment import BUDGETS, request, verify_freeze
from sweep_compare import difference


MECHANISM = Path("runs/mechanism-development-20261111")
METHODS = ("direct", "raw_recovery", "wiki_recovery")
PROPOSER = "intent_raw"
WORKERS = 8


def key(row: dict) -> tuple[str, str]:
    return row["dataset"], row.get("id", row.get("query_id"))


def prepare(output: Path) -> None:
    _, _, queries = verify_freeze(SOURCE)
    if Counter(q["dataset"] for q in queries) != {"hotpot": 50, "musique": 50}:
        raise ValueError("Expected the full archived 50+50 sample")
    for path in (INTENT, MECHANISM):
        archive = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if (archive["model"], archive["system"], archive["max_output"]) != (MODEL, SYSTEM, MAX_OUTPUT):
            raise ValueError(f"Archived selector configuration differs: {path}")
    archive = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    if (archive["model"], archive["system"], archive["max_output"], archive["temperature"], archive["thinking"]) != (
            MODEL, READER_SYSTEM, READER_LIMIT, 0, False):
        raise ValueError("Archived reader configuration differs")
    output.mkdir()
    code = [Path(name) for name in ("join_recovery_experiment.py", "evidence_join_recovery.py", "intent_evidence.py",
                                    "intent_evidence_experiment.py", "mechanism_development.py", "semantic_bridge.py",
                                    "external_compression.py", "cloud_reader.py", "baseline.py", "paired.py", "prepare.py",
                                    "wiki_bridge.py", "wiki_compile_v2.py", "cloud_experiment.py",
                                    "semantic_bridge_experiment.py", "dependency_development.py", "sweep_compare.py",
                                    "scripts/probe_question_intent.py", "scripts/check_paired.py",
                                    "scripts/check_join_recovery.py")]
    snapshots = output / "source"
    for path in code:
        destination = snapshots / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    frozen = [*(SOURCE / name for name in ("config.json", "freeze.json", "compiled.jsonl", "rankings.jsonl",
                                           "scored.jsonl", "queries.jsonl", "corpus.jsonl", "labels-hotpot.jsonl",
                                           "labels-musique.jsonl")),
              *(p / name for p in (INTENT, MECHANISM) for name in ("config.json", "selector_inputs.jsonl",
                                                                  "proposals.jsonl", "scored.jsonl")),
              INTENT / "guard-controls.json", ORACLE / "scored.jsonl", TOKENIZER, *code,
              *(snapshots / path for path in code)]
    dump(output / "config.json", {"scope": "development_replay", "model": MODEL, "python": platform.python_version(),
                                  "prepared_utc": datetime.now(timezone.utc).isoformat(), "temperature": 0,
                                  "thinking": False, "workers": WORKERS, "methods": METHODS, "budgets": BUDGETS,
                                  "selector": {"system": SYSTEM, "max_output": MAX_OUTPUT},
                                  "planner": {"system": PLAN_SYSTEM, "max_output": PLAN_LIMIT},
                                  "sha256": {p.as_posix(): sha256(p) for p in frozen}})


def archived_proposals() -> dict:
    reusable = {}
    for path in (INTENT, MECHANISM):
        inputs = {key(row): row for row in read_jsonl(path / "selector_inputs.jsonl")}
        for row in read_jsonl(path / "proposals.jsonl"):
            if row["method"] == PROPOSER and row["status"] == "ok":
                reusable[key(row)] = inputs[key(row)], row
    return reusable


def collect_proposal(reader: CloudReader, config: dict, query: dict, docs: list[dict], compiled: dict,
                     archive: tuple | None) -> dict:
    question = {k: query[k] for k in ("dataset", "id", "question")}
    if archive:
        previous_input, proposal = archive
        prepared = selector_input(query, docs, compiled, previous_input["plan"])
        if all(prepared[k] == previous_input[k] for k in ("question", "plan", "candidates")):
            return {**proposal, "reused": True}
    plan = query_plan(reader, question, config["planner"])
    if plan["status"] != "ok":
        return {**question, "status": "plan_" + plan["status"], "plan_result": plan, "reused": False}
    prepared = selector_input(query, docs, compiled, plan["plan"])
    proposal = select(reader, config["selector"], prepared, PROPOSER)
    return {**proposal, "plan_result": plan, "reused": False}


def collect(output: Path, config: dict, reader: CloudReader, queries: list[dict], corpus: dict, compiled: dict) -> dict:
    path = output / "proposals.jsonl"
    completed = {key(row): row for row in read_jsonl(path)} if path.exists() else {}
    archived = archived_proposals()
    todo = [q for q in queries if key(q) not in completed]

    def submit(query: dict) -> dict:
        docs = [corpus[c["id"]] for c in query["candidates"]]
        return collect_proposal(reader, config, query, docs, compiled, archived.get(key(query)))

    with path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for row in pool.map(submit, todo):
            append(log, row)
            completed[key(row)] = row
            if len(completed) % 10 == 0:
                print(f"Collected proposals {len(completed)}/{len(queries)}", flush=True)
    return completed


def assemble_question(tokenizer: BudgetTokenizer, query: dict, docs: list[dict], rank: dict, previous: list[dict],
                      proposal: dict, facts: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    candidates = sentence_candidates(docs)
    if rank["status"] != "ok" or rank["candidate_spans"] != [list(c[:2]) for c in candidates]:
        raise ValueError("Candidate spans differ from archived ranking")
    base = {k: query[k] for k in ("dataset", "family_id", "question")} | {
        "query_id": query["id"], "candidate_ids": [doc["id"] for doc in docs]}
    decisions = recover(proposal, docs, facts)
    requests, traces = [], []
    for budget in BUDGETS:
        baseline = pack_ranked(tokenizer, docs, candidates, rank["ranking"], budget)
        old = next(row for row in previous if row["method"] == "semantic" and row["budget"] == budget)
        if baseline[1:3] != (old["evidence"], old["evidence_tokens"]):
            raise ValueError("Semantic pack differs from archived input")
        for method, decision in zip(METHODS, decisions, strict=True):
            pack, assembly = packs(tokenizer, docs, candidates, rank["ranking"], budget, decision.path,
                                   joint=True, baseline=baseline)
            row = request(base, method, budget, *pack[:3], [])
            requests.append(row)
            direct = requests[-(METHODS.index(method) + 1)]
            added, removed = spans(row) - spans(direct), spans(direct) - spans(row)
            traces.append({"dataset": query["dataset"], "query_id": query["id"], "budget": budget, "method": method,
                           "validation": decision.status, "assembly": assembly,
                           "connection": asdict(decision.connection) if decision.connection else None,
                           "changed_from_direct": row["evidence"] != direct["evidence"],
                           "added": describe(docs, added), "removed": describe(docs, removed)})
    return requests, traces


def score_questions(output: Path, reader: CloudReader, queries: list[dict], corpus: dict, compiled: dict,
                    proposals: dict) -> tuple[list[dict], list[dict]]:
    ranks = {key(row): row for row in read_jsonl(SOURCE / "rankings.jsonl")}
    previous = defaultdict(list)
    for path in (SOURCE, INTENT, ORACLE, MECHANISM):
        for row in read_jsonl(path / "scored.jsonl"):
            previous[key(row)].append(row)
    tokenizer = BudgetTokenizer()
    requested, traces = [], []
    for query in queries:
        qkey = key(query)
        docs = [corpus[c["id"]] for c in query["candidates"]]
        old = previous[qkey]
        requests, decisions = assemble_question(tokenizer, query, docs, ranks[qkey], old, proposals[qkey],
                                                [compiled[d["id"]]["llm"] for d in docs])
        requested.extend(requests)
        traces.extend(decisions)
    jsonl(output / "requests.jsonl", requested)
    jsonl(output / "decisions.jsonl", traces)
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique")
              for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    by_key = defaultdict(list)
    for row in requested:
        by_key[key(row)].append(row)

    def answer(query: dict) -> list[dict]:
        qkey = key(query)
        old = previous[qkey]
        prior = {r["evidence"]: {k: v for k, v in r.items() if k.startswith("api_") or k in
                                {"answer", "status", "finish_reason"}} for r in old if r["status"] == "ok"}
        baseline = dict(next(row for row in old if row["method"] == "raw_full"), shared_api_result=True,
                        previous_run_api_result=True, list_price_cny=0, api_attempts=0)
        answers = read_question(reader, by_key[qkey], prior)
        return [score_row(row, query, corpus, labels[qkey]) for row in [baseline, *answers]]

    rows = []
    log_path = output / "scored.jsonl"
    if log_path.exists():
        rows = read_jsonl(log_path)
        expected = {(key(r), r["budget"], r["method"]): r["evidence"] for r in requested}
        for row in rows:
            if row["method"] in METHODS and expected[key(row), row["budget"], row["method"]] != row["evidence"]:
                raise ValueError("Cached reader response has different evidence")
    complete = Counter(key(row) for row in rows)
    if set(complete.values()) - {1 + len(METHODS) * len(BUDGETS)}:
        raise ValueError("Partial scored group in output")
    todo = [q for q in queries if key(q) not in complete]
    with log_path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for group in pool.map(answer, todo):
            rows.extend(group)
            log.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in group))
            log.flush()
            if len(rows) % 70 == 0:
                print(f"Scored {len(rows) // 7}/{len(queries)} questions", flush=True)
    return rows, traces


def usage(calls: list[dict]) -> dict:
    return {"requests": len(calls), "ok": sum(row["status"] == "ok" for row in calls),
            "input_tokens": sum(r.get("api_prompt_tokens", 0) for r in calls),
            "output_tokens": sum(r.get("api_completion_tokens", 0) for r in calls),
            "list_price_cny": sum(r.get("list_price_cny", 0) for r in calls)}


def summarize(rows: list[dict], traces: list[dict], proposals: dict) -> dict:
    groups = defaultdict(dict)
    for row in rows:
        group = groups[row["dataset"], row["budget"], row["method"]]
        if row["query_id"] in group:
            raise ValueError("Duplicate scored condition")
        group[row["query_id"]] = row
    methods, comparisons = {}, {}
    for (dataset, budget, method), by_id in sorted(groups.items()):
        key_name = f"{dataset}/{budget}/{method}"
        data = list(by_id.values())
        methods[key_name] = {"n": len(data), "ok": sum(r["status"] == "ok" for r in data),
                             "f1": mean(r["f1"] for r in data), "em": mean(r["em"] for r in data),
                             "complete_reference": sum(r["all_support_touched"] for r in data),
                             "evidence_tokens_mean": mean(r["evidence_tokens"] for r in data),
                             "reader_input_tokens_total": sum(r.get("api_prompt_tokens", 0) for r in data)}
        if method == "raw_full":
            continue
        references = [groups[dataset, FULL_LIMIT, "raw_full"]]
        if method != "direct":
            references.append(groups[dataset, budget, METHODS[METHODS.index(method) - 1]])
        comparisons[key_name] = {}
        for ref in references:
            if ref.keys() != by_id.keys():
                raise ValueError("Paired comparison has mismatched questions")
            matched = sorted(ref)
            comparisons[key_name][ref[matched[0]]["method"]] = {
                **difference(ref, by_id, matched),
                "changed_inputs": sum(ref[q]["evidence"] != by_id[q]["evidence"] for q in matched),
                "em_gained": sum(ref[q]["em"] < by_id[q]["em"] for q in matched),
                "em_lost": sum(ref[q]["em"] > by_id[q]["em"] for q in matched),
            }
    fresh = [r for r in proposals.values() if not r["reused"]]
    plan_calls = [r["plan_result"].get("api", {"status": "error"}) for r in fresh]
    selector_calls = [r.get("api", {"status": "error"}) for r in fresh if r["plan_result"]["status"] == "ok"]
    return {"methods": methods, "comparisons": comparisons, "scope": "development_replay",
            "validation_1024": {m: dict(Counter(r["validation"] for r in traces if r["budget"] == 1024 and
                                               r["method"] == m)) for m in METHODS},
            "proposal_status": dict(Counter(r["status"] for r in proposals.values())),
            "proposals_reused": len(proposals) - len(fresh), "planner": usage(plan_calls),
            "selector": usage(selector_calls), "reader": usage([r for r in rows if not r["shared_api_result"]]),
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def run(output: Path) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for name, digest in config["sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    _, corpus, queries = verify_freeze(SOURCE)
    compiled = {row["doc_id"]: row for row in read_jsonl(SOURCE / "compiled.jsonl")}
    if compiled.keys() != corpus.keys() or any(r["source_version"] != corpus[d]["version"] for d, r in compiled.items()):
        raise ValueError("Compiled records and frozen source versions differ")
    reader = CloudReader()
    proposals = collect(output, config, reader, queries, corpus, compiled)
    rows, traces = score_questions(output, reader, queries, corpus, compiled, proposals)
    dump(output / "summary.json", summarize(rows, traces, proposals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)
