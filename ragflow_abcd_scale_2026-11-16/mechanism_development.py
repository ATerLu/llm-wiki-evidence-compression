"""Diagnose source coverage, individual selection and joint evidence selection on seen questions."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from statistics import mean

from baseline import sentence_blocks
from cloud_experiment import BudgetTokenizer, TOKENIZER
from cloud_reader import CloudReader, MODEL
from dependency_development import SOURCE, describe, spans
from external_compression import append, dump, pack_ranked, read_question, sentence_candidates
from intent_evidence import validate_proof
from intent_evidence_experiment import MAX_OUTPUT, SYSTEM, select, sha256
from paired import score_row
from prepare import jsonl, read_jsonl
from semantic_bridge import TARGET_WINDOW
from semantic_bridge_experiment import BUDGETS, request, verify_freeze


ORACLE = Path("runs/oracle-window-development-20261111")
INTENT = Path("runs/intent-evidence-development-20261111")
PROPOSERS = ("intent_raw", "intent_wiki")
PACKERS = ("individual", "joint_no_place_check", "joint_checked")


def prepare(output: Path) -> None:
    output.mkdir()
    inputs = [{k: row[k] for k in ("dataset", "id", "question", "plan", "candidates", "wiki_index")}
              for row in read_jsonl(ORACLE / "selector_inputs.jsonl")]
    assert {row["id"] for row in inputs} == json.loads((ORACLE / "cases.json").read_text(encoding="utf-8")).keys()
    jsonl(output / "selector_inputs.jsonl", inputs)
    code = [Path(name) for name in ("mechanism_development.py", "intent_evidence.py", "intent_evidence_experiment.py",
                                    "semantic_bridge.py", "external_compression.py", "cloud_reader.py", "baseline.py",
                                    "paired.py", "wiki_bridge.py", "cloud_experiment.py", "semantic_bridge_experiment.py",
                                    "dependency_development.py", "prepare.py")]
    snapshots = output / "source"
    snapshots.mkdir()
    for path in code:
        (snapshots / path.name).write_bytes(path.read_bytes())
    frozen = [output / "selector_inputs.jsonl", ORACLE / "cases.json", ORACLE / "selector_inputs.jsonl",
              INTENT / "guard-controls.json", *(path / "scored.jsonl" for path in (SOURCE, INTENT, ORACLE)),
              *(SOURCE / name for name in ("compiled.jsonl", "rankings.jsonl", "freeze.json", "config.json")),
              *code, *(snapshots / path.name for path in code), TOKENIZER]
    dump(output / "config.json", {
        "scope": "development_only", "model": MODEL, "python": platform.python_version(),
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "system": SYSTEM, "max_output": MAX_OUTPUT,
        "budgets": BUDGETS, "workers": 4, "temperature": 0, "thinking": False,
        "proposers": PROPOSERS, "packers": PACKERS, "candidate_scope": "identical full archived candidates",
        "pack_priority": "first five semantic candidates, proposed spans in semantic order, remaining semantic order",
        "sha256": {path.as_posix(): sha256(path) for path in frozen},
    })


def validate(proposal: dict, docs: list[dict], *, check_place_scope: bool) -> tuple:
    if proposal["status"] != "ok":
        return None, proposal["status"]
    try:
        path = validate_proof(proposal["proposal"], docs, check_place_scope=check_place_scope)
        return path, "validated" if path else "model_abstained"
    except ValueError as error:
        return None, str(error)


def packs(tokenizer: BudgetTokenizer, docs: list[dict], candidates: list, ranking: list[int], budget: int,
          path: tuple | None, *, joint: bool, baseline: tuple) -> tuple:
    if path is None:
        return baseline, "fallback"
    proof = set(path[1].proof)
    head, tail = ranking[:TARGET_WINDOW], ranking[TARGET_WINDOW:]
    promoted = [i for i in tail if candidates[i][:2] in proof]
    if not joint:
        pack = pack_ranked(tokenizer, docs, candidates, list(dict.fromkeys([*head, *promoted, *tail])), budget)
        return pack, "prioritize_individually"

    # Preserve the same leading sentences that fit under independent selection.
    leading = {candidates[i][:2] for i in head if i in baseline[3]}
    pinned = leading | proof
    blocks = defaultdict(set)
    for page, index in pinned:
        blocks[page].add(index)
    if len(tokenizer.tokenize(sentence_blocks(docs, blocks))) > budget:
        return baseline, "proof_and_leading_over_budget"
    return pack_ranked(tokenizer, docs, candidates, ranking, budget, pinned=tuple(sorted(pinned))), "include_whole_chain"


def evaluate_question(tokenizer: BudgetTokenizer, query: dict, docs: list[dict], rank: dict, proposals: list[dict],
                      previous: list[dict], required: set) -> tuple[list[dict], list[dict]]:
    candidates = sentence_candidates(docs)
    if rank["status"] != "ok" or rank["candidate_spans"] != [list(c[:2]) for c in candidates]:
        raise ValueError("Archived ranking and candidate coordinates disagree")
    core = {k: query[k] for k in ("dataset", "family_id", "question")} | {
        "query_id": query["id"], "candidate_ids": [doc["id"] for doc in docs]}
    ranking, requests, traces = rank["ranking"], [], []
    verdicts = {(row["method"], checked): validate(row, docs, check_place_scope=checked)
                for row in proposals for checked in (False, True)}
    for budget in BUDGETS:
        baseline = pack_ranked(tokenizer, docs, candidates, ranking, budget)
        old = next(r for r in previous if r["method"] == "semantic" and r["budget"] == budget)
        if baseline[1:3] != (old["evidence"], old["evidence_tokens"]):
            raise ValueError("Semantic baseline does not reproduce archived evidence")
        for row in proposals:
            for packer in PACKERS:
                checked = packer != "joint_no_place_check"
                path, validation = verdicts[row["method"], checked]
                pack, decision = packs(tokenizer, docs, candidates, ranking, budget, path,
                                       joint=packer != "individual", baseline=baseline)
                method = f"{row['method']}_{packer}"
                planned = request(core, method, budget, *pack[:3], [])
                chosen, proposed = spans(planned), set(path[1].proof) if path else set()
                if decision == "include_whole_chain" and not proposed <= chosen:
                    raise ValueError("Joint pack lost a pinned original sentence")
                requests.append(planned)
                traces.append({"query_id": query["id"], "method": method, "budget": budget,
                               "validation": validation, "decision": decision, "changed": pack[1] != old["evidence"],
                               "proposal_covers_reference": required <= proposed, "retained_reference": required <= chosen,
                               "proposal_spans": sorted(proposed), "missing_reference": sorted(required - chosen),
                               "added": describe(docs, chosen - spans(old)), "removed": describe(docs, spans(old) - chosen)})
    return requests, traces


def source_coverage(docs: list[dict], required: set, compiled: dict) -> list[dict]:
    coverage = []
    for page, index in sorted(required):
        doc = docs[page]
        row = compiled[doc["id"]]
        if doc["version"] != row["source_version"]:
            raise ValueError("Archived compiler source version differs")
        facts = [{k: fact[k] for k in ("predicate", "object", "sentence_id", "anchor_id")}
                 for fact in row["llm"] if any(part["index"] == index for part in fact["proof"])]
        rejections = [r for r in row["rejected"]
                      if isinstance(r["candidate"], dict) and r["candidate"].get("sentence_id") == index]
        coverage.append({"span": [page, index], "source": describe(docs, {(page, index)})[0],
                         "compiler_parse": row["parse"], "accepted_pointers": facts, "rejected_same_sentence": rejections})
    return coverage


def controls(corpus: dict, queries: dict) -> list[dict]:
    results = []
    for case in json.loads((INTENT / "guard-controls.json").read_text(encoding="utf-8"))["controls"]:
        if "forced_proposal" not in case:
            continue
        q = queries["musique", case["id"]]
        docs = [corpus[c["id"]] for c in q["candidates"]]
        proposal = {"status": "ok", "proposal": case["forced_proposal"]}
        path, unchecked = validate(proposal, docs, check_place_scope=False)
        _, checked = validate(proposal, docs, check_place_scope=True)
        results.append({"id": case["id"], "no_place_check": unchecked, "with_place_check": checked,
                        "join": path[1].relations if path else [],
                        "source": describe(docs, set(map(tuple, proposal["proposal"]["spans"])))})
    return results


def propose(output: Path, reader: CloudReader, config: dict, inputs: list[dict]) -> list[dict]:
    path = output / "proposals.jsonl"
    prior = read_jsonl(path) if path.exists() else []
    done = {(row["dataset"], row["id"], row["method"]) for row in prior}
    jobs = [(row, method) for row in inputs for method in PROPOSERS if (row["dataset"], row["id"], method) not in done]
    with path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for result in pool.map(lambda job: select(reader, config, *job), jobs):
            append(log, result)
            prior.append(result)
            print(f"Proposal {len(prior)}/{len(inputs) * len(PROPOSERS)}: {result['status']}", flush=True)
    return prior


def summarize(rows: list[dict], traces: list[dict], calls: list[dict], audit: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["budget"], row["method"]].append(row)
    methods = {}
    for (budget, method), group in sorted(grouped.items()):
        selected = [r for r in traces if (r["budget"], r["method"]) == (budget, method)]
        methods[f"{budget}/{method}"] = {
            "n": len(group), "ok": sum(r["status"] == "ok" for r in group),
            "f1": mean(r["f1"] for r in group), "em": mean(r["em"] for r in group),
            "complete_reference": sum(r["retained_reference"] for r in group),
            "token_mean": mean(r["evidence_tokens"] for r in group),
            "changed": sum(r["changed"] for r in selected),
            "validated_proposal_covers_reference": sum(r["proposal_covers_reference"] for r in selected),
            "validation": dict(Counter(r["validation"] for r in selected)),
        }
    api = [row["api"] for row in calls if "api" in row]
    return {"methods": methods, "scope": "development_only", "selector_requests": len(calls),
            "selector_status": dict(Counter(r["status"] for r in calls)),
            "selector_input_tokens": sum(r["api_prompt_tokens"] for r in api),
            "selector_output_tokens": sum(r["api_completion_tokens"] for r in api),
            "selector_list_price_cny": sum(r["list_price_cny"] for r in api),
            "new_reader_requests": sum(not r.get("shared_api_result", True) for r in rows),
            "reader_list_price_cny": sum(r.get("list_price_cny", 0) for r in rows if not r.get("shared_api_result", True)),
            "reference_spans_with_any_pointer": sum(bool(s["accepted_pointers"]) for r in audit for s in r["reference"]),
            "reference_spans_total": sum(len(r["reference"]) for r in audit),
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def run(output: Path) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for name, digest in config["sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    _, corpus, frozen = verify_freeze(SOURCE)
    queries = {(q["dataset"], q["id"]): q for q in frozen}
    inputs = read_jsonl(output / "selector_inputs.jsonl")
    reader, tokenizer = CloudReader(), BudgetTokenizer()
    calls = propose(output, reader, config, inputs)
    references = json.loads((ORACLE / "cases.json").read_text(encoding="utf-8"))
    compiled = {row["doc_id"]: row for row in read_jsonl(SOURCE / "compiled.jsonl")}
    ranks = {(r["dataset"], r["query_id"]): r for r in read_jsonl(SOURCE / "rankings.jsonl")}
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique") for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    previous, proposals = defaultdict(list), defaultdict(list)
    for source in (SOURCE, INTENT, ORACLE):
        for row in read_jsonl(source / "scored.jsonl"):
            previous[row["dataset"], row["query_id"]].append(row)
    for row in calls:
        proposals[row["dataset"], row["id"]].append(row)

    rows, traces, audit, all_requests = [], [], [], []
    for item in inputs:
        key = item["dataset"], item["id"]
        q = queries[key]
        docs = [corpus[c["id"]] for c in q["candidates"]]
        required = set(map(tuple, references[q["id"]]["spans"]))
        observed = {(p["page"], s[0]) for p in item["candidates"] for s in p["sentences"]}
        if not required <= observed:
            raise ValueError("Reference evidence not visible in the full frozen selector candidates")
        audit.append({"query_id": q["id"], "reference": source_coverage(docs, required, compiled)})
        old = previous[key]
        requests, decisions = evaluate_question(tokenizer, q, docs, ranks[key], proposals[key], old, required)
        all_requests.extend(requests)
        traces.extend(decisions)
        prior = {r["evidence"]: {k: v for k, v in r.items() if k.startswith("api_") or k in
                                {"answer", "status", "finish_reason"}} for r in old if r["status"] == "ok"}
        scored = [score_row(r, q, corpus, labels[key]) for r in read_question(reader, requests, prior)]
        reference_rows = [dict(r, shared_api_result=True) for r in old if r["method"] in ("raw_full", "semantic", "semantic_rule")]
        rows.extend(dict(r, retained_reference=required <= spans(r)) for r in [*reference_rows, *scored])
        print(f"Scored {len(audit)}/{len(inputs)}: {q['id']}", flush=True)
    jsonl(output / "requests.jsonl", all_requests)
    jsonl(output / "scored.jsonl", rows)
    jsonl(output / "changes.jsonl", traces)
    jsonl(output / "source-coverage.jsonl", audit)
    dump(output / "place-controls.json", controls(corpus, queries))
    dump(output / "summary.json", summarize(rows, traces, calls, audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)
