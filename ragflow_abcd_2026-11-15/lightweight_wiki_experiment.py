"""Compare local original-sentence selection with and without archived Wiki hints."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from statistics import mean

from auto_view_route import choose_chain
from baseline import SYSTEM
from cloud_experiment import BudgetTokenizer, FULL_LIMIT, TOKENIZER, TOKENIZER_REVISION
from cloud_reader import CloudReader, MAX_OUTPUT, MODEL
from dependency_development import SOURCE, describe, spans
from external_compression import append, dump, read_question
from guard_policy import eligible
from intent_evidence import state_qualifiers
from intent_evidence_experiment import sha256
from join_recovery_experiment import key
from paired import score_row
from prepare import jsonl, read_jsonl
from semantic_bridge_experiment import request, verify_freeze
from sweep_compare import difference
from wiki_bridge import accent_fold, guarded_bridges, span_text, title_surface
from wiki_compiler_heldout import hints_for


BUDGETS = (512, 1024, 2560)
METHODS = ("local_raw", "local_wiki")
PREVIOUS = Path("runs/join-recovery-development100-20261112-verified")
WORKERS = 12


def chain_check(question: str, docs: list[dict]) -> set[tuple[int, int]]:
    chain = choose_chain(question, docs, [])
    if chain["reason"] != "short":
        return set()
    entity = chain["chain_value"]
    states = [state_qualifiers(accent_fold(docs[p]["title"]).casefold(),
                              accent_fold(span_text(docs[p], i)).casefold(), entity) for p, i in chain["spans"]]
    if all(states) and states[0].isdisjoint(states[1]):
        return set()
    return set(map(tuple, chain["spans"]))


def route(row: dict, full: dict, chain: set) -> str:
    if row["evidence_tokens"] >= full["evidence_tokens"]:
        return "already_short"
    guard = row["guard"]
    if set(guard["direct_pages"]) - set(guard["direct_pages_covered"]):
        return "fallback_missing_named_page"
    if chain and chain <= spans(row):
        return "short_checked_chain"
    if eligible({**row, "raw_evidence_tokens": full["evidence_tokens"]}, 0.70):
        return "short_coverage"
    return "fallback_unverified"


def assemble_question(tokenizer: BudgetTokenizer, query: dict, docs: list[dict], compiled: dict,
                      full: dict) -> tuple[list[dict], list[dict]]:
    if full["candidate_ids"] != [d["id"] for d in docs] or full["question"] != query["question"]:
        raise ValueError("Archived A has different question or candidates")
    question = query["question"]
    base = {k: full[k] for k in ("dataset", "query_id", "family_id", "candidate_ids", "question")}
    titles = {doc["id"]: title_surface(doc["title"]) for doc in docs}
    hints = hints_for(question, docs, compiled, "llm")
    chain = chain_check(question, docs)
    rows, log = [], []
    for budget in BUDGETS:
        packed = []
        for method, arm_hints in zip(METHODS, ({}, hints), strict=True):
            if full["evidence_tokens"] <= budget:
                row = {**base, "method": method, "budget": budget, "selected_blocks": full["selected_blocks"],
                       "evidence": full["evidence"], "evidence_tokens": full["evidence_tokens"], "wiki_edges": [],
                       "guard": {"direct_pages": [], "direct_pages_covered": []}}
            else:
                blocks, text, count, events, guard = guarded_bridges(tokenizer, question, docs, titles, budget,
                                                                   fact_hints=arm_hints)
                row = {**request(base, method, budget, blocks, text, count, events), "guard": guard}
            packed.append(row)
        raw, wiki = packed
        log.append({"dataset": query["dataset"], "query_id": query["id"], "budget": budget,
                    "local_chain": sorted(chain), "wiki_hints": len(hints),
                    "raw_route": route(raw, full, chain), "wiki_route": route(wiki, full, chain),
                    "fixed_input_changed": raw["evidence"] != wiki["evidence"],
                    "wiki_added": describe(docs, spans(wiki) - spans(raw)),
                    "wiki_removed": describe(docs, spans(raw) - spans(wiki))})
        rows.extend(packed)
    return rows, log


def prepare(output: Path) -> None:
    _, corpus, queries = verify_freeze(SOURCE)
    if Counter(q["dataset"] for q in queries) != {"hotpot": 50, "musique": 50}:
        raise ValueError("Expected the complete archived 50+50 development cohort")
    archived = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    if (archived["model"], archived["system"], archived["max_output"], archived["temperature"], archived["thinking"]) != (
            MODEL, SYSTEM, MAX_OUTPUT, 0, False):
        raise ValueError("Archived A reader differs")
    compiled = {r["doc_id"]: r for r in read_jsonl(SOURCE / "compiled.jsonl")}
    if compiled.keys() != corpus.keys() or any(r["source_version"] != corpus[d]["version"] for d, r in compiled.items()):
        raise ValueError("Compiled rows do not match frozen source versions")
    full = {key(r): r for r in read_jsonl(SOURCE / "scored.jsonl") if r["method"] == "raw_full"}
    output.mkdir()
    tokenizer = BudgetTokenizer()
    rows, decisions = [], []
    for q in queries:
        docs = [corpus[c["id"]] for c in q["candidates"]]
        planned, route_log = assemble_question(tokenizer, q, docs, compiled, full[key(q)])
        rows.extend(planned)
        decisions.extend(route_log)
    jsonl(output / "requests.jsonl", rows)
    jsonl(output / "routes.jsonl", decisions)
    code = [Path(name) for name in ("lightweight_wiki_experiment.py", "wiki_bridge.py", "baseline.py", "guard_policy.py",
                                    "auto_view_route.py", "wiki_compiler_heldout.py", "wiki_compile_v2.py",
                                    "wiki_compile_pilot.py", "intent_evidence.py", "join_recovery_experiment.py",
                                    "cloud_experiment.py", "cloud_reader.py", "external_compression.py", "paired.py",
                                    "prepare.py", "semantic_bridge_experiment.py", "sweep_compare.py",
                                    "scripts/check_paired.py")]
    for path in code:
        snapshot = output / "source" / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    frozen = [*(SOURCE / name for name in ("freeze.json", "queries.jsonl", "corpus.jsonl", "compiled.jsonl",
                                           "scored.jsonl", "labels-hotpot.jsonl", "labels-musique.jsonl")),
              PREVIOUS / "scored.jsonl", TOKENIZER, *code, *(output / "source" / path for path in code),
              output / "requests.jsonl", output / "routes.jsonl"]
    dump(output / "config.json", {"scope": "development_replay", "method": "local guarded original sentence selection",
                                  "model": MODEL, "temperature": 0, "thinking": False, "system": SYSTEM,
                                  "max_output": MAX_OUTPUT, "tokenizer_revision": TOKENIZER_REVISION,
                                  "python": platform.python_version(), "budgets": BUDGETS, "workers": WORKERS,
                                  "coverage_threshold": 0.70, "prepared_utc": datetime.now(timezone.utc).isoformat(),
                                  "cloud_selector_requests": 0, "sha256": {p.as_posix(): sha256(p) for p in frozen}})
    print(f"Prepared {len(rows)} fixed packs and {len(decisions)} paired local routing decisions", flush=True)


def read_all(output: Path, queries: list[dict], corpus: dict, requested: dict, archive: dict) -> list[dict]:
    labels = {(d, r["id"]): r for d in ("hotpot", "musique") for r in read_jsonl(SOURCE / f"labels-{d}.jsonl")}
    reader = CloudReader()

    def evaluate(q: dict) -> list[dict]:
        qkey = key(q)
        old = archive[qkey]
        prior = {r["evidence"]: {k: v for k, v in r.items() if k.startswith("api_") or k in
                                {"answer", "status", "finish_reason"}} for r in old if r["status"] == "ok"}
        full = next(r for r in old if r["method"] == "raw_full")
        prior[full["evidence"]] = {k: v for k, v in full.items() if k.startswith("api_") or k in
                                   {"answer", "status", "finish_reason"}}
        a = {**full, "shared_api_result": True, "previous_run_api_result": True, "list_price_cny": 0.0}
        return [score_row(row, q, corpus, labels[qkey]) for row in [a, *read_question(reader, requested[qkey], prior)]]

    path = output / "scored.jsonl"
    scored = read_jsonl(path) if path.exists() else []
    counts = Counter(key(r) for r in scored)
    if set(counts.values()) - {1 + len(BUDGETS) * len(METHODS)}:
        raise ValueError("Incomplete per-question group in score log")
    pending = [q for q in queries if key(q) not in counts]
    with path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for group in pool.map(evaluate, pending):
            scored.extend(group)
            log.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in group))
            log.flush()
            if len(scored) % 70 == 0:
                print(f"Scored {len(scored)//7}/{len(queries)} questions", flush=True)
    return scored


def routed_rows(scored: list[dict], decisions: list[dict]) -> list[dict]:
    lookup = {(r["query_id"], r["budget"], r["method"]): r for r in scored}
    routed = []
    for row in decisions:
        qid, budget = row["query_id"], row["budget"]
        for method, route_key in zip(METHODS, ("raw_route", "wiki_route"), strict=True):
            choice = row[route_key]
            picked = lookup[qid, budget, method] if choice.startswith("short_") else lookup[qid, FULL_LIMIT, "raw_full"]
            routed.append({**picked, "method": method + "_routed", "target_budget": budget, "route": choice})
    return routed


def summarize(scored: list[dict], routed: list[dict], decisions: list[dict]) -> dict:
    groups = defaultdict(dict)
    for row in [*scored, *routed]:
        budget = row.get("target_budget", row["budget"])
        for ds in (row["dataset"], "all"):
            groups[ds, budget, row["method"]][row["query_id"]] = row
    metrics = {}
    for (ds, budget, method), data in sorted(groups.items()):
        a = groups[ds, FULL_LIMIT, "raw_full"]
        ids = sorted(a)
        if data.keys() != a.keys():
            raise ValueError("Incomplete paired comparison")
        rows = [data[qid] for qid in ids]
        tokens = sum(r.get("api_prompt_tokens", 0) for r in rows)
        result = {"n": len(ids), "ok": sum(r["status"] == "ok" for r in rows),
                  "f1": mean(r["f1"] for r in rows), "em": mean(r["em"] for r in rows),
                  "official_support_touched": sum(r["all_support_touched"] for r in rows),
                  "online_input_tokens": tokens, "output_tokens": sum(r.get("api_completion_tokens", 0) for r in rows),
                  "input_saving_vs_a": 1 - tokens / sum(r["api_prompt_tokens"] for r in a.values()),
                  "routes": dict(Counter(r["route"] for r in rows if "route" in r)),
                  "questions_with_shorter_evidence_than_a": sum(data[q]["evidence_tokens"] < a[q]["evidence_tokens"] for q in ids)}
        if method != "raw_full":
            comparisons = [a]
            if method.startswith("local_wiki"):
                comparisons.append(groups[ds, budget, method.replace("wiki", "raw")])
            for ref in comparisons:
                name = "vs_" + ref[ids[0]]["method"]
                result[name] = {**difference(ref, data, ids),
                                "changed_inputs": sum(ref[q]["evidence"] != data[q]["evidence"] for q in ids),
                                "em_gained": sum(data[q]["em"] > ref[q]["em"] for q in ids),
                                "em_lost": sum(data[q]["em"] < ref[q]["em"] for q in ids)}
        metrics[f"{ds}/{budget}/{method}"] = result
    live = [r for r in scored if not r["shared_api_result"]]
    return {"conditions": metrics, "cloud_selector_calls": 0, "new_reader_calls": len(live),
            "new_reader_input_tokens": sum(r.get("api_prompt_tokens", 0) for r in live),
            "new_reader_output_tokens": sum(r.get("api_completion_tokens", 0) for r in live),
            "new_reader_list_price_cny": sum(r.get("list_price_cny", 0) for r in live),
            "questions_with_predicate_matched_wiki_hints": len({r["query_id"] for r in decisions if r["wiki_hints"]}),
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def run(output: Path) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for name, expected in config["sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    _, corpus, queries = verify_freeze(SOURCE)
    requested, prior = defaultdict(list), defaultdict(list)
    for row in read_jsonl(output / "requests.jsonl"):
        requested[key(row)].append(row)
    for source in (SOURCE, PREVIOUS):
        for row in read_jsonl(source / "scored.jsonl"):
            prior[key(row)].append(row)
    decisions = read_jsonl(output / "routes.jsonl")
    scored = read_all(output, queries, corpus, requested, prior)
    routed = routed_rows(scored, decisions)
    jsonl(output / "routed-scored.jsonl", routed)
    dump(output / "summary.json", summarize(scored, routed, decisions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)
