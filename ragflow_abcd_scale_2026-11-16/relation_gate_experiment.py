"""Run paired development comparisons of pre-reader evidence checks."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from math import isclose
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter

from cloud_experiment import BudgetTokenizer, TOKENIZER, TOKENIZER_REVISION
from cloud_reader import CloudReader, ENDPOINT, INPUT_PRICE, MAX_OUTPUT, MODEL, OUTPUT_PRICE
from dependency_development import SOURCE, describe, spans
from external_compression import dump, read_question
from intent_evidence_experiment import sha256
from join_recovery_experiment import key
from lightweight_wiki_experiment import BUDGETS, SYSTEM
from paired import score_row, support_sets
from prepare import jsonl, read_jsonl
from ragflow_e2_replay import execute as run_e2, prepare_packs as e2_packs
from raw_relation_locator import locate
from relation_gate import choose_pack, find_paths
from scripts.check_paired import official_scores
from semantic_bridge_experiment import verify_freeze
from sweep_compare import difference


OLD = Path("runs/lightweight-wiki-development100-20261112")
CLAIMS = Path("runs/ragflow-compare100-20261112/compile-api.jsonl")
WORKERS = 10
METHODS = ("legacy_raw", "legacy_wiki", "checked_raw", "checked_wiki", "located_raw", "located_wiki")


def labels_for_cohort() -> dict:
    return {(ds, row["id"]): row for ds in ("hotpot", "musique")
            for row in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}


def audit_legacy(output: Path, corpus: dict, queries: list[dict]) -> None:
    labels = labels_for_cohort()
    scores = {(key(r), r["budget"], r["method"]): r for r in read_jsonl(OLD / "scored.jsonl")}
    full = {qid: row for (qid, _, method), row in scores.items() if method == "raw_full"}
    routes = {key(r): r for r in read_jsonl(OLD / "routes.jsonl") if r["budget"] == 1024}
    rows = []
    for q in queries:
        qkey = key(q)
        docs = [corpus[c["id"]] for c in q["candidates"]]
        a = full[qkey]
        short = scores[qkey, 1024, "local_raw"]
        selected = {i: set(indices) for i, indices in short["selected_blocks"]}
        ref, strict, touched = support_sets(q["dataset"], q, docs, labels[qkey], selected)
        missing = [(i, j) for i, c in enumerate(q["candidates"]) for title, j in ref
                   if title == c["key"] and j not in selected.get(i, set())] if q["dataset"] == "hotpot" else []
        rows.append({
            "dataset": q["dataset"], "query_id": q["id"], "question": q["question"],
            "route": routes[qkey]["raw_route"], "a_answer": a["answer"], "short_answer": short["answer"],
            "a_f1": a["f1"], "short_f1": short["f1"], "a_em": a["em"], "short_em": short["em"],
            "short_strict_support": ref <= strict, "short_support_touched": ref <= touched,
            "missing_hotpot_sentences": describe(docs, set(missing)),
            "missing_musique_paragraphs": sorted(ref - touched) if q["dataset"] == "musique" else [],
        })
    jsonl(output / "legacy-audit.jsonl", rows)


def prior_packs(output: Path, corpus: dict, queries: list[dict]) -> tuple[dict, dict]:
    lookup = {(key(r), r["budget"], r["method"]): r for r in read_jsonl(OLD / "requests.jsonl")}
    routes = {(key(r), r["budget"]): r for r in read_jsonl(OLD / "routes.jsonl")}
    claims = defaultdict(list)
    for batch in read_jsonl(CLAIMS):
        if batch["status"] != "ok":
            continue
        for claim in batch["claims"]:
            if corpus[claim["doc_id"]]["version"] != claim["source_version"]:
                raise ValueError("E2 claim source version differs")
            claims[claim["doc_id"]].append(claim)
    jsonl(output / "e2-requests.jsonl", e2_packs(corpus, queries, claims))
    return lookup, routes


def prepare(output: Path) -> None:
    _, corpus, queries = verify_freeze(SOURCE)
    if Counter(q["dataset"] for q in queries) != {"hotpot": 50, "musique": 50}:
        raise ValueError("Expected the archived 50+50 development cohort")
    compiled = {r["doc_id"]: r for r in read_jsonl(SOURCE / "compiled.jsonl")}
    full = {key(r): r for r in read_jsonl(SOURCE / "scored.jsonl") if r["method"] == "raw_full"}
    tokenizer = BudgetTokenizer()
    output.mkdir()
    lookup, routes = prior_packs(output, corpus, queries)
    rows = []
    for q in queries:
        qkey = key(q)
        docs = [corpus[c["id"]] for c in q["candidates"]]
        a = full[qkey]
        rows.append(planned(a, "raw_full", a["budget"], "full", []))
        for method, predict, wiki in (
            ("checked_raw", find_paths, {}), ("checked_wiki", find_paths, compiled),
            ("located_raw", locate, {}), ("located_wiki", locate, compiled),
        ):
            started = perf_counter()
            prediction = predict(q["question"], docs, wiki)
            locate_ms = 1000 * (perf_counter() - started)
            for budget in BUDGETS:
                started = perf_counter()
                original = lookup[qkey, budget, "local_raw"]
                pack, reason, path = choose_pack(tokenizer, docs, budget, a, prediction, original)
                row = planned(pack, method, budget, reason, path)
                row["locator_relation"] = prediction[0]
                row["candidate_proofs"] = len({frozenset(p["spans"]) for p in prediction[1]})
                row["new_spans_vs_legacy_raw"] = describe(docs, spans(pack) - spans(original)) if reason.startswith("checked_") else []
                row["route_ms"] = locate_ms + 1000 * (perf_counter() - started)
                rows.append(row)
        for budget in BUDGETS:
            for suffix in ("raw", "wiki"):
                route = routes[qkey, budget][suffix + "_route"]
                pack = lookup[qkey, budget, "local_" + suffix] if route.startswith("short_") else a
                rows.append(planned(pack, "legacy_" + suffix, budget, route, []))
    jsonl(output / "requests.jsonl", rows)
    code = [Path(name) for name in (
        "relation_gate.py", "raw_relation_locator.py", "relation_gate_experiment.py", "lightweight_wiki_experiment.py",
        "wiki_bridge.py", "wiki_compile_v2.py", "wiki_compile_pilot.py", "evidence_join_recovery.py",
        "intent_evidence.py", "baseline.py", "cloud_reader.py", "cloud_experiment.py",
        "paired.py", "external_compression.py", "ragflow_e2_replay.py", "scripts/check_paired.py",
    )]
    for path in code:
        snapshot = output / "source" / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    files = [*(SOURCE / name for name in ("freeze.json", "queries.jsonl", "corpus.jsonl", "compiled.jsonl",
                                         "scored.jsonl", "labels-hotpot.jsonl", "labels-musique.jsonl")),
             *(OLD / name for name in ("requests.jsonl", "routes.jsonl", "scored.jsonl")), CLAIMS, TOKENIZER,
             output / "requests.jsonl", output / "e2-requests.jsonl", *code]
    dump(output / "config.json", {
        "scope": "development", "n": len(queries), "budgets": BUDGETS, "model": MODEL, "endpoint": ENDPOINT,
        "system": SYSTEM, "max_output": MAX_OUTPUT, "temperature": 0, "thinking": False,
        "workers": WORKERS, "methods": METHODS, "tokenizer_revision": TOKENIZER_REVISION, "python": platform.python_version(),
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "prior_answer_reuse": False,
        "sha256": {p.as_posix(): sha256(p) for p in files},
    })
    audit_legacy(output, corpus, queries)
    print(f"Frozen {len(queries)} questions, {len(rows)} one-read conditions, {len(queries)} E2 conditions", flush=True)


def planned(pack: dict, method: str, target_budget: int, route: str, path: list) -> dict:
    fields = ("query_id", "dataset", "family_id", "question", "candidate_ids",
              "budget", "evidence", "evidence_tokens", "selected_blocks")
    return {**{k: pack[k] for k in fields}, "method": method, "target_budget": target_budget,
            "route": route, "verified_spans": path}


def evaluate(q: dict, corpus: dict, label: dict, rows: list[dict], e2: dict) -> list[dict]:
    reader = CloudReader()
    scored = [score_row(row, q, corpus, label) for row in read_question(reader, rows, {})]
    a = next(r for r in scored if r["method"] == "raw_full")
    result = run_e2(reader, e2, q, corpus, label, a)
    return [*scored, {**result, "method": "e2", "target_budget": 1024}]


def score_run(output: Path, corpus: dict, queries: list[dict]) -> list[dict]:
    labels = labels_for_cohort()
    requested = defaultdict(list)
    for row in read_jsonl(output / "requests.jsonl"):
        requested[key(row)].append(row)
    e2 = {key(r): r for r in read_jsonl(output / "e2-requests.jsonl")}
    path = output / "scored.jsonl"
    completed = read_jsonl(path) if path.exists() else []
    expected = 2 + len(BUDGETS) * len(METHODS)
    done = Counter(key(row) for row in completed)
    if set(done.values()) - {expected} or done.keys() - requested.keys():
        raise ValueError("The score log contains an incomplete or unknown question group")
    todo = [q for q in queries if key(q) not in done]
    with path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for group in pool.map(lambda q: evaluate(q, corpus, labels[key(q)], requested[key(q)], e2[key(q)]), todo):
            completed.extend(group)
            log.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in group))
            log.flush()
            n = len(completed) // expected
            if n % 10 == 0:
                print(f"Scored {n}/{len(queries)} questions", flush=True)
    return completed


def measure(rows: list[dict], a: dict, wiki_ref: dict | None) -> dict:
    by_id = {r["query_id"]: r for r in rows}
    if by_id.keys() != a.keys() or len(by_id) != len(rows):
        raise ValueError("Incomplete paired group")
    ids = sorted(by_id)
    prompt = sum(r.get("api_prompt_tokens", 0) for r in rows)
    tokens = sum(r.get("api_completion_tokens", 0) for r in rows)
    a_tokens = sum(r.get("api_prompt_tokens", 0) for r in a.values())
    result = {
        "n": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
        "f1": mean(r["f1"] for r in rows), "em": mean(r["em"] for r in rows),
        "strict_support": sum(r["selected_support_complete"] for r in rows),
        "support_touched": sum(r["all_support_touched"] for r in rows),
        "shorter": sum(r["evidence_tokens"] < a[r["query_id"]]["evidence_tokens"] for r in rows),
        "changed_from_a": sum(r["evidence"] != a[r["query_id"]]["evidence"] for r in rows),
        "routes": dict(Counter(r["route"] for r in rows)),
        "prompt_tokens": prompt, "completion_tokens": tokens, "saving_vs_a": 1 - prompt / a_tokens,
        "logical_calls": sum(r.get("logical_calls", 1) for r in rows),
        "list_price_cny": (prompt * INPUT_PRICE + tokens * OUTPUT_PRICE) / 1_000_000,
        "f1_vs_a": difference(a, by_id, ids),
    }
    short = [r for r in rows if r["evidence_tokens"] < a[r["query_id"]]["evidence_tokens"]]
    result["short_touched_missing"] = sum(not r["all_support_touched"] for r in short)
    result["short_strict_missing"] = sum(not r["selected_support_complete"] for r in short)
    result["short_em_regressions"] = sum(r["em"] < a[r["query_id"]]["em"] for r in short)
    result["short_f1_regressions"] = sum(r["f1"] < a[r["query_id"]]["f1"] for r in short)
    if wiki_ref is not None:
        result["f1_vs_no_wiki"] = difference(wiki_ref, by_id, ids)
        result["changed_from_no_wiki"] = sum(by_id[q]["evidence"] != wiki_ref[q]["evidence"] for q in ids)
        accepted = {r["query_id"] for r in short}
        reference_short = {q for q in ids if wiki_ref[q]["evidence_tokens"] < a[q]["evidence_tokens"]}
        result["wiki_added_short"] = sorted(accepted - reference_short)
        result["wiki_lost_short"] = sorted(reference_short - accepted)
    return result


def summarize(rows: list[dict]) -> dict:
    if any(r["status"] != "ok" or (r["method"] == "e2" and r["route"] in ("api_error", "invalid_gate"))
           for r in rows):
        raise ValueError("One or more reader responses failed. See scored.jsonl.")
    groups = defaultdict(list)
    for row in rows:
        for ds in (row["dataset"], "all"):
            groups[ds, row["target_budget"], row["method"]].append(row)
    labels = defaultdict(dict)
    for (ds, qid), label in labels_for_cohort().items():
        labels[ds][qid] = label
    result = {}
    for (ds, budget, method), group in sorted(groups.items()):
        a = {r["query_id"]: r for r in rows if r["method"] == "raw_full" and ds in ("all", r["dataset"])}
        wiki_ref = ({r["query_id"]: r for r in groups[ds, budget, method.replace("_wiki", "_raw")]}
                    if method.endswith("_wiki") else None)
        result[f"{ds}/{budget}/{method}"] = measure(group, a, wiki_ref)
        if ds != "all":
            for metric, official in zip(("em", "f1"), official_scores(ds, group, labels[ds]), strict=True):
                local = mean(r[metric] for r in group)
                if not isclose(official, round(local, 3) if ds == "musique" else local, abs_tol=1e-12):
                    raise ValueError(f"Official {metric} differs for {ds}/{budget}/{method}")
    actual = [r for r in rows if r["method"] != "e2" and not r["shared_api_result"]]
    actual.extend(r["gate_api"] for r in rows if r["method"] == "e2" and r["gate_api"])
    return {"conditions": result, "scored_rows": len(rows), "official_checks": 2 * (2 + len(BUDGETS) * len(METHODS)),
            "actual_calls": len(actual), "actual_attempts": sum(r["api_attempts"] for r in actual),
            "actual_input_tokens": sum(r["api_prompt_tokens"] for r in actual),
            "actual_output_tokens": sum(r["api_completion_tokens"] for r in actual),
            "actual_price_cny": sum(r["list_price_cny"] for r in actual),
            "actual_api_p50_ms": median(r["api_ms"] for r in actual),
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def run(output: Path) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for name, expected in config["sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    _, corpus, queries = verify_freeze(SOURCE)
    dump(output / "summary.json", summarize(score_run(output, corpus, queries)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)
