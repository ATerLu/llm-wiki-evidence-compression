"""Fit coverage routing on development and measure a frozen policy on held-out questions."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

from cloud_experiment import data_for
from paired import labels_for
from prepare import read_jsonl
from scripts.check_paired import official_scores
from sweep_compare import difference, measure

LOW = "guard_2048"
HIGH = "guard_2560"
RAW = "raw_full"
THRESHOLDS = (0.70, 0.80, 0.90, 0.95, None)
MARGIN = 0.05


def eligible(row: dict, threshold: float) -> bool:
    guard = row["guard"]
    named = set(guard["direct_pages"])
    covered = set(guard["direct_pages_covered"])
    minimum = min(1.0, threshold + (0.05 if not named else 0))
    return named <= covered and row["evidence_tokens"] >= row["raw_evidence_tokens"] * minimum


def route(conditions: dict, qid: str, threshold: float | None) -> str:
    if threshold is None:
        return RAW
    for mode in (LOW, HIGH):
        if eligible(conditions[mode][qid], threshold):
            return mode
    return RAW


def load_conditions(a: Path, b: Path) -> tuple[dict, dict, list]:
    config_a, config_b = [json.loads((p / "config.json").read_text(encoding="utf-8")) for p in (a, b)]
    for key in ("scope", "model", "endpoint", "temperature", "thinking", "system", "max_output", "tokenizer_sha256"):
        assert config_a[key] == config_b[key]
    a_rows, b_rows = [read_jsonl(p / "scored.jsonl") for p in (a, b)]
    failures = [{k: r[k] for k in ("dataset", "query_id", "method", "budget", "error")}
                for r in a_rows + b_rows if r["status"] == "error"]
    conditions, scopes = {}, {}
    for dataset in ("hotpot", "musique"):
        raw = {r["query_id"]: r for r in a_rows if r["dataset"] == dataset}
        target = [r for r in b_rows if r["dataset"] == dataset]
        groups = {RAW: raw}
        for method, budget in sorted({(r["method"], r["budget"]) for r in target}):
            key = f"{'guard' if method == 'wiki_guarded' else 'prior'}_{budget}"
            groups[key] = {r["query_id"]: r for r in target if (r["method"], r["budget"]) == (method, budget)}
            assert groups[key].keys() == {qid for qid, row in raw.items() if row["status"] == "ok"}
        common = [qid for qid in raw if all(qid in g and g[qid]["status"] == "ok" for g in groups.values())]
        labels = labels_for(dataset, data_for(config_a["scope"]))
        for group in groups.values():
            rows = list(group.values())
            scores = official_scores(dataset, rows, labels)
            for score, name in zip(scores, ("em", "f1"), strict=True):
                computed = mean(r[name] for r in rows)
                assert math.isclose(score, round(computed, 3) if dataset == "musique" else computed, abs_tol=1e-12)
        for qid in common:
            same = {}
            for group in groups.values():
                r = group[qid]
                result_id = same.setdefault(r["evidence"], r["api_response_id"])
                assert result_id == r["api_response_id"]
                if r.get("guard"):
                    selected = {i: set(ids) for i, ids in r["selected_blocks"]}
                    assert all(j in selected.get(i, set()) for i, j in r["guard"]["protected"])
        conditions[dataset] = groups
        scopes[dataset] = common
    run = {"scope": config_a["scope"], "requests": config_a["unique_requests"] + config_b["unique_requests"],
           "rows": len(a_rows) + len(b_rows), "failures": failures,
           "list_price_cny": sum(r.get("list_price_cny", 0) for r in a_rows + b_rows)}
    return conditions, scopes, run


def adaptive(conditions: dict, qids: list[str], threshold: float | None) -> tuple[dict, dict]:
    chosen = {qid: route(conditions, qid, threshold) for qid in qids}
    return {qid: conditions[mode][qid] for qid, mode in chosen.items()}, chosen


def fit(a: Path, b: Path, output: Path) -> None:
    groups, scopes, run = load_conditions(a, b)
    assert run["scope"] == "guard_development"
    conditions, common = groups["musique"], scopes["musique"]
    raw = measure(conditions[RAW], common)
    options = []
    for threshold in THRESHOLDS:
        group, choices = adaptive(conditions, common, threshold)
        stats = measure(group, common)
        options.append({"threshold": threshold, "f1": stats["f1"],
                        "gap_vs_raw": stats["f1"] - raw["f1"], "tokens": stats["api_prompt_total"],
                        "routes": dict(Counter(choices.values()))})
    chosen = min((x for x in options if x["gap_vs_raw"] >= -MARGIN),
                 key=lambda x: (x["tokens"], -x["f1"]))
    output.write_text(json.dumps({
        "scope": run["scope"], "n": len(common), "margin": MARGIN, "selected_threshold": chosen["threshold"],
        "raw_f1": raw["f1"], "options": options,
        "development_fixed": {m: measure(g, common) for m, g in conditions.items()},
        "run": run,
    }, indent=2), encoding="utf-8")
    print(json.dumps(chosen))


def verify(a: Path, b: Path, fit_file: Path) -> None:
    groups, scopes, run = load_conditions(a, b)
    assert run["scope"] == "guard_heldout"
    threshold = json.loads(fit_file.read_text(encoding="utf-8"))["selected_threshold"]
    result = {"run": run, "threshold": threshold, "datasets": {}}
    decisions = []
    for dataset, conditions in groups.items():
        common = scopes[dataset]
        raw = measure(conditions[RAW], common)
        if dataset == "musique":
            conditions["adaptive"], routes = adaptive(conditions, common, threshold)
            decisions = [{"query_id": q, "family_id": conditions[RAW][q]["family_id"], "route": mode,
                          "answer": conditions[mode][q]["answer"], "f1": conditions[mode][q]["f1"],
                          "api_prompt_tokens": conditions[mode][q]["api_prompt_tokens"]}
                         for q, mode in routes.items()]
        rows = {}
        for name, group in conditions.items():
            stats = measure(group, common)
            stats["input_saving"] = 1 - stats["api_prompt_total"] / raw["api_prompt_total"]
            stats["vs_a"] = difference(conditions[RAW], group, common)
            stats["same_input_as_a"] = sum(group[q]["evidence"] == conditions[RAW][q]["evidence"] for q in common)
            if name.startswith("guard_"):
                old = conditions.get(name.replace("guard_", "prior_"))
                stats["conflicts"] = sum(bool(group[q]["guard"]["conflicts"]) for q in common)
                if old is not None:
                    stats["vs_unguarded"] = difference(old, group, common)
                    stats["changed_vs_unguarded"] = sum(old[q]["evidence"] != group[q]["evidence"] for q in common)
            rows[name] = stats
        result["datasets"][dataset] = {"n": len(common), "methods": rows}
    result["adaptive_routes"] = dict(Counter(r["route"] for r in decisions))
    (b / "adaptive-decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    (b / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(b / "comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fit", "verify"])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("policy", type=Path)
    args = parser.parse_args()
    (fit if args.mode == "fit" else verify)(args.a, args.b, args.policy)
