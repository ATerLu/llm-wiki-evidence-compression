"""Check three-budget cloud results and report matched comparisons to the raw context."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from cloud_experiment import DATASETS, SWEEP_BUDGETS, data_for
from paired import labels_for
from prepare import read_jsonl
from scripts.check_paired import family_interval, official_scores

METHODS = ("lexical_sentence", "wiki_ranked")
NEAR_A = 0.05


def measure(group: dict, common: list[str]) -> dict:
    rows = [group[q] for q in common]
    return {
        "em": mean(r["em"] for r in rows), "f1": mean(r["f1"] for r in rows),
        "full_support": sum(r["selected_support_complete"] for r in rows),
        "all_support_touched": sum(r["all_support_touched"] for r in rows),
        "api_prompt_p50": median(r["api_prompt_tokens"] for r in rows),
        "api_prompt_total": sum(r["api_prompt_tokens"] for r in rows),
    }


def difference(reference: dict, candidate: dict, common: list[str]) -> dict:
    deltas = [candidate[q]["f1"] - reference[q]["f1"] for q in common]
    return {
        "delta": mean(deltas), "improved": sum(d > 1e-9 for d in deltas),
        "worsened": sum(d < -1e-9 for d in deltas),
        **family_interval({q: reference[q] for q in common}, {q: candidate[q] for q in common}),
    }


def compare(a: Path, sweep: Path) -> dict:
    config_a, config_s = (json.loads((p / "config.json").read_text(encoding="utf-8")) for p in (a, sweep))
    keys = ("scope", "model", "endpoint", "temperature", "thinking", "system", "max_output", "tokenizer_sha256")
    assert all(config_a[k] == config_s[k] for k in keys)
    raw, runs = (read_jsonl(p / "scored.jsonl") for p in (a, sweep))
    result = {
        "raw_requests": config_a["unique_requests"], "sweep_requests": config_s["unique_requests"],
        "sweep_rows": len(runs), "cost_cny_list_price": sum(r.get("list_price_cny", 0) for r in raw + runs),
        "near_a_margin": NEAR_A, "datasets": {},
    }
    for dataset in DATASETS:
        labels = labels_for(dataset, data_for(config_a["scope"]))
        baseline = {r["query_id"]: r for r in raw if r["dataset"] == dataset}
        by_budget = {
            budget: {method: {r["query_id"]: r for r in runs
                              if (r["dataset"], r["method"], r["budget"]) == (dataset, method, budget)}
                     for method in METHODS}
            for budget in SWEEP_BUDGETS
        }
        expected = {q for q, r in baseline.items() if r["status"] == "ok"}
        conditions = [baseline, *(g for groups in by_budget.values() for g in groups.values())]
        for i, group in enumerate(conditions):
            if i:
                assert group.keys() == expected
            scores = official_scores(dataset, list(group.values()), labels)
            for score, key in zip(scores, ("em", "f1"), strict=True):
                computed = mean(row[key] for row in group.values())
                if dataset == "musique":
                    computed = round(computed, 3)
                assert math.isclose(score, computed, abs_tol=1e-12)
        common = [q for q in baseline if all(q in group and group[q]["status"] == "ok" for group in conditions)]
        a_stats = measure(baseline, common)
        budget_stats = {}
        for budget, groups in by_budget.items():
            off, on = (groups[method] for method in METHODS)
            unchanged = [q for q in common if off[q]["evidence"] == on[q]["evidence"]]
            assert all(off[q]["api_response_id"] == on[q]["api_response_id"] and off[q]["answer"] == on[q]["answer"]
                       for q in unchanged)
            current = measure(on, common)
            vs_a = difference(baseline, on, common)
            budget_stats[budget] = {
                "off": measure(off, common), "on": current,
                "input_token_reduction_vs_a": 1 - current["api_prompt_total"] / a_stats["api_prompt_total"],
                "on_vs_a": vs_a, "on_vs_off": difference(off, on, common),
                "near_a_point_estimate": vs_a["delta"] >= -NEAR_A,
                "near_a_ci_lower": vs_a["f1_delta_ci95"][0] >= -NEAR_A,
                "same_input": len(unchanged),
                "changed_input": len(common) - len(unchanged),
            }
        result["datasets"][dataset] = {
            "common_n": len(common), "a": a_stats, "budgets": budget_stats,
            "a_failures": [[q, r.get("error", r["status"])] for q, r in baseline.items() if r["status"] != "ok"],
            "sweep_failures": [[q, r["method"], r["budget"], r.get("error", r["status"])]
                               for group in conditions[1:] for q, r in group.items() if r["status"] != "ok"],
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()
    output = args.sweep / "comparison.json"
    output.write_text(json.dumps(compare(args.a, args.sweep), indent=2), encoding="utf-8")
    print(output)
