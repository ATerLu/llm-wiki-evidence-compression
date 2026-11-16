"""Verify and compare the original cloud run and matched Wiki runs."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from cloud_experiment import DATASETS, METHODS_B, METHODS_V
from paired import inputs, labels_for
from prepare import read_jsonl
from scripts.check_paired import family_interval, official_scores


def index(rows: list[dict], dataset: str, method: str) -> dict:
    return {r["query_id"]: r for r in rows if r["dataset"] == dataset and r["method"] == method}


def quality(rows: list[dict]) -> dict:
    return {
        "n": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
        "em": mean(r["em"] for r in rows), "f1": mean(r["f1"] for r in rows),
        "full_support": sum(r["selected_support_complete"] for r in rows),
        "all_support_touched": sum(r["all_support_touched"] for r in rows),
        "prompt_tokens_p50": median(r["api_prompt_tokens"] for r in rows if "api_prompt_tokens" in r),
        "evidence_tokens_p50": median(r["evidence_tokens"] for r in rows),
        "list_price_cny": sum(r.get("list_price_cny", 0) for r in rows),
    }


def compare(a: Path, b: Path) -> dict:
    a_rows, b_rows = [read_jsonl(path / "scored.jsonl") for path in (a, b)]
    a_config, b_config = [json.loads((path / "config.json").read_text(encoding="utf-8")) for path in (a, b)]
    config_keys = ("model", "endpoint", "temperature", "thinking", "system", "max_output", "tokenizer_sha256")
    assert all(a_config[k] == b_config[k] for k in config_keys)
    validation = b_config["stage"] == "V"
    method_names = METHODS_V if validation else METHODS_B
    enabled_name = "wiki_bridge" if validation else "wiki_dependency"
    result = {"a_workload_n": len(a_rows), "b_workload_n": len(b_rows),
              "total_list_price_cny": sum(r.get("list_price_cny", 0) for r in a_rows + b_rows),
              "datasets": {}}
    for dataset in DATASETS:
        raw = index(a_rows, dataset, "raw_full")
        methods = {method: index(b_rows, dataset, method) for method in method_names}
        assert all(rows.keys() == {qid for qid, r in raw.items() if r["status"] == "ok"}
                   for rows in methods.values())
        queries, corpus = inputs(dataset)
        for qid, row in raw.items():
            docs = [corpus[c["id"]] for c in queries[qid]["candidates"]]
            assert row["selected_blocks"] == [[i, [s["index"] for s in doc["spans"]]] for i, doc in enumerate(docs)]
        groups = {"raw_full": raw, **methods}
        labels = labels_for(dataset)
        for method, rows in groups.items():
            group = list(rows.values())
            em, f1 = official_scores(dataset, group, labels)
            for actual, key in ((em, "em"), (f1, "f1")):
                computed = mean(r[key] for r in group)
                if dataset == "musique":
                    computed = round(computed, 3)
                assert math.isclose(actual, computed, abs_tol=1e-12), (dataset, method, key)
        common = [qid for qid in raw if all(qid in rows and rows[qid]["status"] == "ok" for rows in groups.values())]
        enabled, disabled = methods[enabled_name], methods["lexical_sentence"]
        changed = [qid for qid in common if enabled[qid]["evidence"] != disabled[qid]["evidence"]]
        activated = [qid for qid in common if any(edge["decision"] == "added" for edge in enabled[qid]["wiki_edges"])]
        a_tokens = sum(raw[qid]["api_prompt_tokens"] for qid in common)
        b_tokens = sum(enabled[qid]["api_prompt_tokens"] for qid in common)
        paired = {}
        for method in ("raw_full", *(m for m in method_names if m != enabled_name)):
            first, last = [{qid: group[qid] for qid in common} for group in (groups[method], enabled)]
            diffs = [last[qid]["f1"] - first[qid]["f1"] for qid in common]
            paired[method] = {
                "f1_delta": mean(diffs), "up": sum(x > 1e-9 for x in diffs),
                "down": sum(x < -1e-9 for x in diffs), "ties": sum(abs(x) <= 1e-9 for x in diffs),
                **family_interval(first, last),
            }
        result["datasets"][dataset] = {
            "a_all": quality(list(raw.values())), "common_n": len(common),
            "common": {method: quality([rows[qid] for qid in common]) for method, rows in groups.items()},
            "common_api_prompt_tokens_reduction": 1 - b_tokens / a_tokens,
            "wiki_proof_added": activated, "wiki_context_changed": changed,
            "wiki_same_context_different_answer": [
                qid for qid in common if enabled[qid]["evidence"] == disabled[qid]["evidence"]
                and enabled[qid]["answer"] != disabled[qid]["answer"]
            ],
            "paired_to_b": paired,
        }
        if validation:
            shared = [qid for qid in common if qid not in changed]
            assert all(enabled[qid]["api_response_id"] == disabled[qid]["api_response_id"]
                       and enabled[qid]["answer"] == disabled[qid]["answer"] for qid in shared)
            result["datasets"][dataset]["shared_same_input_n"] = len(shared)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    result = compare(args.a, args.b)
    path = args.b / "comparison.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)
