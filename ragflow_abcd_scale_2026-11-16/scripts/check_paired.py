"""Cross-check paired answer scores and estimate uncertainty by question family."""

import argparse
import ast
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired import DATASETS, METHODS, labels_for
from prepare import jsonl, read_jsonl
import hotpot_evaluate_v1
import musique_evaluate_v1


def official_scores(dataset: str, rows: list[dict], labels: dict) -> tuple[float, float]:
    with TemporaryDirectory() as directory:
        prediction = Path(directory) / "predictions.json"
        reference = Path(directory) / "reference.json"
        ids = [r["query_id"] for r in rows]
        if dataset == "hotpot":
            prediction.write_text(json.dumps({
                "answer": {r["query_id"]: r["answer"] for r in rows}, "sp": {qid: [] for qid in ids},
            }), encoding="utf-8")
            reference.write_text(json.dumps([
                {"_id": qid, "answer": labels[qid]["answer"], "supporting_facts": labels[qid]["support"]}
                for qid in ids
            ]), encoding="utf-8")
            with redirect_stdout(io.StringIO()) as stdout:
                hotpot_evaluate_v1.eval(str(prediction), str(reference))
            scores = ast.literal_eval(stdout.getvalue())
            return scores["em"], scores["f1"]
        jsonl(prediction, ({
            "id": r["query_id"], "predicted_answer": r["answer"], "predicted_support_idxs": [],
            "predicted_answerable": True,
        } for r in rows))
        jsonl(reference, ({
            "id": qid, "answer": labels[qid]["answer"], "answer_aliases": labels[qid]["aliases"],
            "answerable": labels[qid]["answerable"],
            "paragraphs": [{"idx": i, "is_supporting": True} for i in labels[qid]["support"]],
        } for qid in ids))
        scores = musique_evaluate_v1.evaluate(str(prediction), str(reference))
        return scores["answer_em"], scores["answer_f1"]


def family_interval(reference: dict, candidate: dict) -> dict:
    families = defaultdict(list)
    for qid, row in reference.items():
        families[row["family_id"]].append(candidate[qid]["f1"] - row["f1"])
    clusters = [(sum(group), len(group)) for group in families.values()]
    rng = random.Random(20261110)
    values = []
    for _ in range(10000):
        drawn = rng.choices(clusters, k=len(clusters))
        values.append(sum(total for total, _ in drawn) / sum(n for _, n in drawn))
    values.sort()
    return {"questions": len(reference), "families": len(clusters),
            "largest_family": max(n for _, n in clusters),
            "f1_delta_ci95": [values[249], values[9749]]}


def check(run: Path, pilot: Path | None) -> None:
    rows = read_jsonl(run / "scored.jsonl")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    tuning = json.loads((run / "tuning.json").read_text(encoding="utf-8"))
    assert config["radius"] == tuning["radius"]
    assert len(rows) == len(config["plan"])
    assert all(r["status"] == "ok" and r["cache_n"] == 0 and not r["output_capped"] for r in rows)
    assert all(r["evidence_tokens"] <= config["budget"] and
               r["total_prompt_tokens"] + config["max_output"] <= config["context_limit"] for r in rows)
    for name, expected in config["code_sha256"].items():
        assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == expected

    intervals = {}
    for dataset in DATASETS:
        labels = labels_for(dataset)
        methods = {m: {r["query_id"]: r for r in rows if r["dataset"] == dataset and r["method"] == m}
                   for m in METHODS}
        assert all(methods[m].keys() == methods[METHODS[0]].keys() for m in METHODS)
        assert len(methods[METHODS[0]]) * len(METHODS) == sum(r["dataset"] == dataset for r in rows)
        tuned = {r["family_id"] for r in tuning["measurements"] if r["dataset"] == dataset}
        assert not tuned.intersection(r["family_id"] for r in methods[METHODS[0]].values())
        for method, predictions in methods.items():
            em, f1 = official_scores(dataset, list(predictions.values()), labels)
            expected = summary[dataset][method]
            for actual, key in ((em, "answer_em"), (f1, "answer_f1")):
                target = expected[key] if dataset == "hotpot" else round(expected[key], 3)
                assert math.isclose(actual, target, abs_tol=1e-12), (dataset, method, key, actual, target)
            print(dataset, method, "official scores agree:", em, f1)
        intervals[dataset] = {
            f"{a}__to__{b}": family_interval(methods[a], methods[b])
            for a, b in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2]))
        }

    if pilot:
        indexed = {(r["dataset"], r["query_id"], r["method"]): r for r in rows}
        original = read_jsonl(pilot / "scored.jsonl")
        pairs = [(r, indexed[(r["dataset"], r["query_id"], r["method"])]) for r in original]
        assert all(a["evidence"] == b["evidence"] for a, b in pairs)
        print("Identical pilot evidence:", len(pairs), "; changed answers:", sum(a["answer"] != b["answer"] for a, b in pairs))
    (run / "family_bootstrap.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    print("Checked token caps, cache state, development separation, full pairing and source hashes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    check(args.run, args.pilot)
