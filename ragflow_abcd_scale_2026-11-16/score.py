"""Score saved predictions with pinned official answer metrics and audit input evidence."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

from prepare import jsonl, read_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from hotpot_evaluate_v1 import exact_match_score, f1_score
from metrics.answer import AnswerMetric


def audit(row: dict, query: dict, label: dict, corpus: dict) -> tuple[bool, bool]:
    selected = row.get("selected_candidates", [])
    rendered = "\n\n".join(f"[{corpus[c['id']]['title']}]\n{corpus[c['id']]['text']}" for c in selected)
    if rendered != row.get("evidence", "") or any(c not in query["candidates"] for c in selected):
        raise ValueError(f"{row['query_id']}: recorded evidence cannot be reconstructed")

    def present(candidates: list[dict]) -> set:
        if row["dataset"] == "musique":
            return {c["key"] for c in candidates if corpus[c["id"]]["text"]}
        return {
            (c["key"], span["index"])
            for c in candidates for span in corpus[c["id"]]["spans"]
            if span["end"] > span["start"]
        }

    support = {tuple(item) for item in label["support"]} if row["dataset"] == "hotpot" else set(label["support"])
    return support <= present(query["candidates"]), support <= present(selected)


def official_answer(dataset: str, answer: str, label: dict) -> tuple[float, float]:
    if dataset == "hotpot":
        return float(exact_match_score(answer, label["answer"])), f1_score(answer, label["answer"])[0]
    metric = AnswerMetric()
    metric(answer, [label["answer"], *label["aliases"]])
    em, f1 = metric.get_metric()
    return float(em), float(f1)


def timing(rows: list[dict], name: str) -> dict | None:
    values = sorted(row[name] for row in rows if row.get(name) is not None)
    if not values:
        return None
    return {
        "n": len(values), "mean": round(mean(values), 3), "p50": round(median(values), 3),
        "p95": round(values[math.ceil(0.95 * len(values)) - 1], 3),
    }


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    eligible = [r for r in rows if r["candidate_support_complete"] and "evidence" in r]
    return {
        "n": len(rows), "ok": len(ok), "answer_em": mean(r["em"] for r in rows),
        "answer_f1": mean(r["f1"] for r in rows),
        "evidence_assembled": sum("evidence" in r for r in rows),
        "candidate_support_complete": sum(r["candidate_support_complete"] for r in rows),
        "selected_support_complete": sum(r["selected_support_complete"] for r in rows),
        "support_loss_denominator": len(eligible),
        "support_loss_n": sum(not r["selected_support_complete"] for r in eligible),
        "nonzero_cache_n": sum(r.get("cache_n", 0) != 0 for r in rows),
        "output_capped": sum(bool(r.get("output_capped")) for r in rows),
        "timings_ok": {key: timing(ok, key) for key in (
            "retrieval_ms", "assembly_ms", "prompt_ms", "generation_ms", "model_ttft_ms",
            "query_ttft_ms", "total_ms", "evidence_tokens", "total_prompt_tokens", "output_tokens",
        )},
    }


def score(run: Path, data: Path) -> None:
    predictions = read_jsonl(run / "predictions.jsonl")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    completed = [[row["dataset"], row["query_id"], row["budget"]] for row in predictions]
    if completed != config["plan"] or "end_utc" not in config:
        raise ValueError("Run is incomplete or differs from the frozen run plan")
    scored = []
    for dataset in ("hotpot", "musique"):
        directory = data / dataset
        queries = {row["id"]: row for row in read_jsonl(directory / "queries.jsonl")}
        labels = {row["id"]: row for row in read_jsonl(directory / "labels.jsonl")}
        corpus = {row["id"]: row for row in read_jsonl(directory / "corpus.jsonl")}
        for row in predictions:
            if row["dataset"] != dataset:
                continue
            qid = row["query_id"]
            label = labels[qid]
            em, f1 = official_answer(dataset, row["answer"], label) if row["status"] == "ok" else (0.0, 0.0)
            candidate, selected = audit(row, queries[qid], label, corpus)
            scored.append({**row, "em": em, "f1": f1, "candidate_support_complete": candidate,
                           "selected_support_complete": selected})
    summary = {}
    for dataset in ("hotpot", "musique"):
        rows = [row for row in scored if row["dataset"] == dataset]
        summary[dataset] = summarize(rows)
        summary[dataset]["by_budget"] = {
            budget: summarize([r for r in rows if r["budget"] == budget])
            for budget in sorted({r["budget"] for r in rows})
        }
    jsonl(run / "scored.jsonl", scored)
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for dataset, stats in summary.items():
        print(dataset, {key: stats[key] for key in ("n", "ok", "answer_em", "answer_f1", "support_loss_n")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    score(args.run, args.data)
