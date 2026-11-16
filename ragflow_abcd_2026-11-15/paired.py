"""Tune neighbor width on train development questions and run paired assembly comparisons."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import mean

from baseline import CONTEXT_LIMIT, MAX_OUTPUT, SYSTEM, Reader, pack_sentences, run_query, sentence_blocks
from prepare import jsonl, read_jsonl
from score import official_answer, summarize

DATA = Path("data/processed")
SAMPLE_SOURCE = Path("runs/week1-family-calibration-cuda2/config.json")
DATASETS = ("hotpot", "musique")
METHODS = ("lexical_paragraph", "lexical_sentence", "neighbor_sentence")
BUDGET = 1024
SEED = 20261110


def inputs(dataset: str, root: Path = DATA) -> tuple[dict, dict]:
    directory = root / dataset
    queries = {r["id"]: r for r in read_jsonl(directory / "queries.jsonl")}
    corpus = {r["id"]: r for r in read_jsonl(directory / "corpus.jsonl")}
    return queries, corpus


def labels_for(dataset: str, root: Path = DATA) -> dict:
    return {r["id"]: r for r in read_jsonl(root / dataset / "labels.jsonl")}


def support_sets(dataset: str, query: dict, documents: list[dict], label: dict,
                 selected: dict[int, set[int]]) -> tuple[set, set, set]:
    reference = {tuple(x) for x in label["support"]} if dataset == "hotpot" else set(label["support"])
    strict, touched = set(), set()
    for doc_index, indices in selected.items():
        doc = documents[doc_index]
        key = query["candidates"][doc_index]["key"]
        nonempty = {s["index"] for s in doc["spans"] if doc["text"][s["start"]:s["end"]].strip()}
        retained = indices & nonempty
        if dataset == "hotpot":
            strict.update((key, index) for index in retained)
        else:
            if retained:
                touched.add(key)
            if nonempty and nonempty <= indices:
                strict.add(key)
    return reference, strict, strict if dataset == "hotpot" else touched


def tune(reader: Reader, output: Path) -> None:
    # Choose a window from development labels only. The paired sample is held out of this step.
    measurements = []
    for dataset in DATASETS:
        queries, corpus = inputs(dataset)
        splits = json.loads((DATA / dataset / "splits.json").read_text(encoding="utf-8"))["splits"]
        ids = splits["development"].copy()
        random.Random(SEED).shuffle(ids)
        representatives = {}
        for qid in ids:
            representatives.setdefault(queries[qid]["family_id"], qid)
        labels = labels_for(dataset)
        for qid in list(representatives.values())[:8]:
            query = queries[qid]
            documents = [corpus[c["id"]] for c in query["candidates"]]
            for radius in (1, 2):
                selected, _, count = pack_sentences(reader, query["question"], documents, BUDGET, radius)
                ref, strict, touched = support_sets(dataset, query, documents, labels[qid], selected)
                measurements.append({
                    "dataset": dataset, "id": qid, "family_id": query["family_id"], "radius": radius,
                    "support_recall": len(ref & strict) / len(ref),
                    "touched_recall": len(ref & touched) / len(ref), "tokens": count,
                })
        print(f"development {dataset} done", flush=True)
    # MuSiQue has paragraph annotations. A touched paragraph is only a development proxy.
    objectives = {
        radius: mean(r["support_recall"] if r["dataset"] == "hotpot" else r["touched_recall"]
                     for r in measurements if r["radius"] == radius)
        for radius in (1, 2)
    }
    radius = max(objectives, key=lambda r: (objectives[r], -r))
    output.write_text(json.dumps({
        "split": "development", "seed": SEED, "budget": BUDGET, "radius": radius,
        "objective": "mean Hotpot sentence recall and MuSiQue reference paragraph touch recall; tie selects smaller",
        "objectives": objectives, "measurements": measurements,
    }, indent=2), encoding="utf-8")
    print("selected neighbor radius", radius, objectives, flush=True)


def frozen_sample(scope: str) -> list[tuple[str, str]]:
    if scope == "pilot":
        sample = [(d, q) for d, q, _ in json.loads(SAMPLE_SOURCE.read_text(encoding="utf-8"))["plan"]]
        expected = 15
    else:
        sample = [(dataset, qid) for dataset in DATASETS for qid in
                  json.loads((DATA / dataset / "splits.json").read_text(encoding="utf-8"))["splits"]["smoke"]]
        random.Random(SEED).shuffle(sample)
        expected = 100
    if len(set(sample)) != len(sample) or any(sum(d == dataset for d, _ in sample) != expected
                                              for dataset in DATASETS):
        raise ValueError("Unexpected number of unique questions in frozen sample")
    return sample


def run(reader: Reader, output: Path, radius: int, scope: str) -> None:
    sample = frozen_sample(scope)
    sources = {dataset: inputs(dataset) for dataset in DATASETS}
    rng = random.Random(SEED)
    plan = []
    for dataset, qid in sample:
        methods = list(METHODS)
        rng.shuffle(methods)
        plan.extend([dataset, qid, method] for method in methods)
    config = {
        "start_utc": datetime.now(timezone.utc).isoformat(), "budget": BUDGET, "radius": radius,
        "seed": SEED, "plan": plan, "sample_scope": scope, "system": SYSTEM,
        "sample_source": str(SAMPLE_SOURCE) if scope == "pilot" else "frozen train smoke splits",
        "max_output": MAX_OUTPUT, "context_limit": CONTEXT_LIMIT,
    }
    snapshots = output / "source"
    snapshots.mkdir()
    config["code_sha256"] = {}
    for name in ("baseline.py", "paired.py", "prepare.py", "score.py", "run_week1.ps1"):
        content = Path(name).read_bytes()
        (snapshots / name).write_bytes(content)
        config["code_sha256"][name] = hashlib.sha256(content).hexdigest()
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    radii = dict(zip(METHODS, (None, 0, radius), strict=True))
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as file:
        for number, (dataset, qid, method) in enumerate(plan, 1):
            queries, corpus = sources[dataset]
            row = run_query(reader, output.name, dataset, queries[qid], corpus, BUDGET,
                            method=method, sentence_radius=radii[method])
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            print(f"{number}/{len(plan)} {dataset} {qid} {method} {row['status']} "
                  f"{round(row['total_ms'])} ms", flush=True)
    config["end_utc"] = datetime.now(timezone.utc).isoformat()
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def score_row(row: dict, query: dict, corpus: dict, label: dict) -> dict:
    documents = [corpus[c["id"]] for c in query["candidates"]]
    if row["candidate_ids"] != [d["id"] for d in documents]:
        raise ValueError("Candidate pool mismatch")
    selected = {i: set(indices) for i, indices in row.get("selected_blocks", [])}
    if sentence_blocks(documents, selected) != row.get("evidence", ""):
        raise ValueError("Rendered evidence does not match its original spans")
    if row.get("evidence_tokens", 0) > row["budget"]:
        raise ValueError("Evidence token budget exceeded")
    ref, strict, touched = support_sets(row["dataset"], query, documents, label, selected)
    pool = {i: {s["index"] for s in doc["spans"]} for i, doc in enumerate(documents)}
    _, available, _ = support_sets(row["dataset"], query, documents, label, pool)
    em, f1 = official_answer(row["dataset"], row["answer"], label) if row["status"] == "ok" else (0.0, 0.0)
    return {**row, "em": em, "f1": f1, "candidate_support_complete": ref <= available,
            "selected_support_complete": ref <= strict, "all_support_touched": ref <= touched,
            "strict_recall": len(ref & strict) / len(ref), "touched_recall": len(ref & touched) / len(ref)}


def evaluate(output: Path) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    rows = read_jsonl(output / "predictions.jsonl")
    if "end_utc" not in config or [[r["dataset"], r["query_id"], r["method"]] for r in rows] != config["plan"]:
        raise ValueError("Predictions do not match the complete paired plan")
    results, summary = [], {}
    for dataset in DATASETS:
        queries, corpus = inputs(dataset)
        labels = labels_for(dataset)
        scored = [score_row(r, queries[r["query_id"]], corpus, labels[r["query_id"]])
                  for r in rows if r["dataset"] == dataset]
        results.extend(scored)
        groups = {m: {r["query_id"]: r for r in scored if r["method"] == m} for m in METHODS}
        summary[dataset] = {}
        for method, group in groups.items():
            stats = summarize(list(group.values()))
            stats["all_support_touched"] = sum(r["all_support_touched"] for r in group.values())
            stats["strict_recall"] = mean(r["strict_recall"] for r in group.values())
            stats["touched_recall"] = mean(r["touched_recall"] for r in group.values())
            summary[dataset][method] = stats
        comparisons = {}
        for old, new in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
            pairs = [(previous, groups[new][qid]) for qid, previous in groups[old].items()]
            delta = [b["f1"] - a["f1"] for a, b in pairs]
            comparisons[f"{old}__to__{new}"] = {
                "f1_delta": mean(delta), "improved": sum(d > 1e-9 for d in delta),
                "worsened": sum(d < -1e-9 for d in delta),
                "tied": sum(abs(d) <= 1e-9 for d in delta),
                "complete_support_gained": sum(not a["selected_support_complete"] and b["selected_support_complete"] for a, b in pairs),
                "complete_support_lost": sum(a["selected_support_complete"] and not b["selected_support_complete"] for a, b in pairs),
            }
        summary[dataset]["paired"] = comparisons
    jsonl(output / "scored.jsonl", results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote paired results:", output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tune", "run", "score"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--scope", choices=["pilot", "smoke"], default="pilot")
    args = parser.parse_args()
    if args.mode == "score":
        evaluate(args.output)
    elif args.mode == "tune":
        tune(Reader(args.port), args.output / "tuning.json")
    else:
        radius = json.loads((args.output / "tuning.json").read_text(encoding="utf-8"))["radius"]
        run(Reader(args.port), args.output, radius, args.scope)
