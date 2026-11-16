"""Freeze balanced development cohorts before any retrieval or scoring."""

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from itertools import zip_longest
import json
from pathlib import Path
import random

from pyarrow.parquet import ParquetFile

from prepare import adapt, jsonl, question_groups, read_jsonl
from run_abcd import digest, write_json


DATA = Path("data")
TOPIC = Path("../topic2")
SEED = 20261117
DATASETS = ("hotpot", "musique")


def compiler():
    spec = spec_from_file_location("offline_compiler", DATA / "compile-source.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_ids():
    known = {ds: set() for ds in DATASETS}
    historical = json.loads((TOPIC / "runs/data-use-audit-20261111.json").read_text(encoding="utf-8"))
    for ds in DATASETS:
        known[ds].update(historical["datasets"][ds]["directly_listed_dev_questions"])
        for path in (TOPIC / "data").glob(f"*/{ds}/queries.jsonl"):
            known[ds].update(row["id"] for row in read_jsonl(path))
    names = {"queries.jsonl", "scored.jsonl", "predictions.jsonl", "audit.jsonl", "retrieval.jsonl",
             "native-full-retrieval.jsonl", "native-raw-snapshot-retrieval.jsonl", "diagnostic.jsonl",
             "evidence-audit.jsonl", "rankings.jsonl", "lookup.jsonl"}
    paths = [p for p in (TOPIC / "runs").glob("*/*.jsonl") if p.name in names]
    paths.append(Path("../ragflow_abcd_2026-11-15/data/queries.jsonl"))
    unclassified = set()
    for path in sorted(paths):
        for row in read_jsonl(path):
            qid = row.get("query_id") or (row.get("id") if path.name == "queries.jsonl" else None)
            if isinstance(qid, str):
                known.get(row.get("dataset"), unclassified).add(qid)
    return {ds: ids | unclassified for ds, ids in known.items()}


def select(rows, dataset, history):
    groups = question_groups(rows, dataset)
    raw = {row["id"]: row for row in rows}
    excluded = {f for f, ids in groups.items() if history.intersection(ids)}
    rng = random.Random(SEED)
    families = sorted(groups.keys() - excluded)
    rng.shuffle(families)
    buckets = [[(f, q) for q in sorted(groups[f])] for f in families]
    for bucket in buckets:
        rng.shuffle(bucket)
    chosen = [entry for layer in zip_longest(*buckets) for entry in layer if entry is not None][:250]
    if len(chosen) != 250:
        raise ValueError(f"{dataset}: only {len(chosen)} eligible questions")
    return raw, chosen, {"earlier_ids": sorted(history & raw.keys()),
                         "excluded_official_families": sorted(excluded), "eligible_families": len(families),
                         "selected_families": len({f for f, _ in chosen}), "family_sizes": dict(Counter(f for f, _ in chosen))}


def freeze():
    known = historical_ids()
    raw_dir = TOPIC / "data/raw"
    hotpot = ParquetFile(raw_dir / "hotpot-validation-00000-of-00001.parquet")
    rows_by_dataset = {"hotpot": [row for batch in hotpot.iter_batches() for row in batch.to_pylist()],
                       "musique": read_jsonl(raw_dir / "musique_ans_v1.0_dev.jsonl")}
    corpus, queries, groups, exclusions = {}, [], {}, {}
    for ds, rows in rows_by_dataset.items():
        raw, chosen, exclusions[ds] = select(rows, ds, known[ds])
        labels = []
        groups[ds] = [qid for _, qid in chosen]
        for family, qid in chosen:
            query, label, docs = adapt(raw[qid], ds, family)
            queries.append({"dataset": ds, **query})
            labels.append(label)
            corpus.update((doc["id"], doc) for doc in docs)
        jsonl(DATA / f"labels-{ds}.jsonl", labels)
    (DATA / "cohorts.json").write_text(json.dumps({"200": [q for ids in groups.values() for q in ids[:100]],
                                                   "additional300": [q for ids in groups.values() for q in ids[100:]]},
                                                  indent=2), encoding="utf-8")
    jsonl(DATA / "queries.jsonl", queries)
    jsonl(DATA / "corpus.jsonl", corpus.values())
    source = DATA / "compile-source.py"
    cc = compiler()
    files = ("cohorts.json", "queries.jsonl", "corpus.jsonl", *(f"labels-{ds}.jsonl" for ds in DATASETS))
    config = {"seed": SEED, "per_dataset": 250, "queries": len(queries), "pages": len(corpus),
              "compile_model": cc.MODEL, "system": cc.SYSTEM, "batch": cc.BATCH, "max_output": cc.MAX_OUTPUT,
              "allow_within_cohort_family_reuse": True, "compile_source_sha256": digest(source),
              "input_sha256": {file: digest(DATA / file) for file in files}, "exclusions": exclusions}
    write_json(DATA / "freeze.json", config)
    print(f"Frozen {len(queries)} questions / {len(corpus)} pages; distinct families: "
          + ", ".join(f"{ds} {record['selected_families']}/{record['eligible_families']}" for ds, record in exclusions.items()))


if __name__ == "__main__":
    freeze()
