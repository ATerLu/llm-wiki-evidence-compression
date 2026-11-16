"""Inventory archived question usage and known selection exposure."""

import argparse
from collections import defaultdict
from pathlib import Path

from external_compression import dump
from prepare import question_groups, question_keys, read_jsonl
from wiki_compiler_heldout import DATASETS, raw_rows


QUESTION_FILES = {"queries.jsonl", "predictions.jsonl", "scored.jsonl", "audit.jsonl", "evidence-audit.jsonl",
                  "diagnostic.jsonl", "lookup.jsonl", "rankings.jsonl"}


def inventory() -> dict:
    archive = defaultdict(set)
    per_file = {}
    files = sorted(p for p in Path("runs").glob("*/*.jsonl") if p.name in QUESTION_FILES)
    for path in files:
        hits = defaultdict(set)
        for row in read_jsonl(path):
            query_id = row.get("query_id") or (row.get("id") if path.name == "queries.jsonl" else None)
            if row.get("dataset") in DATASETS and isinstance(query_id, str):
                hits[row["dataset"]].add(query_id)
        per_file[path.as_posix()] = {dataset: len(ids) for dataset, ids in sorted(hits.items())}
        for dataset, ids in hits.items():
            archive[dataset].update(ids)

    datasets = {}
    for dataset in DATASETS:
        raw = raw_rows(dataset)
        by_id = {row["id"]: row for row in raw}
        groups = question_groups(raw, dataset)
        prepared = {}
        for path in sorted(Path("data").glob(f"*/{dataset}/queries.jsonl")):
            prepared[path.as_posix()] = read_jsonl(path)
        prepared_ids = {row["id"] for rows in prepared.values() for row in rows}
        dev_ids = by_id.keys()
        seen = (archive[dataset] | prepared_ids) & dev_ids
        seen_families = {family for family, ids in groups.items() if seen.intersection(ids)}
        train = prepared[f"data/processed/{dataset}/queries.jsonl"]
        train_keys = {question_keys(row, "hotpot")[0] for row in train}
        overlap = sorted(qid for qid, row in by_id.items() if question_keys(row, "hotpot")[0] in train_keys)
        datasets[dataset] = {
            "raw_dev_questions": len(raw), "raw_dev_families": len(groups),
            "archived_questions_all_splits": len(archive[dataset]),
            "archived_questions_in_dev": len(archive[dataset] & dev_ids),
            "prepared_questions_by_file": {path: len(rows) for path, rows in prepared.items()},
            "prepared_train_questions_with_archive_record": sum(row["id"] in archive[dataset] for row in train),
            "directly_listed_dev_questions": sorted(seen),
            "directly_listed_dev_families": sorted(seen_families),
            "other_dev_families_with_prior_screening_exposure": len(groups) - len(seen_families),
            "dev_normalized_exact_matches_with_exported_train": overlap,
            "independent_external_test_frozen": False,
        }
    return {"datasets": datasets, "archive_question_records_by_file": per_file,
            "prior_screening_sources": ["author_dependency_replay.py:screen",
                                       "scripts/audit_reuse_opportunity.py:screen_remaining_musique",
                                       "semantic_restoration.py: archived reuse-rescreen.json"],
            "limitations": ["Counts are based on ID-bearing artifacts and are lower bounds on direct use.",
                            "Remaining official dev families were eligible for prior author-cue screening.",
                            "Exact normalized question overlap is distinct from semantic duplicate or entity overlap."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    dump(parser.parse_args().output, inventory())
