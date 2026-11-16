"""Freeze a new official-dev subset after the selection rules have been set."""

from itertools import zip_longest
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import adapt, jsonl, question_groups, read_jsonl
from pyarrow.parquet import ParquetFile

SEED = 20261112
N = 100


def freeze(dataset: str, raw: Path, root: Path) -> None:
    if dataset == "hotpot":
        table = ParquetFile(raw / "hotpot-validation-00000-of-00001.parquet")
        rows = [row for batch in table.iter_batches() for row in batch.to_pylist()]
    else:
        rows = read_jsonl(raw / "musique_ans_v1.0_dev.jsonl")
    groups = question_groups(rows, dataset)
    families = {qid: family for family, ids in groups.items() for qid in ids}
    rng = random.Random(SEED)
    buckets = [sorted(ids) for _, ids in sorted(groups.items())]
    for ids in buckets:
        rng.shuffle(ids)
    rng.shuffle(buckets)
    chosen = [qid for batch in zip_longest(*buckets) for qid in batch if qid][:N]
    assert len(chosen) == len(set(chosen)) == N
    index = {r["id"]: r for r in rows}
    old = {r["id"] for r in read_jsonl(Path("data/processed") / dataset / "queries.jsonl")}
    assert not old.intersection(chosen)
    corpus, queries, labels = {}, [], []
    for qid in chosen:
        query, label, docs = adapt(index[qid], dataset, families[qid])
        queries.append(query)
        labels.append(label)
        corpus.update((d["id"], d) for d in docs)
    destination = root / dataset
    destination.mkdir(parents=True)
    jsonl(destination / "queries.jsonl", queries)
    jsonl(destination / "corpus.jsonl", corpus.values())
    jsonl(destination / "labels.jsonl", labels)
    (destination / "splits.json").write_text(json.dumps({
        "source_split": "official_dev", "seed": SEED, "source_n": len(rows),
        "splits": {"official_dev": chosen}, "sample_families": len({families[q] for q in chosen}),
    }, indent=2), encoding="utf-8")
    print(dataset, len(rows), N, len({families[q] for q in chosen}))


if __name__ == "__main__":
    for name in ("hotpot", "musique"):
        freeze(name, Path("data/raw"), Path("data/official-dev-sweep"))
