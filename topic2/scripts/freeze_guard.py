"""Freeze independent guard development and unrun official-dev families."""

from itertools import zip_longest
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import adapt, jsonl, question_groups, read_jsonl
from pyarrow.parquet import ParquetFile

SEED = 20261113


def round_robin(families: dict, size: int) -> list[str]:
    rng = random.Random(SEED)
    groups = [sorted(ids) for _, ids in sorted(families.items())]
    for ids in groups:
        rng.shuffle(ids)
    rng.shuffle(groups)
    sample = [qid for batch in zip_longest(*groups) for qid in batch if qid][:size]
    assert len(sample) == len(set(sample)) == size
    return sample


def save(root: Path, dataset: str, queries: list[dict], labels: list[dict], corpus: dict) -> None:
    dst = root / dataset
    dst.mkdir(parents=True)
    for name, records in (("queries", queries), ("labels", labels), ("corpus", corpus.values())):
        jsonl(dst / f"{name}.jsonl", records)
    families = len({r["family_id"] for r in queries})
    (dst / "splits.json").write_text(json.dumps({
        "source_split": root.name, "seed": SEED, "sample_families": families,
        "ids": [q["id"] for q in queries],
    }, indent=2), encoding="utf-8")
    print(root.name, dataset, len(queries), "questions;", families, "families")


def development(dataset: str) -> None:
    path = Path("data/processed") / dataset
    split = json.loads((path / "splits.json").read_text(encoding="utf-8"))["splits"]["development"]
    queries = {r["id"]: r for r in read_jsonl(path / "queries.jsonl")}
    groups = {}
    for qid in split:
        groups.setdefault(queries[qid]["family_id"], []).append(qid)
    sample = round_robin(groups, 40)
    selected = [queries[qid] for qid in sample]
    labels = {r["id"]: r for r in read_jsonl(path / "labels.jsonl")}
    corpus = {r["id"]: r for r in read_jsonl(path / "corpus.jsonl")}
    docs = {c["id"]: corpus[c["id"]] for r in selected for c in r["candidates"]}
    save(Path("data/guard-development"), dataset, selected, [labels[qid] for qid in sample], docs)


def heldout(dataset: str) -> None:
    raw = Path("data/raw")
    if dataset == "hotpot":
        file = ParquetFile(raw / "hotpot-validation-00000-of-00001.parquet")
        rows = [row for batch in file.iter_batches() for row in batch.to_pylist()]
    else:
        rows = read_jsonl(raw / "musique_ans_v1.0_dev.jsonl")
    earlier = read_jsonl(Path("data/official-dev-sweep") / dataset / "queries.jsonl")
    used_families = {r["family_id"] for r in earlier}
    groups = question_groups(rows, dataset)
    sample = round_robin({f: ids for f, ids in groups.items() if f not in used_families}, 100)
    mapping = {qid: f for f, ids in groups.items() for qid in ids}
    assert used_families.isdisjoint(mapping[q] for q in sample)
    by_id = {r["id"]: r for r in rows}
    queries, labels, corpus = [], [], {}
    for qid in sample:
        query, label, docs = adapt(by_id[qid], dataset, mapping[qid])
        queries.append(query)
        labels.append(label)
        corpus.update((doc["id"], doc) for doc in docs)
    save(Path("data/guard-official-dev"), dataset, queries, labels, corpus)


if __name__ == "__main__":
    action = {"development": development, "heldout": heldout}[sys.argv[1]]
    for name in ("hotpot", "musique"):
        action(name)
