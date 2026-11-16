"""Freeze train splits and export candidate inputs separately from evaluation data."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile

SEED = 20261109
SIZES = {"smoke": 100, "development": 500, "validation": 500}


def jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_train(raw: Path, dataset: str):
    if dataset == "hotpot":
        from pyarrow.parquet import ParquetFile

        for path in sorted(raw.glob("hotpot-train-*.parquet")):
            for batch in ParquetFile(path).iter_batches(batch_size=1024):
                yield from batch.to_pylist()
    elif (raw / "musique-v1.0.zip").exists():
        with ZipFile(raw / "musique-v1.0.zip") as archive:
            name, = [n for n in archive.namelist() if Path(n).name == "musique_ans_v1.0_train.jsonl"]
            with archive.open(name) as file:
                yield from (json.loads(line) for line in file if line.strip())
    else:
        yield from read_jsonl(raw / "musique_ans_v1.0_train.jsonl")


def question_keys(row: dict, dataset: str) -> list[tuple[str, str]]:
    normalized = " ".join(re.findall(r"\w+", row["question"].casefold()))
    keys = [("question", normalized)]
    if dataset == "musique":
        keys.extend(("component", str(part["id"])) for part in row["question_decomposition"])
    return keys


def question_groups(rows: list[dict], dataset: str) -> dict:
    # Transitive sharing of a source question keeps the whole component in one split.
    parent = list(range(len(rows)))
    owners = {}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, row in enumerate(rows):
        for key in question_keys(row, dataset):
            previous = owners.setdefault(key, index)
            parent[root(index)] = root(previous)

    id_field = "id"
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row[id_field])
    if len({row[id_field] for row in rows}) != len(rows):
        raise ValueError(f"{dataset}: duplicate query ID")
    return {min(group): group for group in groups.values()}


def freeze_splits(rows: list[dict], dataset: str) -> tuple[dict, dict]:
    groups = question_groups(rows, dataset)
    families = {qid: family for family, group in groups.items() for qid in group}
    available = sorted(groups.values(), key=min)
    random.Random(SEED).shuffle(available)
    splits = {}
    for name, size in SIZES.items():
        selected, unused = [], []
        for group in available:
            if len(group) <= size - len(selected):
                selected.extend(sorted(group))
            else:
                unused.append(group)
        if len(selected) != size:
            raise ValueError(f"{dataset}: unable to freeze {size} queries for {name}")
        splits[name] = selected
        available = unused
    return splits, families


def document(title: str, texts: list[str]) -> dict:
    text = "".join(texts)
    canonical = json.dumps([title, texts], ensure_ascii=False).encode("utf-8")
    version = hashlib.sha256(canonical).hexdigest()
    offset = 0
    spans = []
    for index, sentence in enumerate(texts):
        end = offset + len(sentence)
        spans.append({"index": index, "start": offset, "end": end})
        offset = end
    return {"id": version, "version": version, "title": title, "text": text, "spans": spans}


def adapt(row: dict, dataset: str, family: str) -> tuple[dict, dict, list[dict]]:
    if dataset == "hotpot":
        context = row["context"]
        documents = [
            document(title, sentences)
            for title, sentences in zip(context["title"], context["sentences"], strict=True)
        ]
        support = row["supporting_facts"]
        references = list(zip(support["title"], support["sent_id"], strict=True))
        for title, index in references:
            matching = {doc["id"] for doc in documents if doc["title"] == title}
            if len(matching) != 1:
                raise ValueError(f"{row['id']}: ambiguous or missing support title {title}")
            doc = next(doc for doc in documents if doc["id"] in matching)
            if not 0 <= index < len(doc["spans"]):
                raise ValueError(f"{row['id']}: invalid support sentence index {index}")
        keys = [doc["title"] for doc in documents]
        labels = {"answer": row["answer"], "support": references, "type": row["type"]}
    else:
        paragraphs = row["paragraphs"]
        documents = [
            document(p["title"], re.split(r"(?<=[.!?])(?=\s+\S)", p["paragraph_text"]))
            for p in paragraphs
        ]
        keys = [p["idx"] for p in paragraphs]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{row['id']}: duplicate paragraph index")
        labels = {
            "answer": row["answer"], "aliases": row["answer_aliases"],
            "support": [p["idx"] for p in paragraphs if p["is_supporting"]],
            "hops": len(row["question_decomposition"]), "answerable": row["answerable"],
        }
    query = {
        "id": row["id"], "family_id": family, "question": row["question"],
        "candidates": [{"id": doc["id"], "key": key}
                       for doc, key in zip(documents, keys, strict=True)],
    }
    return query, {"id": row["id"], **labels}, documents


def prepare(raw: Path, output: Path, dataset: str) -> None:
    # Keep only grouping keys and IDs on the first pass over the full train split.
    if dataset == "hotpot" and len(list(raw.glob("hotpot-train-*.parquet"))) != 2:
        raise ValueError("Expected both pinned HotpotQA distractor train shards")
    grouping_rows = []
    for row in load_train(raw, dataset):
        keys = {"id": row["id"], "question": row["question"]}
        if dataset == "musique":
            keys["question_decomposition"] = [{"id": part["id"]} for part in row["question_decomposition"]]
        grouping_rows.append(keys)
    splits, families = freeze_splits(grouping_rows, dataset)
    selected = {qid for split in splits.values() for qid in split}
    corpus, queries, labels = {}, [], []
    for row in load_train(raw, dataset):
        if row["id"] in selected:
            query, label, documents = adapt(row, dataset, families[row["id"]])
            queries.append(query)
            labels.append(label)
            corpus.update((doc["id"], doc) for doc in documents)
    if len(queries) != len(selected):
        raise ValueError("Train data changed during preparation")
    output.mkdir(parents=True)
    jsonl(output / "corpus.jsonl", corpus.values())
    jsonl(output / "queries.jsonl", queries)
    jsonl(output / "labels.jsonl", labels)
    (output / "splits.json").write_text(json.dumps({
        "dataset": dataset, "source_split": "train", "source_rows": len(grouping_rows),
        "seed": SEED, "family_rule": "normalized question; MuSiQue also links shared component IDs",
        "splits": splits,
    }, indent=2), encoding="utf-8")
    print(f"{dataset}: {len(grouping_rows)} source rows, {len(queries)} queries, {len(corpus)} documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["hotpot", "musique"])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.dataset)
