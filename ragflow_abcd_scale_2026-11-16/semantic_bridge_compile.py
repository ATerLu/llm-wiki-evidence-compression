"""Freeze unrun question families and compile source-grounded Wiki facts offline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

from cloud_experiment import DATASETS, data_for
from cloud_reader import CloudReader, ENDPOINT, MODEL, ReaderError
from external_compression import append, dump
from prepare import adapt, jsonl, question_groups, read_jsonl
from wiki_bridge import span_text
from wiki_compile_pilot import PROMPT, save_index
from wiki_compile_v2 import filter_v2
from wiki_compiler_heldout import raw_rows

SEED = 20261116
N = 50
BATCH = 5
MAX_OUTPUT = 3000
SYSTEM = PROMPT.replace("Output a JSON array only, each item", "For each page, produce facts with fields").replace(
    "If no facts qualify, output [].", "If no facts qualify, its facts list is empty.") + (
    '\nTreat each page independently. Return only a JSON object {"pages":[{"page_id":0,"facts":[]}]}, '
    "with one entry for each input page_id. Sentence and anchor IDs are local to that page."
)
PRIOR_COMPILER = Path("runs/wiki-compiler-v2-heldout20/queries.jsonl")


def prior_ids(dataset: str) -> set[str]:
    ids = {q["id"] for scope in ("official_dev", "guard_heldout", "guard_development")
           for q in read_jsonl(data_for(scope) / dataset / "queries.jsonl")}
    ids.update(q["id"] for q in read_jsonl(PRIOR_COMPILER) if q["dataset"] == dataset)
    for path in Path("runs").glob("*/scored.jsonl"):
        ids.update(r["query_id"] for r in read_jsonl(path)
                   if r.get("dataset") == dataset and "query_id" in r)
    return ids


def freeze(output: Path) -> tuple[list[dict], dict, dict]:
    queries, corpus, exclusions = [], {}, {}
    for dataset in DATASETS:
        rows = raw_rows(dataset)
        groups = question_groups(rows, dataset)
        old = prior_ids(dataset)
        blocked = {f for f, ids in groups.items() if old.intersection(ids)}
        choices = sorted(groups.keys() - blocked)
        rng = random.Random(SEED)
        picked = [(f, rng.choice(sorted(groups[f]))) for f in rng.sample(choices, N)]
        raw = {row["id"]: row for row in rows}
        labels = []
        for family, qid in picked:
            query, label, docs = adapt(raw[qid], dataset, family)
            queries.append({"dataset": dataset, **query})
            labels.append(label)
            corpus.update({doc["id"]: doc for doc in docs})
        jsonl(output / f"labels-{dataset}.jsonl", labels)
        exclusions[dataset] = {"earlier_ids": sorted(old), "excluded_official_families": sorted(blocked),
                               "eligible_families": len(choices)}
    jsonl(output / "queries.jsonl", queries)
    jsonl(output / "corpus.jsonl", corpus.values())
    return queries, corpus, exclusions


def compile_batch(reader: CloudReader, docs: list[dict], *, author_extension: bool = False) -> tuple[dict, list[dict]]:
    pages = [{"page_id": i, "title": doc["title"],
              "sentences": [[s["index"], span_text(doc, s["index"])] for s in doc["spans"]]}
             for i, doc in enumerate(docs)]
    try:
        api = reader.generate([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": json.dumps(pages, ensure_ascii=False)}],
                              max_output=MAX_OUTPUT, json_object=True)
    except ReaderError as error:
        api = {"status": "error", "error": str(error)}
    return api, parse_batch(api, docs, author_extension=author_extension)


def parse_batch(api: dict, docs: list[dict], *, author_extension: bool) -> list[dict]:
    returned = {}
    try:
        payload = json.loads(api.get("answer", ""))["pages"]
        for page in payload:
            if type(page["page_id"]) is int and isinstance(page["facts"], list):
                pid = page["page_id"]
                returned[pid] = None if pid in returned else page["facts"]
    except (TypeError, KeyError, ValueError):
        returned = {}
    rows = []
    for i, doc in enumerate(docs):
        items = returned.get(i)
        facts, rejected = filter_v2(doc, items, author_extension=author_extension) if items is not None else ([], [])
        rows.append({"doc_id": doc["id"], "source_version": doc["version"], "title": doc["title"],
                     "parse": "ok" if items is not None else "missing_or_invalid_page", "rules": [],
                     "llm": facts, "rejected": rejected})
    return rows


def prepare(output: Path, workers: int, *, author_extension: bool = False) -> None:
    output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    (output / "compile-source.py").write_bytes(source)
    if author_extension:
        from author_bridge import N as per_dataset, freeze_authors

        queries, corpus, exclusions = freeze_authors(output)
    else:
        queries, corpus, exclusions = freeze(output)
        per_dataset = N
    ordered = sorted(corpus.values(), key=lambda d: d["id"])
    batches = [ordered[i:i + BATCH] for i in range(0, len(ordered), BATCH)]
    files = ["queries.jsonl", "corpus.jsonl", *(f"labels-{ds}.jsonl" for ds in DATASETS)]
    config = {"seed": SEED, "per_dataset": per_dataset, "pages": len(ordered), "queries": len(queries), "batch": BATCH,
              "author_extension": author_extension,
              "workers": workers, "compile_model": MODEL, "endpoint": ENDPOINT, "system": SYSTEM,
              "max_output": MAX_OUTPUT, "compile_source_sha256": hashlib.sha256(source).hexdigest(),
              "input_sha256": {f: hashlib.sha256((output / f).read_bytes()).hexdigest() for f in files},
              "exclusions": exclusions}
    dump(output / "freeze.json", config)
    print(f"Frozen {len(queries)} questions, {len(ordered)} pages, {len(batches)} compilation requests", flush=True)
    reader = CloudReader()
    compiled = []
    with ThreadPoolExecutor(max_workers=workers) as pool, (output / "compile-api.jsonl").open("x", encoding="utf-8") as file:
        for i, (api, rows) in enumerate(pool.map(lambda docs: compile_batch(reader, docs, author_extension=author_extension), batches), 1):
            compiled.extend(rows)
            append(file, {"pages": [r["doc_id"] for r in rows], **api})
            if i % 25 == 0 or i == len(batches):
                print(f"Compilation {i}/{len(batches)}", flush=True)
    jsonl(output / "compiled.jsonl", compiled)
    save_index(compiled, output / "wiki.sqlite3")
    print(f"Compiled {sum(r['parse'] == 'ok' for r in compiled)}/{len(compiled)} pages", flush=True)


def revalidate(output: Path) -> None:
    corpus = {d['id']: d for d in read_jsonl(output / 'corpus.jsonl')}
    compiled = [row for api in read_jsonl(output / 'compile-api.jsonl')
                for row in parse_batch(api, [corpus[doc] for doc in api['pages']], author_extension=True)]
    jsonl(output / 'compiled-reviewed.jsonl', compiled)
    save_index(compiled, output / 'wiki-reviewed.sqlite3')
    print(f"Reviewed {len(compiled)} saved responses by page", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--author-extension", action="store_true")
    parser.add_argument("--revalidate", action="store_true")
    args = parser.parse_args()
    if args.revalidate:
        revalidate(args.output)
    else:
        prepare(args.output, args.workers, author_extension=args.author_extension)
