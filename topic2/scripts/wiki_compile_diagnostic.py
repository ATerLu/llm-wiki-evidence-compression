"""Check compiler coverage of eight known development evidence slots."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_experiment import data_for
from cloud_reader import CloudReader, MODEL
from paired import inputs
from prepare import jsonl
from wiki import normalize
from wiki_compile_pilot import SCOPE, compile_doc, save_index
from wiki_bridge import span_text

# These are hand-located development targets, for diagnosis rather than a prevalence estimate.
CASES = (
    ("hotpot", "5a7717fe5542993569682cb4", 0, 0, "death_date", "1994"),
    ("hotpot", "5a7f0dce5542993067513605", 5, 2, "alias", "Cherokee River"),
    ("hotpot", "5ae353875542992f92d8227b", 1, 1, "founded_date", "1890"),
    ("hotpot", "5a80b5ba554299260e20a138", 3, 0, "death_date", "1950"),
    ("hotpot", "5a8764ba5542996e4f3087c8", 4, 0, "released_by", "Flying Dog Brewery"),
    ("musique", "2hop__47608_71972", 0, 0, "educated_at", "Alvin High School"),
    ("musique", "2hop__431059_720696", 4, 0, "located_in", "Mississippi County"),
    ("musique", "2hop__147208_565529", 19, 0, "located_in", "Odessa Oblast"),
)


def run(path: Path) -> None:
    path.mkdir(parents=True)
    stores = {d: inputs(d, data_for(SCOPE)) for d in ("hotpot", "musique")}
    targets, docs = [], []
    for dataset, qid, candidate, sentence_id, predicate, object_text in CASES:
        queries, corpus = stores[dataset]
        q = queries[qid]
        doc = corpus[q["candidates"][candidate]["id"]]
        assert object_text in span_text(doc, sentence_id)
        targets.append({"dataset": dataset, "query_id": qid, "question": q["question"], "title": doc["title"],
                        "doc_id": doc["id"], "source_version": doc["version"], "sentence_id": sentence_id,
                        "predicate": predicate, "object_contains": object_text,
                        "text": span_text(doc, sentence_id)})
        docs.append((dataset, "targeted_development", doc))
    (path / "config.json").write_text(json.dumps({"model": MODEL, "scope": SCOPE, "targets": targets},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    reader = CloudReader()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda entry: compile_doc(reader, entry), docs))
    jsonl(path / "compilation.jsonl", rows)
    save_index(rows, path / "wiki-diagnostic.sqlite3")
    scored = []
    for target, row in zip(targets, rows, strict=True):
        hits = {}
        for method in ("rules", "llm"):
            hits[method] = [r for r in row[method] if r["predicate"] == target["predicate"]
                            and r["sentence_id"] == target["sentence_id"]
                            and f" {normalize(target['object_contains'])} " in f" {normalize(r['object'])} "]
        scored.append({**target, "parse": row["parse"], **hits})
        print(target["title"], "rule", bool(hits["rules"]), "LLM", bool(hits["llm"]), row["parse"])
    jsonl(path / "diagnostic.jsonl", scored)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
