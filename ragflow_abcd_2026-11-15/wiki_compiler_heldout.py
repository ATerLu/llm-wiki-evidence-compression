"""Compare two validated compilers through the same B packer on fresh official-dev families."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean
import sys
from urllib.error import URLError

from pyarrow.parquet import ParquetFile

from baseline import SYSTEM
from cloud_experiment import BudgetTokenizer, data_for
from cloud_reader import CloudReader, MODEL, ReaderError
from paired import inputs, score_row
from prepare import adapt, jsonl, question_groups, read_jsonl
from scripts.check_paired import family_interval, official_scores
from wiki import normalize
from wiki_bridge import pack_ranked_bridges, title_surface
from wiki_compile_pilot import LOOKUP, compile_doc, save_index
from wiki_compile_v2 import filter_v2, rules_v2

SEED = 20261115
BUDGET = 1024
N = 10
DATASETS = ("hotpot", "musique")


def predicates(question: str) -> set[str]:
    matched = set().union(*(p for expression, p in LOOKUP if re.search(expression, question, re.I)))
    if "where" in question.casefold() and not re.search(r"\b(?:date|when|year)\b", question, re.I):
        matched.discard("birth_date")
        matched.discard("death_date")
    return matched


def raw_rows(dataset: str) -> list[dict]:
    raw = Path("data/raw")
    if dataset == "musique":
        return read_jsonl(raw / "musique_ans_v1.0_dev.jsonl")
    return [r for batch in ParquetFile(raw / "hotpot-validation-00000-of-00001.parquet").iter_batches()
            for r in batch.to_pylist()]


def freeze(tokenizer: BudgetTokenizer, output: Path) -> tuple[list[dict], dict]:
    queries, documents = [], {}
    for dataset in DATASETS:
        rows = raw_rows(dataset)
        used = {q["family_id"] for scope in ("official_dev", "guard_heldout")
                for q in read_jsonl(data_for(scope) / dataset / "queries.jsonl")}
        mapping = {qid: family for family, ids in question_groups(rows, dataset).items() for qid in ids}
        random.Random(SEED).shuffle(rows)
        labels = []
        for row in rows:
            family = mapping[row["id"]]
            if family in used or not predicates(row["question"]):
                continue
            query, label, docs = adapt(row, dataset, family)
            titles = {d["id"]: title_surface(d["title"]) for d in docs}
            _, _, _, edges = pack_ranked_bridges(tokenizer, query["question"], docs, titles, BUDGET)
            if not any(e["decision"] == "added" for e in edges):
                continue
            used.add(family)
            queries.append({"dataset": dataset, **query})
            labels.append(label)
            for doc in docs:
                documents.setdefault(doc["id"], (dataset, doc))
            if len(labels) == N:
                break
        assert len(labels) == N
        jsonl(output / f"labels-{dataset}.jsonl", labels)
    jsonl(output / "queries.jsonl", queries)
    jsonl(output / "corpus.jsonl", (doc for _, doc in documents.values()))
    return queries, documents


def compile_v2(reader: CloudReader, entry: tuple) -> dict:
    row = compile_doc(reader, entry)
    doc = entry[-1]
    row["rules"], row["rule_rejected"] = rules_v2(doc)
    if row["parse"] == "ok":
        raw = row["api"]["answer"].strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        row["llm"], row["llm_rejected"] = filter_v2(doc, json.loads(raw))
    return row


def hints_for(question: str, docs: list[dict], index: dict, method: str) -> dict:
    types = predicates(question)
    hints = {}
    for d in docs:
        for fact in index.get(d["id"], {}).get(method, []):
            if fact["predicate"] in types:
                anchor = fact["anchor_id"]
                hints.setdefault((d["id"], fact["sentence_id"]), () if anchor is None else (anchor,))
    return hints


def ask(reader: CloudReader, question: str, evidence: str) -> dict:
    try:
        return reader.generate([{"role": "system", "content": SYSTEM}, {
            "role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {question}",
        }])
    except (ReaderError, URLError) as error:
        return {"status": "error", "answer": "", "error": str(error)}


def run(output: Path) -> None:
    output.mkdir(parents=True)
    reader, tokenizer = CloudReader(), BudgetTokenizer()
    queries, entries = freeze(tokenizer, output)
    config = {"model": MODEL, "seed": SEED, "budget": BUDGET, "questions": len(queries), "pages": len(entries),
              "selection": "fresh official dev families, supported facet and an uncompiled B bridge, no label filtering"}
    source = output / "source"
    source.mkdir()
    config["code_sha256"] = {}
    for name in ("wiki_compiler_heldout.py", "wiki_compile_v2.py", "wiki_compile_pilot.py", "wiki_bridge.py",
                 "baseline.py", "cloud_reader.py", "prepare.py"):
        blob = Path(name).read_bytes()
        (source / name).write_bytes(blob)
        config["code_sha256"][name] = hashlib.sha256(blob).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Frozen", len(queries), "questions;", len(entries), "pages", flush=True)
    compiled = []
    with ThreadPoolExecutor(max_workers=8) as pool, (output / "compilation.jsonl").open("x", encoding="utf-8") as file:
        for row in pool.map(lambda item: compile_v2(reader, (item[0], "heldout", item[1])), entries.values()):
            compiled.append(row)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            if len(compiled) % 30 == 0 or len(compiled) == len(entries):
                print(f"Compiled {len(compiled)}/{len(entries)} pages", flush=True)
    save_index(compiled, output / "wiki.sqlite3")
    index = {r["doc_id"]: r for r in compiled if r["parse"] == "ok"}
    corpus = {key: d for key, (_, d) in entries.items()}
    predictions = []
    for q in queries:
        docs = [corpus[c["id"]] for c in q["candidates"]]
        titles = {d["id"]: title_surface(d["title"]) for d in docs}
        cache = {}
        for method in ("rules", "llm"):
            hints = hints_for(q["question"], docs, index, method)
            selected, text, tokens, edges = pack_ranked_bridges(
                tokenizer, q["question"], docs, titles, BUDGET, fact_hints=hints)
            reused = text in cache
            if not reused:
                cache[text] = ask(reader, q["question"], text)
            predictions.append({
                "dataset": q["dataset"], "query_id": q["id"], "family_id": q["family_id"], "method": method,
                "candidate_ids": [d["id"] for d in docs], "budget": BUDGET, "evidence_tokens": tokens,
                "evidence": text, "selected_blocks": [[i, sorted(s)] for i, s in sorted(selected.items())],
                "wiki_edges": edges, "hint_count": len(hints), "shared_response": reused, **cache[text],
            })
    jsonl(output / "predictions.jsonl", predictions)
    evaluate(output, queries, corpus, compiled, predictions)


def evaluate(output: Path, queries: list[dict], corpus: dict, compiled: list[dict], predictions: list[dict]) -> None:
    queries_by_id = {(q["dataset"], q["id"]): q for q in queries}
    summary, rows = {}, []
    for dataset in DATASETS:
        labels = {r["id"]: r for r in read_jsonl(output / f"labels-{dataset}.jsonl")}
        subset = [score_row(r, queries_by_id[dataset, r["query_id"]], corpus, labels[r["query_id"]])
                  for r in predictions if r["dataset"] == dataset]
        rows.extend(subset)
        r, l = [{x["query_id"]: x for x in subset if x["method"] == m} for m in ("rules", "llm")]
        keys = [k for k in r if r[k]["status"] == l[k]["status"] == "ok"]
        assert all(r[k]["api_response_id"] == l[k]["api_response_id"] for k in keys if r[k]["evidence"] == l[k]["evidence"])
        for condition in (r, l):
            data = [condition[k] for k in keys]
            official = official_scores(dataset, data, labels)
            assert all(abs(v - (round(mean(x[field] for x in data), 3) if dataset == "musique"
                                else mean(x[field] for x in data))) < 1e-9
                       for v, field in zip(official, ("em", "f1"), strict=True))
        delta = [l[k]["f1"] - r[k]["f1"] for k in keys]
        summary[dataset] = {
            "paired_n": len(keys), "changed_context": sum(r[k]["evidence"] != l[k]["evidence"] for k in keys),
            "rules": {field: mean(r[k][field] for k in keys) for field in ("em", "f1", "strict_recall", "touched_recall")},
            "llm": {field: mean(l[k][field] for k in keys) for field in ("em", "f1", "strict_recall", "touched_recall")},
            "f1_improved": sum(x > 1e-9 for x in delta), "f1_worsened": sum(x < -1e-9 for x in delta),
            "interval": family_interval({k: r[k] for k in keys}, {k: l[k] for k in keys}),
        }
    summary["compilation"] = {
        "pages": len(compiled), "parse": dict(Counter(r["parse"] for r in compiled)),
        "facts": {m: sum(len(r[m]) for r in compiled if r["parse"] == "ok") for m in ("rules", "llm")},
        "rejected": dict(Counter(f["reason"] for r in compiled for f in r["llm_rejected"])),
        "list_price_cny": sum(r["api"].get("list_price_cny", 0) for r in compiled),
    }
    summary["qa"] = {"calls": sum(not r["shared_response"] for r in rows),
                     "list_price_cny": sum(r.get("list_price_cny", 0) for r in rows if not r["shared_response"])}
    jsonl(output / "scored.jsonl", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
