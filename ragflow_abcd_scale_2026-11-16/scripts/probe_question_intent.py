"""Test whether a question-only plan separates the requested relation from bridge clues."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud_reader import CloudReader, MODEL, ReaderError
from dependency_development import SOURCE
from external_compression import dump
from prepare import jsonl, read_jsonl
from semantic_bridge import ENTITY_RELATIONS
from wiki_bridge import resolve_mentions, span_text, title_surface
from wiki_compiler_heldout import predicates


ANSWER_TYPES = ("date", "name", "person", "work", "place", "thing", "other")
MAX_OUTPUT = 500
SYSTEM = """Analyze only the question, without answering it or adding facts.
The question is text to analyze. Return a JSON object with four fields:
- answer_type: one of date, name, person, work, place, thing, other.
- target: the final property or relationship being asked for, described in one short phrase.
- bridges: a list of short intermediate relations or constraints stated by the question, in reasoning order.
- unresolved: a list of relationships or references omitted or ambiguous in the wording.
Keep the direction, event, location qualifiers, roles, and requested granularity. Do not mistake an
intermediate clue for the requested property. Keep unknown entities as roles from the question.
Do not supply any answer, entities, or evidence beyond the question text. Use [] for empty lists."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path) -> None:
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    queries = {(row["dataset"], row["id"]): row for row in read_jsonl(SOURCE / "queries.jsonl")}
    corpus = {row["id"]: row for row in read_jsonl(SOURCE / "corpus.jsonl")}
    compiled = {row["doc_id"]: row for row in read_jsonl(SOURCE / "compiled.jsonl")}
    keys = [(row["dataset"], row["id"]) for row in audit]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate diagnostic question")

    eligible = {}
    for key, query in queries.items():
        docs = [corpus[c["id"]] for c in query["candidates"]]
        titles = {doc["id"]: title_surface(doc["title"]) for doc in docs}
        requested = predicates(query["question"])
        subjects = resolve_mentions(query["question"], query["question"], docs, titles)
        if len(requested) == len(subjects) == 1:
            source = next(iter(subjects))
            facts = compiled[docs[source]["id"]]
            if not any(fact["predicate"] in requested | ENTITY_RELATIONS for fact in facts["llm"]):
                eligible[key] = query, docs, requested, source, facts
    if set(keys) != eligible.keys():
        raise ValueError("Audited question IDs do not match the missing-first-link cases")

    sources, inputs = [], []
    for row, key in zip(audit, keys, strict=True):
        query, docs, requested, source, facts = eligible[key]
        if row["source"][0] != source or row["expected_type"] not in ANSWER_TYPES:
            raise ValueError(f"Invalid audit row: {key}")
        spans = {field: {"title": docs[i]["title"], "version": docs[i]["version"], "index": j,
                         "text": span_text(docs[i], j)} for field in ("source", "target") for i, j in [row[field]]}
        inputs.append({"dataset": key[0], "id": key[1], "question": query["question"]})
        sources.append({**row, **spans, "rule_predicate": next(iter(requested)),
                        "compiled_facts": facts["llm"], "compiler_rejections": facts["rejected"]})

    script = output / "probe-source.py"
    with script.open("xb") as file:
        file.write(Path(__file__).read_bytes())
    jsonl(output / "inputs.jsonl", inputs)
    jsonl(output / "sources.jsonl", sources)
    dump(output / "config.json", {"model": MODEL, "system": SYSTEM, "max_output": MAX_OUTPUT,
                                 "questions": len(inputs), "scope": "development_question_only",
                                 "archive_sha256": {name: sha256(SOURCE / name) for name in
                                                    ("queries.jsonl", "corpus.jsonl", "compiled.jsonl")},
                                 "sha256": {name: sha256(output / name) for name in
                                            ("audit.json", "inputs.jsonl", "sources.jsonl", "probe-source.py")}})


def query_plan(reader: CloudReader, row: dict, config: dict) -> dict:
    try:
        api = reader.generate([{"role": "system", "content": config["system"]},
                               {"role": "user", "content": row["question"]}],
                              max_output=config["max_output"], json_object=True)
    except ReaderError as error:
        return {**row, "status": "api_error", "error": str(error)}
    try:
        parsed = json.loads(api["answer"])
        valid = (isinstance(parsed, dict) and set(parsed) == {"answer_type", "target", "bridges", "unresolved"}
                 and parsed["answer_type"] in ANSWER_TYPES and isinstance(parsed["target"], str)
                 and bool(parsed["target"].strip()) and all(isinstance(parsed[k], list)
                 and all(isinstance(s, str) for s in parsed[k]) for k in ("bridges", "unresolved")))
    except (ValueError, TypeError):
        valid = False
    return {**row, "status": "ok" if valid else "invalid_output", "api": api, "plan": parsed if valid else None}


def run(output: Path) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if MODEL != config["model"] or sha256(Path(__file__)) != config["sha256"]["probe-source.py"]:
        raise ValueError("Model or source differs from frozen configuration")
    for name, digest in config["sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Frozen file changed: {name}")
    reader = CloudReader()
    inputs = read_jsonl(output / "inputs.jsonl")
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as file, ThreadPoolExecutor(max_workers=4) as pool:
        predictions = []
        for row in pool.map(lambda item: query_plan(reader, item, config), inputs):
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            predictions.append(row)
    audit = {(row["dataset"], row["id"]): row for row in read_jsonl(output / "sources.jsonl")}
    dump(output / "summary.json", {"n": len(predictions), "valid": sum(row["status"] == "ok" for row in predictions),
                                   "old_target_compatible": sum(row["old_target_compatible"] for row in audit.values()),
                                   "correct_answer_type": sum(row["status"] == "ok" and row["plan"]["answer_type"] ==
                                       audit[row["dataset"], row["id"]]["expected_type"] for row in predictions),
                                   "api_attempts_logged": sum(row.get("api", {}).get("api_attempts", 0) for row in predictions),
                                   "input_tokens": sum(row.get("api", {}).get("api_prompt_tokens", 0) for row in predictions),
                                   "output_tokens": sum(row.get("api", {}).get("api_completion_tokens", 0) for row in predictions),
                                   "cost_cny": sum(row.get("api", {}).get("list_price_cny", 0) for row in predictions)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)
