"""Development comparison of indexed, title-linked and human-verified evidence windows."""

import argparse
import json
from pathlib import Path

from baseline import words
from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader, MODEL
from dependency_development import SOURCE, describe
from external_compression import append, dump
from intent_evidence_experiment import prepare as prepare_selectors, selector_input, sha256
from prepare import jsonl, read_jsonl
from scripts.probe_question_intent import MAX_OUTPUT, SYSTEM, query_plan
from semantic_bridge import TARGET_WINDOW, rule_links
from semantic_bridge_experiment import verify_freeze
from wiki_bridge import span_text, title_surface
from wiki_window_experiment import (CHAIN_AUDIT, INTENT_AUDIT, PREVIOUS, WIKI_QUOTA, WINDOW_CAP,
                                    candidate_window, wiki_groups)


METHODS = ("rank_narrow", "wiki_narrow", "entity_narrow", "oracle_narrow")
PRIOR_WINDOW = Path("runs/wiki-window-development-20261111")
SOURCE_CASES = {"5ae5a63c5542992663a4f203", "5a7a567255429941d65f25bd", "5a7ef18c55429934daa2fc84",
                "5a821b4b5542995ce29dccab"}


def cases() -> dict:
    result = {qid: {"annotation": "clear_chain", "spans": sorted({tuple(span) for step in row["steps"]
                                                                   for span in step["alternatives"][0]})}
              for qid, row in json.loads(CHAIN_AUDIT.read_text(encoding="utf-8")).items()
              if row["tier"] == "clear" and row["run"] == SOURCE.name}
    for row in json.loads(INTENT_AUDIT.read_text(encoding="utf-8")):
        if row["id"] in SOURCE_CASES:
            result[row["id"]] = {"annotation": "two_source_spans", "spans": sorted([row["source"], row["target"]])}
    return result


def entity_groups(item: dict, docs: list[dict], order: list[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    titles = {doc["id"]: title_surface(doc["title"]) for doc in docs}
    seeds = order[:TARGET_WINDOW]
    links = rule_links(item["question"], docs, titles, seeds)
    target_terms = words(item["plan"]["target"])
    groups = []
    for seed in seeds:
        for target in sorted(links.get(seed, {})):
            choices = [span for span in order if span[0] == target][:TARGET_WINDOW]
            if choices:
                chosen = max(choices, key=lambda s: len(words(span_text(docs[s[0]], s[1])) & target_terms))
                groups.append({seed, chosen})
    return groups


def prepare(output: Path) -> None:
    prepare_selectors(output)
    _, corpus, queries = verify_freeze(SOURCE)
    annotated = cases()
    definition = output / "cases.json"
    dump(definition, annotated)
    paths = (Path(__file__), Path("wiki_window_experiment.py"), Path("scripts/probe_question_intent.py"))
    for path in paths:
        (output / "source" / path.name).write_bytes(path.read_bytes())
    inputs_path = output / "selector_inputs.jsonl"
    archived = {row["id"]: row for row in read_jsonl(inputs_path)}
    ranks = {(r["dataset"], r["query_id"]): r for r in read_jsonl(SOURCE / "rankings.jsonl")}
    compiled = {row["doc_id"]: row for row in read_jsonl(SOURCE / "compiled.jsonl")}
    tokenizer, reader = BudgetTokenizer(), CloudReader()
    inputs, plan_calls, logs = [], [], []
    plan_config = {"model": MODEL, "system": SYSTEM, "max_output": MAX_OUTPUT}
    dump(output / "plan-config.json", {**plan_config, "cases_sha256": sha256(definition)})
    for query in queries:
        qid = query["id"]
        if qid not in annotated:
            continue
        docs = [corpus[c["id"]] for c in query["candidates"]]
        rank = ranks[query["dataset"], qid]
        order = [tuple(rank["candidate_spans"][i]) for i in rank["ranking"]]
        item = archived.get(qid)
        if item is None:
            result = query_plan(reader, {key: query[key] for key in ("dataset", "id", "question")}, plan_config)
            plan_calls.append(result)
            if result["status"] != "ok":
                jsonl(output / "new-plans.jsonl", plan_calls)
                raise ValueError(f"Question-only parser failed for {qid}")
            item = selector_input(query, docs, compiled, result["plan"])
        required = frozenset(map(tuple, annotated[qid]["spans"]))
        groups = ([], wiki_groups(item, docs, order, compiled), entity_groups(item, docs, order), [])
        item["candidate_sets"] = {}
        for method, promotion in zip(METHODS, groups, strict=True):
            pins = required if method == "oracle_narrow" else frozenset()
            window, tokens, promoted = candidate_window(tokenizer, item["candidates"], order, promotion, pinned=pins)
            item["candidate_sets"][method] = window
            spans = {(p["page"], s[0]) for p in window for s in p["sentences"]}
            logs.append({"query_id": qid, "method": method, "candidate_json_tokens": tokens,
                         "proof_visible": required <= spans, "missing_proof": sorted(required - spans),
                         "promoted": describe(docs, promoted), "spans": sorted(spans)})
        inputs.append(item)
    if {row["id"] for row in inputs} != annotated.keys():
        raise ValueError("The archived query set is missing an annotated development case")
    with inputs_path.open("w", encoding="utf-8") as file:
        for item in inputs:
            append(file, item)
    jsonl(output / "new-plans.jsonl", plan_calls)
    jsonl(output / "windows.jsonl", logs)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reuse = [PREVIOUS / "scored.jsonl", PRIOR_WINDOW / "scored.jsonl"]
    frozen = [CHAIN_AUDIT, INTENT_AUDIT, *reuse, inputs_path, definition, output / "windows.jsonl",
              output / "plan-config.json", output / "new-plans.jsonl", *(output / "source" / p.name for p in paths)]
    config["sha256"].update({p.as_posix(): sha256(p) for p in frozen})
    config["live_code_sha256"].update({p.as_posix(): sha256(p) for p in paths})
    config.update(questions=len(inputs), methods=METHODS, upstream_candidate_json_cap=WINDOW_CAP, wiki_quota=WIKI_QUOTA,
                  reuse_scored=[p.as_posix() for p in reuse], scope="development_oracle")
    dump(config_path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
