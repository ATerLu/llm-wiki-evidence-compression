"""Compare same-budget candidate windows with and without archived Wiki evidence pointers."""

import argparse
import json
from pathlib import Path

from cloud_experiment import BudgetTokenizer
from dependency_development import SOURCE, describe
from external_compression import append, dump
from intent_evidence_experiment import prepare as prepare_selectors, sha256
from prepare import jsonl, read_jsonl
from semantic_bridge import TARGET_WINDOW, compiled_links, rule_links
from semantic_bridge_experiment import verify_freeze
from wiki_bridge import SEED_ANCHORS, title_surface


WINDOW_CAP = 768
WIKI_QUOTA = 256
WINDOW_METHODS = ("semantic_window", "wiki_window")
PREVIOUS = Path("runs/intent-evidence-development-20261111")
CHAIN_AUDIT = Path("runs/author-dependency-replay/audit.json")
INTENT_AUDIT = Path("runs/question-intent-diagnostic-20261111/audit.json")


def wiki_groups(item: dict, docs: list[dict], order: list[tuple[int, int]],
                compiled: dict) -> list[set[tuple[int, int]]]:
    ranks = {span: i for i, span in enumerate(order)}
    titles = {doc["id"]: title_surface(doc["title"]) for doc in docs}
    seeds = order[:TARGET_WINDOW]
    facts = [compiled[doc["id"]]["llm"] for doc in docs]
    rules = rule_links(item["question"], docs, titles, seeds)
    links = compiled_links(item["question"], docs, titles, seeds, rules, facts)
    groups = []
    for seed in seeds:
        for page, bridge in links.get(seed, {}).items():
            target = next((span for span in order if span[0] == page), None)
            if bridge.relations and target is not None:
                groups.append({seed, target, *bridge.proof})
    ordered_facts = sorted(item["wiki_index"], key=lambda fact: ranks.get((fact["page"], fact["sentence_ids"][-1]), len(order)))
    for fact in ordered_facts:
        if (fact["page"], fact["sentence_ids"][-1]) in order[:SEED_ANCHORS]:
            groups.append({(fact["page"], i) for i in fact["sentence_ids"]})
    return groups


def candidate_window(tokenizer: BudgetTokenizer, candidates: list[dict], order: list[tuple[int, int]],
                     groups: list[set[tuple[int, int]]], *,
                     pinned: frozenset[tuple[int, int]] = frozenset()) -> tuple[list[dict], int, set[tuple[int, int]]]:
    def payload(spans: set[tuple[int, int]]) -> list[dict]:
        pages = []
        for page in candidates:
            kept = [s for s in page["sentences"] if (page["page"], s[0]) in spans]
            if kept:
                pages.append({**page, "sentences": kept})
        return pages

    def cost(spans: set[tuple[int, int]]) -> int:
        return len(tokenizer.tokenize(json.dumps(payload(spans), ensure_ascii=False)))

    available = {(page["page"], span[0]) for page in candidates for span in page["sentences"]}
    chosen = set(pinned)
    if chosen - available or cost(chosen) > WINDOW_CAP:
        raise ValueError("Pinned proof exceeds the window or references unavailable evidence")
    for span in order[:TARGET_WINDOW]:
        if cost(chosen | {span}) <= WINDOW_CAP:
            chosen.add(span)
    wiki_cap = min(WINDOW_CAP, cost(chosen) + WIKI_QUOTA)
    promoted = set()
    for group in groups:
        if cost(chosen | group) <= wiki_cap:
            promoted.update(group - chosen)
            chosen.update(group)
    for span in order:
        if span not in chosen and cost(chosen | {span}) <= WINDOW_CAP:
            chosen.add(span)
    return payload(chosen), cost(chosen), promoted


def audit_cases() -> list[dict]:
    chains = json.loads(CHAIN_AUDIT.read_text(encoding="utf-8"))
    cases = [{"query_id": qid, **value, "audit": "step_annotations"} for qid, value in chains.items()]
    for row in json.loads(INTENT_AUDIT.read_text(encoding="utf-8")):
        cases.append({"query_id": row["id"], "run": SOURCE.name, "tier": row["source_status"],
                      "audit": "intent_diagnostic_spans", "steps": [
                          {"fact": role, "alternatives": [[row[role]]]} for role in ("source", "target")]})
    return cases


def evidence_audit(windows: dict) -> tuple[list[dict], set[Path]]:
    archives = {}
    rows, sources = [], {CHAIN_AUDIT, INTENT_AUDIT}
    for case in audit_cases():
        root = Path("runs") / case["run"]
        if root not in archives:
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            compiled = root / config.get("compiled_file", "compiled.jsonl")
            sources.update((root / "queries.jsonl", root / "corpus.jsonl", compiled))
            archives[root] = ({q["id"]: q for q in read_jsonl(root / "queries.jsonl")},
                              {d["id"]: d for d in read_jsonl(root / "corpus.jsonl")},
                              {d["doc_id"]: d for d in read_jsonl(compiled)})
        queries, corpus, compiled = archives[root]
        qid = case["query_id"]
        query = queries[qid]
        docs = [corpus[c["id"]] for c in query["candidates"]]
        if any(compiled[d["id"]]["source_version"] != d["version"] for d in docs):
            raise ValueError(f"Audit source mismatch for {qid}")
        indexed = {(page, p["index"]) for page, doc in enumerate(docs) for f in compiled[doc["id"]]["llm"]
                   for p in f["proof"]}
        for step, record in enumerate(case["steps"], 1):
            alternatives = [set(map(tuple, alt)) for alt in record["alternatives"] if alt]
            cited = set().union(*alternatives) & indexed
            matches = [{"page": page, "span": index, "predicate": f["predicate"], "object": f["object"]}
                       for page, index in sorted(cited) for f in compiled[docs[page]["id"]]["llm"]
                       if index in {p["index"] for p in f["proof"]}]
            rows.append({"audit": case["audit"], "tier": case["tier"], "query_id": qid, "step": step,
                         "fact": record["fact"], "original_spans": [describe(docs, group) for group in alternatives],
                         "wiki_covers_all_spans_of_an_alternative": any(group <= indexed for group in alternatives),
                         "wiki_records_on_those_spans": matches,
                         "window_coverage": {method: any(group <= pool for group in alternatives)
                                             for method, pool in windows.get(qid, {}).items()}})
    return rows, sources


def prepare(output: Path) -> None:
    prepare_selectors(output)
    _, corpus, queries = verify_freeze(SOURCE)
    query_index = {(row["dataset"], row["id"]): row for row in queries}
    ranks = {(row["dataset"], row["query_id"]): row for row in read_jsonl(SOURCE / "rankings.jsonl")}
    compiled = {row["doc_id"]: row for row in read_jsonl(SOURCE / "compiled.jsonl")}
    tokenizer = BudgetTokenizer()
    inputs_path = output / "selector_inputs.jsonl"
    inputs, window_logs, windows = read_jsonl(inputs_path), [], {}
    for item in inputs:
        key = item["dataset"], item["id"]
        query = query_index[key]
        docs = [corpus[c["id"]] for c in query["candidates"]]
        rank = ranks[key]
        order = [tuple(rank["candidate_spans"][i]) for i in rank["ranking"]]
        groups = wiki_groups(item, docs, order, compiled)
        item["candidate_sets"] = {}
        windows[item["id"]] = {}
        for method, promotion in zip(WINDOW_METHODS, ([], groups), strict=True):
            candidates, cost, promoted = candidate_window(tokenizer, item["candidates"], order, promotion)
            item["candidate_sets"][method] = candidates
            spans = {(p["page"], s[0]) for p in candidates for s in p["sentences"]}
            windows[item["id"]][method] = spans
            window_logs.append({"query_id": item["id"], "method": method, "candidate_json_tokens": cost,
                                "spans": sorted(spans), "wiki_promoted": describe(docs, promoted)})
    with inputs_path.open("w", encoding="utf-8") as file:
        for item in inputs:
            append(file, item)
    jsonl(output / "windows.jsonl", window_logs)
    audit, sources = evidence_audit(windows)
    jsonl(output / "coverage-audit.jsonl", audit)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(__file__)
    snapshot = output / "source" / source.name
    snapshot.write_bytes(source.read_bytes())
    reuse_scored = PREVIOUS / "scored.jsonl"
    frozen = {*sources, snapshot, reuse_scored, inputs_path, output / "windows.jsonl", output / "coverage-audit.jsonl"}
    config["sha256"].update({p.as_posix(): sha256(p) for p in sorted(frozen)})
    config["live_code_sha256"][source.name] = sha256(source)
    config.update(methods=WINDOW_METHODS, upstream_candidate_json_cap=WINDOW_CAP, wiki_quota=WIKI_QUOTA,
                  reuse_scored=[reuse_scored.as_posix()])
    dump(config_path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
