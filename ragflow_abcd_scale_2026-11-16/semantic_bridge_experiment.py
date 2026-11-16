"""Run a paired ablation of rule and compiled links over the same semantic rank."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median

from baseline import SYSTEM, sentence_blocks
from author_bridge import author_pair, restore_author_pair
from cloud_experiment import BudgetTokenizer, DATASETS, FULL_LIMIT, TOKENIZER
from cloud_reader import CloudReader, ENDPOINT, MAX_OUTPUT, MODEL, ReaderError
from external_compression import (RERANK_ENDPOINT, RERANK_MODEL, append, dump, read_question, rerank,
                                  sentence_candidates)
from paired import score_row
from prepare import jsonl, read_jsonl
from scripts.check_paired import official_scores
from semantic_bridge import (MAX_BUNDLES, TARGET_WINDOW, assemble, choose_bundles, compiled_links,
                             restore_dependency, rule_links)
from sweep_compare import difference, measure
from wiki_bridge import SEED_ANCHORS, title_surface

BUDGETS = (512, 1024)
METHODS = ("semantic", "semantic_rule", "semantic_wiki")
KEYS = (("raw_full", FULL_LIMIT), *((m, b) for b in BUDGETS for m in METHODS))
ORDER_SEED = 20261117
SOURCE_FILES = ("semantic_bridge_experiment.py", "semantic_bridge.py", "external_compression.py", "author_bridge.py",
                "semantic_bridge_compile.py",
                "wiki_bridge.py", "wiki_compile_v2.py", "wiki_compile_pilot.py", "wiki_compiler_heldout.py",
                "baseline.py", "cloud_reader.py", "cloud_experiment.py", "prepare.py", "paired.py", "score.py",
                "sweep_compare.py", "scripts/check_paired.py")


def score_candidates(question: str, docs: list[dict]) -> tuple[list, dict]:
    candidates = sentence_candidates(docs)
    try:
        rank = rerank(question, [candidate[2] for candidate in candidates])
    except ReaderError as error:
        rank = {"status": "error", "error": str(error)}
    return candidates, {"candidate_spans": [item[:2] for item in candidates], **rank}


def packs(tokenizer: BudgetTokenizer, query: dict, docs: list[dict], facts: list[list[dict]],
          *, author_extension: bool = False, dependency_extension: bool = False,
          archived_rank: dict | None = None) -> tuple[dict, dict]:
    question = query["question"]
    if archived_rank is None:
        candidates, rank = score_candidates(question, docs)
    else:
        candidates, rank = sentence_candidates(docs), archived_rank
        if [item[:2] for item in candidates] != list(map(tuple, rank["candidate_spans"])):
            raise ValueError(f"Archived rank does not match candidates for {query['id']}")
    positions = rank.get("ranking", [])
    order = [candidates[i][:2] for i in positions]
    titles = {d["id"]: title_surface(d["title"]) for d in docs}
    rules = rule_links(question, docs, titles, order[:SEED_ANCHORS])
    wiki = compiled_links(question, docs, titles, order[:SEED_ANCHORS], rules, facts)
    pair = author_pair(question, docs, titles, order, facts) if author_extension else None
    options = ({}, choose_bundles(question, docs, order, rules), choose_bundles(question, docs, order, wiki))
    base = {"dataset": query["dataset"], "query_id": query["id"], "family_id": query["family_id"],
            "candidate_ids": [d["id"] for d in docs], "question": question}
    full = {i: {s["index"] for s in d["spans"]} for i, d in enumerate(docs)}
    text = sentence_blocks(docs, full)
    result = {KEYS[0]: request(base, *KEYS[0], full, text, len(tokenizer.tokenize(text)), [])}
    for budget in BUDGETS:
        for method, bundles in zip(METHODS, options, strict=True):
            pack = assemble(tokenizer, docs, candidates, positions, budget, bundles)
            if method == "semantic_wiki":
                if dependency_extension:
                    pack = restore_dependency(tokenizer, question, docs, titles, order, facts, budget, pack)
                elif pair:
                    pack = restore_author_pair(tokenizer, docs, order, budget, pack, pair)
            row = request(base, method, budget, *pack)
            if rank["status"] != "ok":
                row["compression_error"] = rank["error"]
            result[method, budget] = row
    log = {**rank, "dataset": query["dataset"], "query_id": query["id"],
           "rule_link_count": sum(map(len, rules.values())),
           "new_compiled_link_count": sum(len(edges) - len(rules.get(anchor, {})) for anchor, edges in wiki.items()),
           "same_author_candidate": bool(pair)}
    return result, log


def request(base: dict, method: str, budget: int, selected: dict, evidence: str, tokens: int, events: list) -> dict:
    if tokens > budget:
        raise ValueError(f"Budget exceeded in {base['query_id']}: {tokens} > {budget}")
    return {**base, "method": method, "budget": budget, "selected_blocks": sorted((i, sorted(s)) for i, s in selected.items()),
            "evidence": evidence, "evidence_tokens": tokens, "wiki_edges": events}


def verify_freeze(output: Path) -> tuple[dict, dict, list]:
    config = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    for name, digest in config["input_sha256"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input differs from frozen digest: {name}")
    if hashlib.sha256((output / "compile-source.py").read_bytes()).hexdigest() != config["compile_source_sha256"]:
        raise ValueError("Compiler snapshot differs from freeze")
    queries = read_jsonl(output / "queries.jsonl")
    corpus = {d["id"]: d for d in read_jsonl(output / "corpus.jsonl")}
    for ds in DATASETS:
        subset = [q for q in queries if q["dataset"] == ds]
        exclusions = config["exclusions"][ds]
        assert len(subset) == len({q["id"] for q in subset}) == config["per_dataset"]
        if not config.get("allow_within_cohort_family_reuse", False):
            assert len({q["family_id"] for q in subset}) == config["per_dataset"]
        assert {q["family_id"] for q in subset}.isdisjoint(exclusions["excluded_official_families"])
        assert {q["id"] for q in subset}.isdisjoint(exclusions["earlier_ids"])
    return config, corpus, queries


def run(output: Path, workers: int, *, dependency_extension: bool = False) -> None:
    frozen, corpus, queries = verify_freeze(output)
    author_extension = frozen.get("author_extension", False)
    if author_extension and dependency_extension:
        raise ValueError("Choose a single bridge extension per experiment")
    compiled_file = "compiled-reviewed.jsonl" if author_extension else "compiled.jsonl"
    compiled = {r["doc_id"]: r for r in read_jsonl(output / compiled_file)}
    assert compiled.keys() == corpus.keys()
    assert all(row["source_version"] == corpus[doc]["version"] for doc, row in compiled.items())
    source = output / "source"
    source.mkdir()
    hashes = {}
    for name in SOURCE_FILES:
        contents = Path(name).read_bytes()
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        hashes[name] = hashlib.sha256(contents).hexdigest()
    rng = random.Random(ORDER_SEED)
    jobs = [(q, rng.sample(KEYS, len(KEYS))) for q in rng.sample(queries, len(queries))]
    config = {"start_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL, "endpoint": ENDPOINT,
              "system": SYSTEM, "max_output": MAX_OUTPUT, "temperature": 0, "thinking": False,
              "rerank_model": RERANK_MODEL, "rerank_endpoint": RERANK_ENDPOINT, "order_seed": ORDER_SEED,
              "workers": workers, "code_sha256": hashes, "max_bundles": MAX_BUNDLES, "target_window": TARGET_WINDOW,
              "author_extension": author_extension, "dependency_extension": dependency_extension,
              "compiled_file": compiled_file,
              "seed_anchors": SEED_ANCHORS, "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
              "plan": [[q["dataset"], q["id"], m, b] for q, order in jobs for m, b in order]}
    dump(output / "config.json", config)
    tokenizer, reader = BudgetTokenizer(), CloudReader()

    def answer(job):
        query, order = job
        docs = [corpus[c["id"]] for c in query["candidates"]]
        rows, ranking = packs(tokenizer, query, docs, [compiled[d["id"]]["llm"] for d in docs],
                              author_extension=author_extension, dependency_extension=dependency_extension)
        return read_question(reader, [rows[key] for key in order], {}), ranking

    with ThreadPoolExecutor(max_workers=workers) as pool, \
            (output / "predictions.jsonl").open("x", encoding="utf-8") as answers, \
            (output / "rankings.jsonl").open("x", encoding="utf-8") as ranks:
        for n, (rows, ranking) in enumerate(pool.map(answer, jobs), 1):
            append(ranks, ranking)
            for row in rows:
                append(answers, row)
            if n % 10 == 0 or any(r["status"] != "ok" for r in rows):
                print(f"QA {n}/{len(jobs)}, {sum(r['status'] != 'ok' for r in rows)} failures", flush=True)
    config["end_utc"] = datetime.now(timezone.utc).isoformat()
    dump(output / "config.json", config)


def measures(group: dict, ids: list[str]) -> dict:
    rows = [group[q] for q in ids]
    return {**measure(group, ids), "strict_recall": mean(r["strict_recall"] for r in rows),
            "touched_recall": mean(r["touched_recall"] for r in rows),
            "evidence_p50": median(r["evidence_tokens"] for r in rows),
            "bundle_questions": sum(any(e["decision"] == "added" for e in r["wiki_edges"]) for r in rows),
            "compiled_bundle_questions": sum(any(e["kind"] == "compiled" and e["decision"] == "added"
                                                for e in r["wiki_edges"]) for r in rows)}


def evaluate(output: Path) -> None:
    _, corpus, queries = verify_freeze(output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(output / "predictions.jsonl")
    assert "end_utc" in config and [[r[k] for k in ("dataset", "query_id", "method", "budget")]
                                    for r in predictions] == config["plan"]
    assert all(hashlib.sha256((output / "source" / name).read_bytes()).hexdigest() == digest
               for name, digest in config["code_sha256"].items())
    tokenizer = BudgetTokenizer()
    assert all(len(tokenizer.tokenize(r["evidence"])) == r["evidence_tokens"] for r in predictions)
    rankings = read_jsonl(output / "rankings.jsonl")
    compiled = read_jsonl(output / config.get("compiled_file", "compiled.jsonl"))
    compile_api = read_jsonl(output / "compile-api.jsonl")
    summary = {"answer_rows": len(predictions), "answer_status": dict(Counter(r["status"] for r in predictions)),
               "answer_requests": sum(not r["shared_api_result"] for r in predictions if "shared_api_result" in r),
               "rank_requests": len(rankings), "compile_requests": len(compile_api),
               "compiled_pages": dict(Counter(r["parse"] for r in compiled)),
               "compiled_facts": sum(len(r["llm"]) for r in compiled),
               "cost_cny": {k: sum(r.get("list_price_cny", 0) for r in rows) for k, rows in
                            (("answer", predictions), ("rerank", rankings), ("compile", compile_api))}, "datasets": {}}
    scored = []
    for ds in DATASETS:
        labels = {r["id"]: r for r in read_jsonl(output / f"labels-{ds}.jsonl")}
        qmap = {q["id"]: q for q in queries if q["dataset"] == ds}
        rows = [score_row(r, qmap[r["query_id"]], corpus, labels[r["query_id"]])
                for r in predictions if r["dataset"] == ds]
        scored.extend(rows)
        summary["datasets"][ds] = dataset_results(ds, rows, labels, [r for r in rankings if r["dataset"] == ds])
    jsonl(output / "scored.jsonl", scored)
    dump(output / "comparison.json", summary)


def dataset_results(ds: str, rows: list[dict], labels: dict, rankings: list[dict]) -> dict:
    groups = {key: {r["query_id"]: r for r in rows if (r["method"], r["budget"]) == key} for key in KEYS}
    common = sorted(set.intersection(*({q for q, r in g.items() if r["status"] == "ok"} for g in groups.values())))
    for group in groups.values():
        values = [group[q] for q in common]
        for actual, metric in zip(official_scores(ds, values, labels), ("em", "f1"), strict=True):
            computed = mean(r[metric] for r in values)
            assert math.isclose(actual, round(computed, 3) if ds == "musique" else computed, abs_tol=1e-12)
    stats = {"common_n": len(common), "a": measures(groups[KEYS[0]], common), "budgets": {},
             "questions_with_new_compiled_links": sum(r["new_compiled_link_count"] > 0 for r in rankings),
             "questions_with_same_author_candidates": sum(r.get("same_author_candidate", False) for r in rankings),
             "failures": [r["query_id"] for r in rows if r["status"] != "ok"]}
    for budget in BUDGETS:
        pure, rules, wiki = (groups[m, budget] for m in METHODS)
        pairs = (("rule_vs_semantic", pure, rules), ("wiki_vs_semantic", pure, wiki), ("wiki_vs_rule", rules, wiki))
        stats["budgets"][budget] = {
            **{m: measures(g, common) for m, g in zip(METHODS, (pure, rules, wiki), strict=True)},
            **{name: {**difference(a, b, common),
                      "changed_input": sum(a[q]["evidence"] != b[q]["evidence"] for q in common),
                      "all_support_gained": sum(not a[q]["all_support_touched"] and b[q]["all_support_touched"] for q in common),
                      "all_support_lost": sum(a[q]["all_support_touched"] and not b[q]["all_support_touched"] for q in common)}
               for name, a, b in pairs},
        }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dependency-extension", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.score_only:
        run(args.output, args.workers, dependency_extension=args.dependency_extension)
    evaluate(args.output)
