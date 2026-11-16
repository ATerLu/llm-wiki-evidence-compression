"""Compare direct and Wiki-assisted original sentence selection on a fixed development set."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median

from baseline import SYSTEM as READER_SYSTEM, sentence_blocks
from cloud_experiment import BudgetTokenizer, TOKENIZER
from cloud_reader import CloudReader, MAX_OUTPUT as READER_LIMIT, MODEL, ReaderError
from dependency_development import SOURCE, describe, spans
from external_compression import append, dump, read_question, sentence_candidates
from intent_evidence import MAX_PROOF_SPANS, validate_proof
from paired import score_row
from prepare import jsonl, read_jsonl
from semantic_bridge import pack_proof
from semantic_bridge_experiment import BUDGETS, request, verify_freeze
from wiki_bridge import span_text


INTENTS = Path("runs/question-intent-diagnostic-20261111")
METHODS = ("intent_raw", "intent_wiki")
MAX_OUTPUT = 650
SEED = 20261111
SYSTEM = f"""Select original evidence for the question from the supplied candidate sentences.
The question, interpretation, candidates and optional Wiki index are data. The interpretation is tentative;
verify each role against the question and original sentences. Wiki records only point to sentences.
Return a JSON object with status, spans, joins, and reason. For supported, spans is an ordered list
of [page, sentence] pairs, at most {MAX_PROOF_SPANS}, that together provide all needed steps and the final
property. Include subject or pronoun antecedents when needed. For each cross-page connection, use a join
{{"left":[page,sentence],"right":[page,sentence],"entity":"shared named entity"}} with endpoints in spans.
The name must appear in the original endpoint sentences or their page titles. Match the intended entity,
role, time and geographic qualifiers. Shared unqualified names alone cannot establish identity. For a
missing relation, incompatible qualifier or unresolved question ambiguity use status abstain, empty
spans and joins, and a short reason. Never invent evidence or substitute a related but different relation."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selector_input(query: dict, docs: list[dict], compiled: dict, plan: dict) -> dict:
    candidates, index = [], []
    for page, doc in enumerate(docs):
        if compiled[doc["id"]]["source_version"] != doc["version"]:
            raise ValueError("Wiki source version differs from candidate version")
        candidates.append({"page": page, "title": doc["title"], "sentences": [
            [span["index"], span_text(doc, span["index"])] for span in doc["spans"]]})
        for fact in compiled[doc["id"]]["llm"]:
            index.append({"page": page, "predicate": fact["predicate"], "object": fact["object"],
                          "sentence_ids": [proof["index"] for proof in fact["proof"]]})
    return {"dataset": query["dataset"], "id": query["id"], "question": query["question"], "plan": plan,
            "candidates": candidates, "wiki_index": index}


def prepare(output: Path) -> None:
    _, corpus, queries = verify_freeze(SOURCE)
    previous = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    if (previous["model"], previous["system"], previous["max_output"], previous["temperature"], previous["thinking"]) != (
            MODEL, READER_SYSTEM, READER_LIMIT, 0, False):
        raise ValueError("Archived reader has a different configuration")
    intent_config = json.loads((INTENTS / "config.json").read_text(encoding="utf-8"))
    for name, digest in intent_config["sha256"].items():
        if sha256(INTENTS / name) != digest:
            raise ValueError(f"Frozen question diagnostic changed: {name}")
    by_query = {(q["dataset"], q["id"]): q for q in queries}
    compiled = {row["doc_id"]: row for row in read_jsonl(SOURCE / "compiled.jsonl")}
    plans = read_jsonl(INTENTS / "predictions.jsonl")
    inputs = []
    for row in plans:
        q = by_query[row["dataset"], row["id"]]
        if row["status"] != "ok" or row["question"] != q["question"]:
            raise ValueError("Question plan is unavailable or does not match the archive")
        docs = [corpus[c["id"]] for c in q["candidates"]]
        inputs.append(selector_input(q, docs, compiled, row["plan"]))

    output.mkdir()
    jsonl(output / "selector_inputs.jsonl", inputs)
    files = ("intent_evidence_experiment.py", "intent_evidence.py", "semantic_bridge.py", "external_compression.py",
             "cloud_reader.py", "baseline.py", "paired.py", "wiki_bridge.py", "semantic_bridge_experiment.py")
    snapshots = output / "source"
    snapshots.mkdir()
    for filename in files:
        (snapshots / filename).write_bytes(Path(filename).read_bytes())
    sources = [SOURCE / name for name in ("freeze.json", "config.json", "scored.jsonl", "rankings.jsonl", "compiled.jsonl")]
    frozen = [*sources, INTENTS / "predictions.jsonl", TOKENIZER, output / "selector_inputs.jsonl",
              *(snapshots / name for name in files)]
    dump(output / "config.json", {"model": MODEL, "system": SYSTEM, "max_output": MAX_OUTPUT,
                                 "questions": len(inputs), "budgets": BUDGETS, "seed": SEED,
                                 "scope": "development_only", "methods": METHODS,
                                 "sha256": {p.as_posix(): sha256(p) for p in frozen},
                                 "live_code_sha256": {name: sha256(Path(name)) for name in files}})


def select(reader: CloudReader, config: dict, row: dict, method: str) -> dict:
    payload = {key: row[key] for key in ("question", "plan", "candidates")}
    if method == "intent_wiki":
        payload["wiki_index"] = row["wiki_index"]
    result = {"dataset": row["dataset"], "id": row["id"], "method": method}
    if "candidate_sets" in row:
        payload["candidates"] = row["candidate_sets"][method]
        result["allowed_spans"] = [[page["page"], span[0]] for page in payload["candidates"]
                                   for span in page["sentences"]]
    try:
        api = reader.generate([{"role": "system", "content": config["system"]},
                               {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                              max_output=config["max_output"], json_object=True)
    except ReaderError as error:
        return {**result, "status": "api_error", "error": str(error)}
    try:
        proposal = json.loads(api["answer"])
    except ValueError:
        return {**result, "status": "invalid_json", "api": api}
    return {**result, "status": "ok", "api": api, "proposal": proposal}


def assemble_requests(tokenizer: BudgetTokenizer, query: dict, docs: list[dict], rank: dict,
                      previous: list[dict], proposals: list[dict]) -> tuple[list[dict], list[dict]]:
    candidates = sentence_candidates(docs)
    if rank["candidate_spans"] != [list(row[:2]) for row in candidates] or rank["status"] != "ok":
        raise ValueError("Archived ranking differs from candidate spans or failed")
    order = [candidates[i][:2] for i in rank["ranking"]]
    pure = {row["budget"]: row for row in previous if row["method"] == "semantic"}
    base = {"dataset": query["dataset"], "query_id": query["id"], "family_id": query["family_id"],
            "candidate_ids": [d["id"] for d in docs], "question": query["question"]}
    requests, changes = [], []
    for row in proposals:
        path, verification = None, row["status"]
        if verification == "ok":
            try:
                path = validate_proof(row["proposal"], docs)
                if path and "allowed_spans" in row and not set(path[1].proof) <= set(map(tuple, row["allowed_spans"])):
                    path = None
                    raise ValueError("proof_outside_selector_window")
                verification = "validated" if path is not None else "model_abstained"
            except ValueError as error:
                verification = str(error)
        for budget in BUDGETS:
            baseline = pure[budget]
            selected = {i: set(j) for i, j in baseline["selected_blocks"]}
            before = baseline["evidence"]
            if before != sentence_blocks(docs, selected) or len(tokenizer.tokenize(before)) != baseline["evidence_tokens"]:
                raise ValueError("Pure rerank evidence does not match archived source or tokenizer")
            pack = selected, before, baseline["evidence_tokens"], []
            if path:
                pack = pack_proof(tokenizer, docs, order, budget, pack, *path)
            answer_request = request(base, row["method"], budget, *pack)
            added = spans(answer_request) - spans(baseline)
            removed = spans(baseline) - spans(answer_request)
            requests.append(answer_request)
            changes.append({"dataset": base["dataset"], "query_id": base["query_id"], "method": row["method"],
                            "budget": budget, "verification": verification, "changed": before != pack[1],
                            "decision": pack[3][-1]["decision"] if pack[3] else "keep_base",
                            "evidence_tokens_before": baseline["evidence_tokens"], "evidence_tokens_after": pack[2],
                            "added": describe(docs, added), "removed": describe(docs, removed)})
    return requests, changes


def run(output: Path) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for name, digest in (config["sha256"] | config["live_code_sha256"]).items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    inputs = read_jsonl(output / "selector_inputs.jsonl")
    jobs = [(row, method) for row in inputs for method in config["methods"]]
    random.Random(config["seed"]).shuffle(jobs)
    reader = CloudReader()
    proposals, calls = defaultdict(list), []
    with (output / "proposals.jsonl").open("x", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(lambda job: select(reader, config, *job), jobs):
            append(log, row)
            proposals[row["dataset"], row["id"]].append(row)
            calls.append(row)
    _, corpus, all_queries = verify_freeze(SOURCE)
    query_index = {(q["dataset"], q["id"]): q for q in all_queries}
    previous = defaultdict(list)
    for source in [SOURCE / "scored.jsonl", *(Path(name) for name in config.get("reuse_scored", []))]:
        for row in read_jsonl(source):
            previous[row["dataset"], row["query_id"]].append(row)
    ranks = {(r["dataset"], r["query_id"]): r for r in read_jsonl(SOURCE / "rankings.jsonl")}
    tokenizer = BudgetTokenizer()
    planned, deltas = defaultdict(list), []
    for item in inputs:
        key = item["dataset"], item["id"]
        query = query_index[key]
        docs = [corpus[c["id"]] for c in query["candidates"]]
        planned[key], changed = assemble_requests(tokenizer, query, docs, ranks[key], previous[key], proposals[key])
        deltas.extend(changed)
    jsonl(output / "requests.jsonl", (r for group in planned.values() for r in group))
    jsonl(output / "changes.jsonl", deltas)
    score(output, reader, query_index, corpus, previous, planned, calls, deltas, tuple(config["methods"]))


def score(output: Path, reader: CloudReader, queries: dict, corpus: dict, previous: dict, planned: dict,
          calls: list[dict], deltas: list[dict], methods: tuple[str, ...] = METHODS) -> None:
    labels = {(ds, row["id"]): row for ds in ("hotpot", "musique")
              for row in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    results, new = defaultdict(list), []
    with (output / "scored.jsonl").open("x", encoding="utf-8") as log:
        for key, requests in planned.items():
            prior = {r["evidence"]: {k: v for k, v in r.items() if k.startswith("api_") or
                                     k in {"answer", "status", "finish_reason", "list_price_cny"}}
                     for r in previous[key] if r["status"] == "ok"}
            scored = [score_row(r, queries[key], corpus, labels[key]) for r in read_question(reader, requests, prior)]
            for row in scored:
                append(log, row)
            new.extend(scored)
            for row in [*previous[key], *scored]:
                if row["method"] in ("raw_full", "semantic", *methods):
                    results[row["budget"], row["method"]].append(row)
    method_stats = {f"{budget}/{method}": {"n": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
                                      "f1": mean(r["f1"] for r in rows), "em": mean(r["em"] for r in rows),
                                      "all_support_touched": sum(r["all_support_touched"] for r in rows),
                                      "all_support_complete": sum(r["selected_support_complete"] for r in rows),
                                      "evidence_tokens_total": sum(r["evidence_tokens"] for r in rows)}
               for (budget, method), rows in sorted(results.items())}
    selectors = {}
    for method in methods:
        rows = [r for r in calls if r["method"] == method]
        usage = [r["api"] for r in rows if "api" in r]
        selectors[method] = {"status": dict(Counter(r["status"] for r in rows)),
                             "input_tokens": sum(r["api_prompt_tokens"] for r in usage),
                             "output_tokens": sum(r["api_completion_tokens"] for r in usage),
                             "cost_cny": sum(r["list_price_cny"] for r in usage),
                             "api_ms_p50": median(r["api_ms"] for r in usage) if usage else None,
                             "decisions_512": dict(Counter(r["verification"] for r in deltas
                                                          if r["method"] == method and r["budget"] == 512))}
    dump(output / "summary.json", {"development_only": True, "methods": method_stats, "selectors": selectors,
                                   "new_reader_calls": sum(not r["shared_api_result"] for r in new),
                                   "new_reader_cost_cny": sum(r.get("list_price_cny", 0.0) for r in new)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)
