"""Compare cloud sentence selectors with frozen A/B inputs on the official-dev sweep."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, median
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from baseline import SYSTEM, sentence_blocks
from cloud_experiment import BudgetTokenizer, SWEEP_BUDGETS, TOKENIZER, data_for, evaluate, sample_for
from cloud_reader import ENDPOINT, MAX_OUTPUT, MODEL, CloudReader, NoRedirect, ReaderError
from paired import DATASETS, inputs, labels_for
from prepare import read_jsonl
from scripts.check_paired import official_scores
from sweep_compare import difference, measure

SCOPE = "official_dev"
REFERENCES = (Path("runs/cloud-a-official-dev-qwen35"), Path("runs/cloud-ranked-sweep-dev-qwen35"))
RERANK_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = "qwen3-rerank"
RERANK_PRICE = 0.5
LISTWISE_LIMIT = 1600
LISTWISE_SYSTEM = (
    "Rank the numbered source sentences for answering the question using only the supplied sources. "
    "Rank sentences that establish intermediate facts as well as sentences that directly answer the question. "
    'Consider how the sentences work together. Return a JSON object {"ranking":[...]} with integer IDs, '
    "most useful first. Include every ID exactly once, even irrelevant ones at the end. Only output the ranking."
)
METHODS = ("lexical_sentence", "wiki_ranked", "qwen3_rerank_sentence", "qwen_listwise_sentence")
FIELDS = ("dataset", "query_id", "method", "budget")
SOURCE_FILES = ("external_compression.py", "cloud_experiment.py", "cloud_reader.py", "baseline.py",
                "paired.py", "prepare.py", "score.py", "sweep_compare.py", "scripts/check_paired.py")


def dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(file, row: dict) -> None:
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()


def sentence_candidates(docs: list[dict]) -> list[tuple[int, int, str]]:
    candidates = []
    for doc_index, doc in enumerate(docs):
        for span in doc["spans"]:
            text = doc["text"][span["start"]:span["end"]].strip()
            if text:
                candidates.append((doc_index, span["index"], f"[{doc['title']}] {text}"))
    return candidates


def rerank(question: str, texts: list[str]) -> dict:
    payload = {"model": RERANK_MODEL, "input": {"query": question, "documents": texts},
               "parameters": {"top_n": len(texts), "return_documents": False}}
    request = Request(RERANK_ENDPOINT, json.dumps(payload, ensure_ascii=False).encode("utf-8"), {
        "Authorization": "Bearer " + os.environ["DASHSCOPE_API_KEY"], "Content-Type": "application/json",
    })
    started = perf_counter()
    failures = []
    opener = build_opener(NoRedirect())
    for attempt in range(1, 4):
        try:
            with opener.open(request, timeout=50) as response:
                result = json.load(response)
            ranks = result["output"]["results"]
            tokens = result["usage"]["total_tokens"]
            ids = [row["index"] for row in ranks]
            if sorted(ids) != list(range(len(texts))):
                raise ReaderError("Reranker did not return one result per candidate sentence")
            return {"status": "ok", "ranking": ids, "scores": [row["relevance_score"] for row in ranks],
                    "api_response_id": result.get("request_id"), "api_prompt_tokens": tokens,
                    "api_attempts": attempt, "api_retry_statuses": failures,
                    "api_ms": (perf_counter() - started) * 1000,
                    "list_price_cny": tokens * RERANK_PRICE / 1e6}
        except HTTPError as error:
            failure = f"HTTP {error.code}"
            error.close()
            if error.code not in {429, 500, 502, 503, 504}:
                raise ReaderError(failure) from None
        except (URLError, TimeoutError, ConnectionError) as error:
            failure = type(error).__name__
        failures.append(failure)
        if attempt < 3:
            sleep(1.5 * attempt)
    raise ReaderError("Rerank retry limit: " + ", ".join(failures))


def listwise(reader: CloudReader, question: str, texts: list[str]) -> dict:
    items = "\n".join(f"{index}: {item}" for index, item in enumerate(texts))
    result = reader.generate([{"role": "system", "content": LISTWISE_SYSTEM},
                              {"role": "user", "content": f"Question: {question}\n\nSources:\n{items}"}],
                             max_output=LISTWISE_LIMIT, json_object=True)
    raw = result.pop("answer")
    try:
        ids = json.loads(raw).get("ranking")
    except (ValueError, AttributeError):
        ids = None
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        return {**result, "status": "error", "error": "Invalid sentence ranking JSON", "raw_output": raw}
    valid = list(dict.fromkeys(i for i in ids if 0 <= i < len(texts)))
    if not valid:
        return {**result, "status": "error", "error": "No valid sentence IDs", "raw_output": raw}
    missing = sorted(set(range(len(texts))) - set(valid))
    return {**result, "ranking": valid + missing, "ranked_count": len(valid),
            "unranked_count": len(missing), "invalid_or_duplicate_count": len(ids) - len(valid),
            "raw_output": raw}


def pack_ranked(tokenizer: BudgetTokenizer, docs: list[dict], candidates: list[tuple[int, int, str]],
                ranking: list[int], budget: int, *, pinned: tuple[tuple[int, int], ...] = (),
                ) -> tuple[dict[int, set[int]], str, int, set[int]]:
    selected, accepted = {}, set()
    for i, j in pinned:
        selected.setdefault(i, set()).add(j)
    evidence = sentence_blocks(docs, selected)
    count = len(tokenizer.tokenize(evidence))
    if count > budget:
        raise ValueError("Pinned spans exceed the evidence budget")
    for index in ranking:
        doc_index, span_index, _ = candidates[index]
        proposed = {**selected, doc_index: selected.get(doc_index, set()) | {span_index}}
        rendered = sentence_blocks(docs, proposed)
        tokens = len(tokenizer.tokenize(rendered))
        if tokens <= budget:
            selected, evidence, count = proposed, rendered, tokens
            accepted.add(index)
    return selected, evidence, count, accepted


def frozen_inputs() -> dict:
    questions = {(ds, q): [] for ds, q in sample_for(SCOPE)}
    keys = ("scope", "model", "endpoint", "temperature", "thinking", "system", "max_output")
    expected = (SCOPE, MODEL, ENDPOINT, 0, False, SYSTEM, MAX_OUTPUT)
    for reference in REFERENCES:
        config = json.loads((reference / "config.json").read_text(encoding="utf-8"))
        if tuple(config[key] for key in keys) != expected or "end_utc" not in config:
            raise ValueError(f"Incompatible frozen reader configuration in {reference}")
        for row in read_jsonl(reference / "predictions.jsonl"):
            questions[row["dataset"], row["query_id"]].append(row)
    for rows in questions.values():
        methods = {(r["method"], r["budget"]) for r in rows if r["status"] == "ok"}
        expected_methods = {("raw_full", 16000)} | {(m, b) for m in METHODS[:2] for b in SWEEP_BUDGETS}
        if methods != expected_methods or len(rows) != len(methods):
            raise ValueError("Frozen A/B is missing a planned input")
    return questions


def verified_references(tokenizer: BudgetTokenizer, docs: list[dict], query: dict, rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        selected = {i: set(indices) for i, indices in row["selected_blocks"]}
        if ([doc["id"] for doc in docs] != row["candidate_ids"] or query["question"] != row["question"]
                or sentence_blocks(docs, selected) != row["evidence"]
                or len(tokenizer.tokenize(row["evidence"])) != row["evidence_tokens"]):
            raise ValueError("Frozen A/B input no longer matches source text or tokenizer")
        fields = (*FIELDS, "family_id", "candidate_ids", "selected_blocks", "wiki_edges", "evidence",
                  "evidence_tokens", "question", "assembly_ms")
        records.append({field: row[field] for field in fields})
    return records


def external_packs(tokenizer, reader, docs, reference, *, only_listwise=False):
    candidates = sentence_candidates(docs)
    texts = [candidate[2] for candidate in candidates]
    question = reference["question"]
    if not 0 < len(texts) <= 500:
        raise ValueError("Candidate sentence count outside rerank API range")
    requests, logs = [], []
    base = {k: reference[k] for k in ("dataset", "query_id", "family_id", "candidate_ids", "question")}
    selectors = [(METHODS[2], rerank), (METHODS[3], lambda q, t: listwise(reader, q, t))]
    for method, select in selectors[1:] if only_listwise else selectors:
        try:
            ranked = select(question, texts)
        except ReaderError as error:
            ranked = {"status": "error", "error": str(error)}
        logs.append({"dataset": base["dataset"], "query_id": base["query_id"], "method": method,
                     "candidate_spans": [c[:2] for c in candidates], **ranked})
        for budget in SWEEP_BUDGETS:
            start = perf_counter()
            selected, evidence, count, accepted = pack_ranked(
                tokenizer, docs, candidates, ranked.get("ranking", []), budget)
            fallback = set(ranked.get("ranking", [])[ranked.get("ranked_count", len(texts)):])
            row = {**base, "method": method, "budget": budget, "wiki_edges": [], "evidence": evidence,
                   "evidence_tokens": count, "selected_blocks": sorted((i, sorted(s)) for i, s in selected.items()),
                   "fallback_sentences_selected": len(accepted & fallback),
                   "assembly_ms": (perf_counter() - start) * 1000}
            if ranked["status"] != "ok":
                row.update(compression_error=ranked.get("error", ranked["status"]))
            requests.append(row)
    return requests, logs


def read_question(reader: CloudReader, rows: list[dict], prior: dict) -> list[dict]:
    answers = {}
    for row in rows:
        if "compression_error" in row:
            row.update(answer="", status="error", error=row["compression_error"])
            continue
        evidence = row["evidence"]
        reused = evidence in answers
        if not reused and evidence in prior:
            answers[evidence] = prior[evidence]
        elif not reused:
            try:
                answers[evidence] = reader.generate([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {row['question']}"},
                ])
            except ReaderError as error:
                answers[evidence] = {"answer": "", "status": "error", "error": str(error)}
        row.update(answers[evidence], shared_api_result=reused or evidence in prior,
                   previous_run_api_result=evidence in prior)
        if reused or evidence in prior:
            row.update(list_price_cny=0.0, api_attempts=0)
    return rows


def revision_inputs(previous: Path) -> tuple[dict, dict, dict]:
    config = json.loads((previous / "config.json").read_text(encoding="utf-8"))
    if "end_utc" not in config or config["scope"] != SCOPE or config["model"] != MODEL:
        raise ValueError("Revision requires a completed compatible run")
    predictions = read_jsonl(previous / "predictions.jsonl")
    rankings = read_jsonl(previous / "rankings.jsonl")
    rerank_rows, rerank_logs, answers = {}, {}, {}
    for row in predictions:
        key = row["dataset"], row["query_id"]
        if row["method"] == METHODS[2]:
            row_fields = (*FIELDS, "family_id", "candidate_ids", "selected_blocks", "wiki_edges", "evidence",
                          "evidence_tokens", "question", "assembly_ms", "fallback_sentences_selected")
            rerank_rows.setdefault(key, []).append({field: row[field] for field in row_fields})
        if row["status"] == "ok":
            answers.setdefault(key, {})[row["evidence"]] = {
                k: v for k, v in row.items() if k.startswith("api_") or k in
                {"answer", "status", "finish_reason", "list_price_cny"}
            }
    for row in rankings:
        if row["method"] == METHODS[2]:
            rerank_logs[row["dataset"], row["query_id"]] = {**row, "previous_run_api_result": True}
    return rerank_rows, rerank_logs, answers


def run(output: Path, workers: int, previous: Path | None) -> None:
    references = frozen_inputs()
    sources = {dataset: inputs(dataset, data_for(SCOPE)) for dataset in DATASETS}
    tokenizer, reader = BudgetTokenizer(), CloudReader()
    rng = random.Random(20261114)
    budget_plan = [(method, b) for method in METHODS for b in SWEEP_BUDGETS] + [("raw_full", 16000)]
    rerank_rows, rerank_logs, answers = revision_inputs(previous) if previous else ({}, {}, {})
    jobs = []
    for (dataset, qid), rows in references.items():
        queries, corpus = sources[dataset]
        query = queries[qid]
        docs = [corpus[c["id"]] for c in query["candidates"]]
        rng.shuffle(budget_plan)
        key = dataset, qid
        frozen = verified_references(tokenizer, docs, query, rows)
        if previous:
            frozen.extend(verified_references(tokenizer, docs, query, rerank_rows[key]))
        jobs.append((docs, frozen, budget_plan.copy(), rerank_logs.get(key), answers.get(key, {})))
    snapshot = output / "source"
    snapshot.mkdir(parents=True)
    hashes = {}
    for name in SOURCE_FILES:
        content = Path(name).read_bytes()
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    config = {"stage": "S", "scope": SCOPE, "model": MODEL, "system": SYSTEM, "workers": workers,
              "endpoint": ENDPOINT, "temperature": 0, "thinking": False, "max_output": MAX_OUTPUT,
              "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
              "source_sha256": hashes, "references": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for root in REFERENCES for p in (root / "config.json", root / "predictions.jsonl")},
              "rerank_model": RERANK_MODEL, "rerank_endpoint": RERANK_ENDPOINT,
              "listwise_system": LISTWISE_SYSTEM, "listwise_max_output": LISTWISE_LIMIT,
              "listwise_response_format": "json_object", "previous_run": str(previous) if previous else None,
              "listwise_missing_id_policy": "append in source order and log fallback selections",
              "plan": [[ds, qid, m, b] for ds, qid in references for m, b in budget_plan],
              "start_utc": datetime.now(timezone.utc).isoformat()}
    dump(output / "config.json", config)
    execute(output, jobs, tokenizer, reader, workers)
    config["end_utc"] = datetime.now(timezone.utc).isoformat()
    dump(output / "config.json", config)


def execute(output, jobs, tokenizer, reader, workers):
    def work(job):
        docs, frozen, order, rerank_log, prior = job
        external, ranking_logs = external_packs(tokenizer, reader, docs, frozen[0], only_listwise=bool(rerank_log))
        if rerank_log:
            ranking_logs.insert(0, rerank_log)
        lookup = {(r["method"], r["budget"]): r for r in frozen + external}
        return read_question(reader, [lookup[key] for key in order], prior), ranking_logs

    with ThreadPoolExecutor(max_workers=workers) as pool, \
            (output / "predictions.jsonl").open("x", encoding="utf-8") as predictions, \
            (output / "rankings.jsonl").open("x", encoding="utf-8") as rankings:
        for n, (rows, logs) in enumerate(pool.map(work, jobs), start=1):
            for row in rows:
                append(predictions, row)
            for log in logs:
                append(rankings, log)
            if n % 10 == 0 or any(r["status"] != "ok" for r in rows):
                print(f"{n}/{len(jobs)} questions, {sum(r['status'] != 'ok' for r in rows)} failed rows here",
                      flush=True)


def compare(output: Path) -> None:
    scored = read_jsonl(output / "scored.jsonl")
    rankings = read_jsonl(output / "rankings.jsonl")
    summary = {"answer_rows": len(scored), "rank_rows": len(rankings),
               "actual_answer_requests": sum(not row["shared_api_result"] for row in scored
                                             if "shared_api_result" in row),
               "answer_cost_cny": sum(row.get("list_price_cny", 0) for row in scored),
               "rank_cost_cny": sum(row.get("list_price_cny", 0) for row in rankings
                                    if not row.get("previous_run_api_result")),
               "datasets": {}}
    for dataset in DATASETS:
        labels = labels_for(dataset, data_for(SCOPE))
        groups = {(method, budget): {r["query_id"]: r for r in scored
                  if (r["dataset"], r["method"], r["budget"]) == (dataset, method, budget)}
                  for method, budget in [(m, b) for m in METHODS for b in SWEEP_BUDGETS] + [("raw_full", 16000)]}
        common = sorted(set.intersection(*({q for q, row in group.items() if row["status"] == "ok"}
                                           for group in groups.values())))
        if not common:
            raise ValueError(f"No shared successful questions for {dataset}")
        raw = groups["raw_full", 16000]
        stats = {"common_n": len(common), "raw": measure(raw, common), "budgets": {}}
        for key, group in groups.items():
            for official, metric in zip(official_scores(dataset, [group[q] for q in common], labels), ("em", "f1")):
                computed = mean(group[q][metric] for q in common)
                assert math.isclose(official, round(computed, 3) if dataset == "musique" else computed, abs_tol=1e-12)
            method, budget = key
            if method == "raw_full":
                continue
            assert all(group[q]["evidence_tokens"] <= budget for q in common)
            wiki = groups["wiki_ranked", budget]
            stats["budgets"].setdefault(budget, {})[method] = {
                **measure(group, common), "evidence_token_p50": median(group[q]["evidence_tokens"] for q in common),
                "vs_wiki": difference(wiki, group, common),
                "fallback_rows": sum(group[q].get("fallback_sentences_selected", 0) > 0 for q in common),
            }
        summary["datasets"][dataset] = stats
    dump(output / "comparison.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--revise-listwise", type=Path)
    args = parser.parse_args()
    if not args.score_only:
        run(args.output, args.workers, args.revise_listwise)
    evaluate(args.output)
    compare(args.output)
