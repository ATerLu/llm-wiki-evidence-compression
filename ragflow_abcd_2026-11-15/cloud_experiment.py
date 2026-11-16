"""Run the frozen public QA workload with a cloud reader and score it offline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import random
from time import perf_counter

from tokenizers import Tokenizer

from baseline import SYSTEM, assemble, sentence_blocks
from cloud_reader import ENDPOINT, INPUT_PRICE, MAX_OUTPUT, MODEL, OUTPUT_PRICE, CloudReader, ReaderError
from paired import DATA, DATASETS, frozen_sample, inputs, labels_for, score_row
from prepare import jsonl, read_jsonl
from score import summarize, timing

BUDGET = 1024
FULL_LIMIT = 16000
SEED = 20261110
TOKENIZER = Path("vendor/tokenizer/qwen3.5.json")
TOKENIZER_REVISION = "e2ef7817335642e36630277239ea21d48f4e9fb4"
METHODS_B = ("lexical_paragraph", "lexical_sentence", "neighbor_sentence", "wiki_dependency")
METHODS_V = ("lexical_sentence", "wiki_bridge")
SWEEP_BUDGETS = (512, 1024, 2048)


class BudgetTokenizer:
    def __init__(self) -> None:
        self._tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids


def data_for(scope: str) -> Path:
    return {
        "official_dev": Path("data/official-dev-sweep"),
        "guard_development": Path("data/guard-development"),
        "guard_heldout": Path("data/guard-official-dev"),
    }.get(scope, DATA)


def sample_for(scope: str) -> list[tuple[str, str]]:
    if scope in ("guard_development", "guard_heldout"):
        sample = [(dataset, qid) for dataset in DATASETS for qid in
                  json.loads((data_for(scope) / dataset / "splits.json").read_text(encoding="utf-8"))["ids"]]
        random.Random(20261113).shuffle(sample)
        return sample
    if scope == "official_dev":
        sample = [(dataset, qid) for dataset in DATASETS for qid in
                  json.loads((data_for(scope) / dataset / "splits.json").read_text(encoding="utf-8"))
                  ["splits"]["official_dev"]]
        random.Random(20261112).shuffle(sample)
        return sample
    if scope != "validation":
        return frozen_sample(scope)
    rng = random.Random(20261111)
    sample = []
    for dataset in DATASETS:
        directory = DATA / dataset
        ids = json.loads((directory / "splits.json").read_text(encoding="utf-8"))["splits"]["validation"]
        families = {row["id"]: row["family_id"] for row in read_jsonl(directory / "queries.jsonl")}
        rng.shuffle(ids)
        buckets = {}
        for qid in ids:
            buckets.setdefault(families[qid], []).append(qid)
        chosen = [qid for batch in zip_longest(*buckets.values()) for qid in batch if qid][:100]
        assert len(chosen) == len(set(chosen)) == 100
        sample.extend((dataset, qid) for qid in chosen)
    rng.shuffle(sample)
    return sample


def prepare_requests(stage: str, sample: list[tuple[str, str]], output: Path,
                     data_root: Path) -> list[dict]:
    tokenizer = BudgetTokenizer()
    sources = {dataset: inputs(dataset, data_root) for dataset in DATASETS}
    if stage == "B":
        from wiki import compile_wiki, pack_dependencies

        wiki = {dataset: compile_wiki(corpus, output / f"wiki-{dataset}.sqlite3")
                for dataset, (_, corpus) in sources.items()}
    if stage in ("V", "S", "G"):
        from wiki_bridge import build_titles, guarded_bridges, pack_bridges, pack_ranked_bridges

        titles = {dataset: build_titles(corpus, output / f"wiki-titles-{dataset}.sqlite3")
                  for dataset, (_, corpus) in sources.items()}
    rng = random.Random(SEED)
    requests = []
    for dataset, qid in sample:
        queries, corpus = sources[dataset]
        query = queries[qid]
        docs = [corpus[c["id"]] for c in query["candidates"]]
        if stage == "G":
            designs = ([("wiki_ranked", 512), ("wiki_guarded", 512)] if dataset == "hotpot" else
                       [("wiki_ranked", 2048), ("wiki_guarded", 2048), ("wiki_guarded", 2560)])
            raw_blocks = {i: {s["index"] for s in doc["spans"]} for i, doc in enumerate(docs)}
            raw_tokens = len(tokenizer.tokenize(sentence_blocks(docs, raw_blocks)))
        else:
            methods = {"A": ("raw_full",), "B": METHODS_B, "V": METHODS_V,
                       "S": ("lexical_sentence", "wiki_ranked")}[stage]
            designs = [(method, budget) for method in methods
                       for budget in (SWEEP_BUDGETS if stage == "S" else (BUDGET,))]
        rng.shuffle(designs)
        for method, budget in designs:
            start = perf_counter()
            edges = []
            guard = {}
            if method == "raw_full":
                selected = {i: {s["index"] for s in doc["spans"]} for i, doc in enumerate(docs)}
                evidence = sentence_blocks(docs, selected)
                count = len(tokenizer.tokenize(evidence))
                if count > FULL_LIMIT:
                    raise ValueError(f"Full evidence exceeds safety limit: {dataset} {qid}")
            elif method == "wiki_dependency":
                selected, evidence, count, edges = pack_dependencies(tokenizer, query["question"], docs, wiki[dataset], BUDGET)
            elif method == "wiki_bridge":
                selected, evidence, count, edges = pack_bridges(tokenizer, query["question"], docs, titles[dataset], BUDGET)
            elif method == "wiki_ranked":
                selected, evidence, count, edges = pack_ranked_bridges(
                    tokenizer, query["question"], docs, titles[dataset], budget)
            elif method == "wiki_guarded":
                selected, evidence, count, edges, guard = guarded_bridges(
                    tokenizer, query["question"], docs, titles[dataset], budget)
            else:
                radius = {"lexical_paragraph": None, "lexical_sentence": 0, "neighbor_sentence": 1}[method]
                selected, evidence, count = assemble(tokenizer, query["question"], docs, budget, radius)
            requests.append({
                "dataset": dataset, "query_id": qid, "family_id": query["family_id"], "method": method,
                "budget": FULL_LIMIT if method == "raw_full" else budget,
                "candidate_ids": [doc["id"] for doc in docs],
                "selected_blocks": [[i, sorted(spans)] for i, spans in sorted(selected.items())],
                "wiki_edges": edges, "evidence": evidence, "evidence_tokens": count,
                "guard": guard, "raw_evidence_tokens": raw_tokens if stage == "G" else None,
                "question": query["question"], "assembly_ms": 1000 * (perf_counter() - start),
            })
    return requests


def run(stage: str, scope: str, output: Path, workers: int, reference_a: Path | None) -> None:
    sample = sample_for(scope)
    excluded = []
    prior = {}
    if stage in ("B", "V", "S", "G"):
        if reference_a is None:
            raise ValueError("B requires the completed raw-candidate run")
        reference = json.loads((reference_a / "config.json").read_text(encoding="utf-8"))
        rows = read_jsonl(reference_a / "scored.jsonl")
        if (reference["model"] != MODEL or reference["system"] != SYSTEM
                or reference["stage"] != "A" or reference["scope"] != scope or "end_utc" not in reference
                or {(r["dataset"], r["query_id"]) for r in rows} != set(sample)):
            raise ValueError("A does not match B's frozen sample or reader configuration")
        excluded = [[r["dataset"], r["query_id"], r.get("error", r["status"])]
                    for r in rows if r["status"] != "ok"]
        excluded_ids = {(ds, qid) for ds, qid, _ in excluded}
        sample = [(ds, qid) for ds, qid in sample if (ds, qid) not in excluded_ids]
        if stage == "G":
            api_fields = {"answer", "status", "finish_reason", "api_attempts", "api_retry_statuses",
                          "api_response_id", "api_prompt_tokens", "api_completion_tokens", "api_cached_tokens",
                          "api_ms", "list_price_cny"}
            prior = {(r["dataset"], r["query_id"], r["evidence"]): {k: r[k] for k in api_fields if k in r}
                     for r in rows if r["status"] == "ok"}
    reader = CloudReader()
    output.mkdir(parents=True)
    requests = prepare_requests(stage, sample, output, data_for(scope))
    # 256 tokens per request cover the fixed prompt, question and template overhead here.
    estimate = sum(((r["evidence_tokens"] + 256) * INPUT_PRICE + MAX_OUTPUT * OUTPUT_PRICE) / 1e6
                   for r in requests)
    if 3 * estimate > 10:
        raise ValueError(f"Estimated upper planning bound including retries is ¥{3 * estimate:.2f}")
    source = output / "source"
    source.mkdir()
    files = ["cloud_experiment.py", "cloud_reader.py", "baseline.py", "paired.py", "prepare.py", "score.py"]
    if stage == "B":
        files.append("wiki.py")
    if stage in ("V", "S", "G"):
        files.append("wiki_bridge.py")
    checksums = {}
    for name in files:
        content = Path(name).read_bytes()
        (source / name).write_bytes(content)
        checksums[name] = hashlib.sha256(content).hexdigest()
    config = {
        "start_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, "scope": scope,
        "workers": workers, "model": MODEL, "endpoint": ENDPOINT, "temperature": 0,
        "thinking": False, "system": SYSTEM, "max_output": MAX_OUTPUT,
        "budget": (None if stage == "A" else list(SWEEP_BUDGETS) if stage == "S"
                   else [512, 2048, 2560] if stage == "G" else BUDGET),
        "tokenizer_revision": TOKENIZER_REVISION,
        "reference_a": str(reference_a) if reference_a else None, "excluded_after_a": excluded,
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "planned_list_price_cny": estimate, "code_sha256": checksums,
        "plan": [[r["dataset"], r["query_id"], r["method"], r["budget"]] for r in requests],
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    unique = {}
    keys = []
    for row in requests:
        key = (row["dataset"], row["query_id"], row["evidence"])
        keys.append(key)
        unique.setdefault(key, row)
    pending = [key for key in unique if key not in prior]
    print(f"Planned {len(requests)} method rows, {len(pending)} calls; price estimate at most ¥{estimate:.2f}", flush=True)

    def generate(row: dict) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Evidence:\n{row['evidence']}\n\nQuestion: {row['question']}"},
        ]
        try:
            result = reader.generate(messages)
            return result
        except ReaderError as error:
            return {"answer": "", "status": "error", "error": str(error)}

    with (output / "predictions.jsonl").open("x", encoding="utf-8") as file:
        from collections import defaultdict

        consumers = defaultdict(list)
        for key, row in zip(keys, requests, strict=True):
            consumers[key].append(row)
        written = 0

        def write_group(key, result):
            nonlocal written
            for number, row in enumerate(consumers[key]):
                reused = bool(number) or key in prior
                joined = {**row, **result, "shared_api_result": reused}
                if "api_ms" in result:
                    joined["total_ms"] = row["assembly_ms"] + result["api_ms"]
                if reused:
                    joined["list_price_cny"] = 0.0
                    joined["api_attempts"] = 0
                file.write(json.dumps(joined, ensure_ascii=False) + "\n")
                file.flush()
                written += 1
                if written % 20 == 0 or joined["status"] != "ok" or written == len(requests):
                    print(f"{written}/{len(requests)} {row['dataset']} {row['method']} {joined['status']}", flush=True)

        for key in unique:
            if key in prior:
                write_group(key, prior[key])
        if pending:
            first_key, *rest = pending
            first = generate(unique[first_key])
            if first.get("error", "").startswith(("HTTP 401", "HTTP 403", "HTTP 404")):
                raise ReaderError(first["error"])
            write_group(first_key, first)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for key, result in zip(rest, pool.map(generate, (unique[k] for k in rest)), strict=True):
                    write_group(key, result)
    config["unique_requests"] = len(pending)
    config["end_utc"] = datetime.now(timezone.utc).isoformat()
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def evaluate(output: Path) -> None:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(output / "predictions.jsonl")
    fields = ("dataset", "query_id", "method", "budget")[:len(config["plan"][0])]
    if "end_utc" not in config or sorted([r[field] for field in fields] for r in predictions) != sorted(config["plan"]):
        raise ValueError("Run is incomplete or differs from its frozen plan")
    scored, summary = [], {}
    for dataset in DATASETS:
        queries, corpus = inputs(dataset, data_for(config["scope"]))
        labels = labels_for(dataset, data_for(config["scope"]))
        rows = [score_row(r, queries[r["query_id"]], corpus, labels[r["query_id"]])
                for r in predictions if r["dataset"] == dataset]
        scored.extend(rows)
        summary[dataset] = {}
        for method, budget in sorted({(row["method"], row["budget"]) for row in rows}):
            group = [row for row in rows if (row["method"], row["budget"]) == (method, budget)]
            stats = summarize(group)
            for field in ("api_prompt_tokens", "api_completion_tokens", "api_cached_tokens", "list_price_cny"):
                stats[field] = sum(r.get(field) or 0 for r in group)
            stats["api_cache_field_present"] = sum(r.get("api_cached_tokens") is not None for r in group)
            stats["api_retries"] = sum(max(0, r.get("api_attempts", 1) - 1) for r in group)
            stats["all_support_touched"] = sum(r["all_support_touched"] for r in group)
            stats["wiki_activated"] = sum(bool(r["wiki_edges"]) for r in group)
            stats["wiki_proof_added"] = sum(any(edge["decision"] == "added" for edge in r["wiki_edges"]) for r in group)
            stats["output_capped"] = sum(r.get("finish_reason") == "length" for r in group)
            stats["api_latency_ms"] = timing(group, "api_ms")
            key = f"{method}_{budget}" if config["stage"] in ("S", "G") else method
            summary[dataset][key] = stats
            print(dataset, method, stats["ok"], round(100 * stats["answer_f1"], 2), flush=True)
    jsonl(output / "scored.jsonl", scored)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["A", "B", "V", "S", "G", "score"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--scope", choices=["pilot", "smoke", "validation", "official_dev",
                                           "guard_development", "guard_heldout"], default="smoke")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--reference-a", type=Path)
    args = parser.parse_args()
    if args.stage != "score":
        run(args.stage, args.scope, args.output, args.workers, args.reference_a)
    evaluate(args.output)
