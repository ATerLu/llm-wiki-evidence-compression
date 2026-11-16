"""Run a label-free lexical paragraph baseline against a local llama-server."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import random
import re
import sys
from time import perf_counter
from urllib.error import URLError
from urllib.request import Request, urlopen

from prepare import read_jsonl

SYSTEM = "Answer the question using the evidence. Return only the short answer, without explanation."
CONTEXT_LIMIT = 4096
MAX_OUTPUT = 128
BUDGETS = (512, 1024, 2048)
NETWORK_TIMEOUT = 120


class Reader:
    def __init__(self, port: int):
        self.url = f"http://127.0.0.1:{port}"

    def post(self, endpoint: str, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.url + endpoint, data, {"Content-Type": "application/json"})
        return urlopen(request, timeout=NETWORK_TIMEOUT)

    def json(self, endpoint: str, payload: dict) -> dict:
        with self.post(endpoint, payload) as response:
            return json.load(response)

    def tokenize(self, text: str, special: bool = False) -> list[int]:
        return self.json("/tokenize", {
            "content": text, "add_special": special, "parse_special": special,
        })["tokens"]

    def generate(self, messages: list[dict], query_started: float, max_output: int) -> dict:
        prompt = self.json("/apply-template", {"messages": messages})["prompt"]
        tokens = self.tokenize(prompt, special=True)
        if len(tokens) + max_output > CONTEXT_LIMIT:
            raise ValueError(f"Prompt exceeds context limit: {len(tokens)} + {max_output}")
        payload = {
            "prompt": tokens, "stream": True, "cache_prompt": False, "temperature": 0,
            "seed": 20261109, "n_predict": max_output,
        }
        model_started = perf_counter()
        first_content, content, final = None, [], None
        with self.post("/completion", payload) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                event = json.loads(line[5:])
                if fragment := event.get("content"):
                    first_content = first_content or perf_counter()
                    content.append(fragment)
                if event.get("stop") is True:
                    final = event
                    break
        finished = perf_counter()
        if final is None:
            raise ValueError("Completion stream ended without final status")
        timings = final["timings"]
        cache_n = timings["cache_n"]
        if cache_n or timings["prompt_n"] != len(tokens) or final.get("truncated"):
            raise ValueError(f"Prompt verification failed: expected {len(tokens)}, server {timings}")
        answer = "".join(content).strip()
        return {
            "answer": answer, "status": "ok" if answer else "empty_answer",
            "output_capped": final.get("stop_type") == "limit",
            "total_prompt_tokens": len(tokens), "output_tokens": timings["predicted_n"],
            "cache_n": cache_n, "prompt_ms": timings["prompt_ms"],
            "generation_ms": timings["predicted_ms"],
            "model_ttft_ms": None if first_content is None else 1000 * (first_content - model_started),
            "query_ttft_ms": None if first_content is None else 1000 * (first_content - query_started),
            "total_ms": 1000 * (finished - query_started),
        }


def words(text: str) -> set[str]:
    stop = {"a", "an", "the", "in", "of", "is", "was", "to", "and", "what", "which", "who", "where", "when", "did", "does", "by", "for", "on"}
    return set(re.findall(r"\w+", text.casefold())) - stop


def pack(reader: Reader, question: str, documents: list[dict], budget: int) -> tuple[list[int], str, int]:
    terms = words(question)
    blocks = [f"[{doc['title']}]\n{doc['text']}" for doc in documents]
    scores = [
        len(terms & words(doc["text"])) + 2 * len(terms & words(doc["title"]))
        for doc in documents
    ]
    ranked = sorted(range(len(documents)), key=lambda index: -scores[index])
    chosen, evidence, token_count = [], "", 0
    for index in ranked:
        proposed = sorted([*chosen, index])
        candidate = "\n\n".join(blocks[i] for i in proposed)
        count = len(reader.tokenize(candidate))
        if count <= budget:
            chosen, evidence, token_count = proposed, candidate, count
    return chosen, evidence, token_count


def sentence_blocks(documents: list[dict], selected: dict[int, set[int]]) -> str:
    blocks = []
    for doc_index, indices in sorted(selected.items()):
        doc = documents[doc_index]
        fragments = []
        previous = None
        for index in sorted(indices):
            span = doc["spans"][index]
            if previous is not None and previous + 1 != index:
                fragments.append("\n")
            fragments.append(doc["text"][span["start"]:span["end"]])
            previous = index
        blocks.append(f"[{doc['title']}]\n{''.join(fragments)}")
    return "\n\n".join(blocks)


def sentence_anchors(question: str, documents: list[dict]) -> list[tuple[int, int]]:
    terms = words(question)
    anchors = []
    for doc_index, doc in enumerate(documents):
        title_overlap = len(terms & words(doc["title"]))
        for span in doc["spans"]:
            fragment = doc["text"][span["start"]:span["end"]]
            if fragment.strip():
                anchors.append((-len(terms & words(fragment)), -title_overlap, doc_index, span["index"]))
    return [(doc_index, index) for _, _, doc_index, index in sorted(anchors)]


def pack_sentences(reader: Reader, question: str, documents: list[dict], budget: int,
                   radius: int) -> tuple[dict[int, set[int]], str, int]:
    selected, evidence, token_count = {}, "", 0
    for doc_index, anchor in sentence_anchors(question, documents):
        current = selected.get(doc_index, set())
        end = min(len(documents[doc_index]["spans"]), anchor + radius + 1)
        window = set(range(max(0, anchor - radius), end))
        if window <= current:
            continue
        proposed = {**selected, doc_index: current | window}
        candidate = sentence_blocks(documents, proposed)
        count = len(reader.tokenize(candidate))
        if count <= budget:
            selected, evidence, token_count = proposed, candidate, count
    return selected, evidence, token_count


def assemble(reader: Reader, question: str, documents: list[dict], budget: int,
             sentence_radius: int | None) -> tuple[dict[int, set[int]], str, int]:
    if sentence_radius is not None:
        return pack_sentences(reader, question, documents, budget, sentence_radius)
    chosen, evidence, count = pack(reader, question, documents, budget)
    return {index: {s["index"] for s in documents[index]["spans"]} for index in chosen}, evidence, count


def run_query(reader: Reader, run_id: str, dataset: str, query: dict, corpus: dict, budget: int,
              *, method: str = "lexical_paragraph", sentence_radius: int | None = None) -> dict:
    started = perf_counter()
    candidates = query["candidates"]
    documents = [corpus[c["id"]] for c in candidates]
    found = perf_counter()
    row = {
        "run_id": run_id, "dataset": dataset, "query_id": query["id"], "family_id": query["family_id"],
        "method": method, "budget": budget, "candidate_ids": [d["id"] for d in documents],
        "source_versions": [d["version"] for d in documents], "view_ids": [],
        "cache_state": "server_prompt_and_ram_disabled", "retrieval_ms": 1000 * (found - started),
    }
    try:
        selected, evidence, count = assemble(reader, query["question"], documents, budget, sentence_radius)
        row.update({
            "selected_span_ids": [[documents[i]["id"], index]
                                  for i, spans in sorted(selected.items()) for index in sorted(spans)],
            "selected_candidates": [candidates[i] for i in sorted(selected)],
            "selected_blocks": [[i, sorted(spans)] for i, spans in sorted(selected.items())],
            "evidence": evidence, "evidence_tokens": count,
            "assembly_ms": 1000 * (perf_counter() - found),
        })
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {query['question']}"},
        ]
        row.update(reader.generate(messages, started, MAX_OUTPUT))
    except (URLError, OSError, ValueError, KeyError) as error:
        row.update(status="error", error=str(error), answer="", total_ms=1000 * (perf_counter() - started))
    return row


def calibrate(reader: Reader, root: Path, output: Path, per_budget: int) -> None:
    output.mkdir(parents=True)
    datasets = ["hotpot", "musique"]
    sources = {}
    plan = []
    for dataset in datasets:
        directory = root / dataset
        splits = json.loads((directory / "splits.json").read_text(encoding="utf-8"))["splits"]
        queries = {row["id"]: row for row in read_jsonl(directory / "queries.jsonl")}
        corpus = {row["id"]: row for row in read_jsonl(directory / "corpus.jsonl")}
        sources[dataset] = (queries, corpus)
        candidates = splits["smoke"].copy()
        random.Random(20261109).shuffle(candidates)
        families = {}
        for qid in candidates:
            families.setdefault(queries[qid]["family_id"], []).append(qid)
        balanced = [qid for round_ids in zip_longest(*families.values()) for qid in round_ids if qid]
        for index, qid in enumerate(balanced[:per_budget * len(BUDGETS)]):
            plan.append((dataset, qid, BUDGETS[index % len(BUDGETS)]))
    random.Random(20261109).shuffle(plan)
    metadata = {
        "start_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "context": CONTEXT_LIMIT, "max_output": MAX_OUTPUT, "temperature": 0,
        "system": SYSTEM, "seed": 20261109, "plan": plan,
        "sampling": "seeded family round-robin within the frozen smoke split",
        "manifest_sha256": hashlib.sha256(Path("downloads.json").read_bytes()).hexdigest(),
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                        for name in ("baseline.py", "prepare.py", "score.py")},
    }
    config = output / "config.json"
    config.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as file:
        for dataset, qid, budget in plan:
            queries, corpus = sources[dataset]
            row = run_query(reader, output.name, dataset, queries[qid], corpus, budget)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            print(dataset, qid, budget, row["status"], round(row["total_ms"]), flush=True)
    metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
    config.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def smoke(reader: Reader, output: Path) -> None:
    with output.open("x", encoding="utf-8") as file:
        for index in range(10):
            started = perf_counter()
            result = reader.generate([
                {"role": "system", "content": "Return only the answer."},
                {"role": "user", "content": "What is the capital of France?"},
            ], started, 16)
            file.write(json.dumps({"request": index + 1, **result}) + "\n")
            file.flush()
            print(index + 1, result["answer"], result["cache_n"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "calibrate"])
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--per-budget", type=int, choices=range(1, 34), default=5)
    args = parser.parse_args()
    reader = Reader(args.port)
    if args.mode == "smoke":
        smoke(reader, args.output)
    else:
        calibrate(reader, args.data, args.output, args.per_budget)
