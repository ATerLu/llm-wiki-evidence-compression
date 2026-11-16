"""Probe the frozen reader using reference paragraphs for six MuSiQue train questions."""

import json
from pathlib import Path
import random
import sys
from time import perf_counter

sys.path.insert(0, str(Path.cwd()))
from baseline import MAX_OUTPUT, SYSTEM, Reader
from prepare import read_jsonl
from score import official_answer


def diagnose(port: int, output: Path) -> None:
    baseline = Path("runs/week1-calibration-cuda2")
    directory = Path("data/processed/musique")
    corpus = {row["id"]: row for row in read_jsonl(directory / "corpus.jsonl")}
    queries = {row["id"]: row for row in read_jsonl(directory / "queries.jsonl")}
    labels = {row["id"]: row for row in read_jsonl(directory / "labels.jsonl")}
    rows = [r for r in read_jsonl(baseline / "scored.jsonl") if r["dataset"] == "musique"]
    complete = [r for r in rows if r["selected_support_complete"]]
    incomplete = [r for r in rows if not r["selected_support_complete"]]
    sample = complete + random.Random(20261109).sample(incomplete, 3)
    reader = Reader(port)
    output.mkdir()
    with (output / "diagnostic.jsonl").open("x", encoding="utf-8") as file:
        for row in sample:
            started = perf_counter()
            qid = row["query_id"]
            query, label = queries[qid], labels[qid]
            references = set(label["support"])
            candidates = [c for c in query["candidates"] if c["key"] in references]
            if {c["key"] for c in candidates} != references:
                raise ValueError(f"{qid}: missing reference paragraph")
            docs = [corpus[c["id"]] for c in candidates]
            evidence = "\n\n".join(f"[{doc['title']}]\n{doc['text']}" for doc in docs)
            result = reader.generate([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {query['question']}"},
            ], started, MAX_OUTPUT)
            em, f1 = official_answer("musique", result["answer"], label)
            record = {
                "query_id": qid, "baseline_full_support": row["selected_support_complete"],
                "baseline_em": row["em"], "baseline_f1": row["f1"], "reference_em": em,
                "reference_f1": f1, "support": candidates, "evidence": evidence, **result,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            print(qid, em, f1, flush=True)


if __name__ == "__main__":
    diagnose(int(sys.argv[1]), Path(sys.argv[2]))
