"""Audit where archived Wiki pointers overlap evidence omitted from frozen reader packs."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dependency_development import SOURCE
from external_compression import dump
from intent_evidence_experiment import sha256
from prepare import jsonl, read_jsonl
from semantic_bridge_experiment import BUDGETS, verify_freeze
from wiki_bridge import accent_fold, literal, span_text


REPLAY = Path("runs/join-recovery-development100-20261112-verified")
MUSIQUE_DEV = Path("data/raw/musique_ans_v1.0_dev.jsonl")
METHODS = ("raw_recovery", "direct")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", accent_fold(text).casefold()))


def fact_pointers(doc: dict, compiled: dict) -> dict[int, list[dict]]:
    if compiled["source_version"] != doc["version"]:
        raise ValueError("Compiled source version differs from original")
    pointers = defaultdict(list)
    for fact in compiled["llm"]:
        if fact["source_version"] != doc["version"]:
            raise ValueError("Fact source version differs from original")
        for proof in fact["proof"]:
            if proof["text"] != span_text(doc, proof["index"]):
                raise ValueError("Compiled proof text differs from original")
            pointers[proof["index"]].append({"predicate": fact["predicate"], "object": fact["object"],
                                            "fact_sentence_id": fact["sentence_id"], "anchor_id": fact["anchor_id"]})
    return pointers


def sentence_record(doc: dict, index: int, pointers: dict, rank: dict, page: int) -> dict:
    return {"index": index, "text": span_text(doc, index), "rank": rank[page, index],
            "wiki_pointers": pointers.get(index, [])}


def unit(base: dict, kind: str, indices: set[int], doc: dict, rank: dict, pointers: dict, steps: list[dict]) -> dict:
    if not indices:
        raise ValueError("Missing evidence unit is empty")
    lines = [sentence_record(doc, i, pointers, rank, base["page"]) for i in sorted(indices)]
    return {**base, "kind": kind, "support_steps": steps, "wiki_points_into_unit": any(x["wiki_pointers"] for x in lines),
            "examined_sentences": lines}


def analyze_question(row: dict, query: dict, label: dict, docs: list[dict], compiled: dict, ranking: dict,
                     decomposition: dict, a: dict) -> list[dict]:
    qid, dataset = row["query_id"], row["dataset"]
    ranked = {tuple(ranking["candidate_spans"][i]): j for j, i in enumerate(ranking["ranking"], 1)}
    selected = {p: set(ids) for p, ids in row["selected_blocks"]}
    reference = {tuple(s) for s in label["support"]} if dataset == "hotpot" else set(label["support"])
    units, visible, retained = [], set(), set()
    for page, (candidate, doc) in enumerate(zip(query["candidates"], docs, strict=True)):
        key = candidate["key"]
        indices = {s["index"] for s in doc["spans"] if span_text(doc, s["index"]).strip()}
        kept = indices & selected.get(page, set())
        common = {"dataset": dataset, "query_id": qid, "method": row["method"], "budget": row["budget"],
                  "question": query["question"], "answer": label["answer"], "reader_answer": row["answer"],
                  "reader_f1": row["f1"], "a_answer": a["answer"], "a_f1": a["f1"], "page": page,
                  "title": doc["title"], "source_version": doc["version"], "retained_indices": sorted(kept)}
        if dataset == "hotpot":
            supporting = {i for title, i in reference if title == key}
            if not supporting:
                continue
            pointers = fact_pointers(doc, compiled[doc["id"]])
            visible.update((key, i) for i in supporting & indices)
            retained.update((key, i) for i in supporting & kept)
            units.extend(unit(common, "missing_official_sentence", {i}, doc, ranked, pointers, [])
                         for i in sorted((supporting & indices) - kept))
            continue
        if key not in reference or not indices:
            continue
        visible.add(key)
        steps = [s for s in decomposition[qid] if s["paragraph_support_idx"] == key]
        if not steps:
            raise ValueError(f"Missing decomposition for official paragraph: {qid}/{key}")
        pointers = fact_pointers(doc, compiled[doc["id"]])
        if not kept:
            units.append(unit(common, "untouched_official_paragraph", indices, doc, ranked, pointers, steps))
            continue
        retained.add(key)
        for step in steps:
            needle = normalize(step["answer"])
            if not needle:
                continue
            match = literal(needle)
            if any(match.search(normalize(span_text(doc, i))) for i in kept):
                continue
            missed = {i for i in indices - kept if match.search(normalize(span_text(doc, i)))}
            if missed:
                units.append(unit(common, "touched_paragraph_literal_screen", missed, doc, ranked, pointers, [step]))
    if (visible == reference) != row["candidate_support_complete"] or (retained == reference) != row["all_support_touched"]:
        raise ValueError(f"Audit disagrees with archived reference coverage: {qid}")
    return units


def summarize(rows: list[dict], audit: list[dict]) -> dict:
    groups, audited = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["method"], row["dataset"], row["budget"]].append(row)
    for row in audit:
        audited[row["method"], row["dataset"], row["budget"]].append(row)
    summary = {}
    for key, cohort in sorted(groups.items()):
        counts = {"questions": len(cohort), "reference_available": sum(r["candidate_support_complete"] for r in cohort),
                  "questions_with_official_gap": sum(not r["all_support_touched"] for r in cohort)}
        for kind in ("missing_official_sentence", "untouched_official_paragraph", "touched_paragraph_literal_screen"):
            units = [r for r in audited[key] if r["kind"] == kind]
            with_wiki = [r for r in units if r["wiki_points_into_unit"]]
            if units:
                counts[kind] = {"units": len(units), "questions": len({r["query_id"] for r in units}),
                                "units_with_pointer": len(with_wiki),
                                "questions_with_pointer": len({r["query_id"] for r in with_wiki})}
        summary["/".join(map(str, key))] = counts
    return summary


def main(output: Path) -> None:
    _, corpus, queries = verify_freeze(SOURCE)
    qindex = {(q["dataset"], q["id"]): q for q in queries}
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique")
              for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    decomposition = {r["id"]: r["question_decomposition"] for r in read_jsonl(MUSIQUE_DEV)
                     if ("musique", r["id"]) in qindex}
    ranks = {(r["dataset"], r["query_id"]): r for r in read_jsonl(SOURCE / "rankings.jsonl")}
    compiled = {r["doc_id"]: r for r in read_jsonl(SOURCE / "compiled.jsonl")}
    scored = read_jsonl(REPLAY / "scored.jsonl")
    baseline = {(r["dataset"], r["query_id"]): r for r in scored if r["method"] == "raw_full"}
    rows = [r for r in scored if r["method"] in METHODS]
    if Counter((r["method"], r["budget"]) for r in rows) != {(m, b): len(queries) for m in METHODS for b in BUDGETS}:
        raise ValueError("Archived comparison is incomplete")
    audit = []
    for row in rows:
        key = row["dataset"], row["query_id"]
        q = qindex[key]
        docs = [corpus[c["id"]] for c in q["candidates"]]
        audit.extend(analyze_question(row, q, labels[key], docs, compiled, ranks[key], decomposition, baseline[key]))
    output.mkdir()
    jsonl(output / "gaps.jsonl", audit)
    dump(output / "summary.json", summarize(rows, audit))
    script = Path(__file__)
    (output / script.name).write_bytes(script.read_bytes())
    inputs = [SOURCE / name for name in ("freeze.json", "queries.jsonl", "corpus.jsonl", "compiled.jsonl",
                                        "rankings.jsonl", "labels-hotpot.jsonl", "labels-musique.jsonl")]
    inputs.extend([MUSIQUE_DEV, REPLAY / "scored.jsonl", script])
    dump(output / "config.json", {"python": platform.python_version(), "completed_utc": datetime.now(timezone.utc).isoformat(),
                                  "baseline": "raw_recovery", "sensitivity": "direct", "new_model_calls": 0,
                                  "scope": "development_label_conditioned_audit", "inputs": {p.as_posix(): sha256(p) for p in inputs}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
