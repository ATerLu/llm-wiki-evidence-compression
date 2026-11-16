"""Replay known positive and conflicting joins, plus source-bound Wiki controls."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dependency_development import SOURCE
from evidence_join_recovery import recover
from external_compression import dump
from mechanism_development import INTENT
from prepare import document, read_jsonl
from semantic_bridge_experiment import verify_freeze
from wiki_compile_v2 import filter_v2


def observe(name: str, payload: dict, docs: list[dict], facts: list[list[dict]], expected: tuple | None = None) -> dict:
    decisions = recover({"status": "ok", "proposal": payload}, docs, facts)
    statuses = tuple(r.status for r in decisions)
    if expected and statuses != expected:
        raise ValueError(f"{name}: expected {expected}, received {statuses}")
    return {"case": name, "statuses": statuses,
            "connections": [asdict(r.connection) if r.connection else None for r in decisions]}


def constructed() -> list[dict]:
    simple = {"status": "supported", "spans": [[0, 0], [1, 0]], "joins": []}
    docs = [document(name, [f"{name} was designed by Ada Lovelace."])
            for name in ("Instrument One", "Instrument Two")]
    wiki = [filter_v2(doc, [{"predicate": "created_by", "object": "Ada Lovelace", "sentence_id": 0}])[0]
            for doc in docs]
    versioned = [document(name, [f"{name} is performed by Apollo Controller version {version}."])
                 for name, version in (("Weekly Export", "2.1"), ("Daily Snapshot", "3.1"))]
    versioned_wiki = [filter_v2(doc, [{"predicate": "performed_by", "object": "Apollo Controller",
                                      "sentence_id": 0}])[0] for doc in versioned]
    return [observe("synthetic_wiki_only", simple, docs, wiki,
                    ("disconnected_proof", "no_grounded_connection", "recovered_wiki")),
            observe("synthetic_version_conflict", simple, versioned, versioned_wiki,
                    ("disconnected_proof", "no_grounded_connection", "conflicting_version_qualifiers"))]


def main() -> list[dict]:
    _, corpus, queries = verify_freeze(SOURCE)
    by_id = {row["id"]: row for row in queries}
    compiled = {row["doc_id"]: row["llm"] for row in read_jsonl(SOURCE / "compiled.jsonl")}

    def sources(qid: str) -> tuple[list[dict], list[list[dict]]]:
        docs = [corpus[x["id"]] for x in by_id[qid]["candidates"]]
        return docs, [compiled[doc["id"]] for doc in docs]

    archive = Path("runs/mechanism-development-20261111/proposals.jsonl")
    results = [observe(f"{r['id']}/{r['method']}", r["proposal"], *sources(r["id"])) for r in read_jsonl(archive)
               if r["method"] == "intent_raw" or r["id"] == "2hop__647996_551681"]
    recovered = {r["connections"][1]["entity"] for r in results if r["statuses"][1] == "recovered_raw"}
    if recovered != {"Newcastle upon Tyne", "Ryki County", "Gotthold Ephraim Lessing"}:
        raise ValueError(f"Known connections failed: {recovered}")
    for control in json.loads((INTENT / "guard-controls.json").read_text(encoding="utf-8"))["controls"]:
        if "forced_proposal" not in control:
            continue
        proposal = control["forced_proposal"]
        docs, facts = sources(control["id"])
        results.append(observe(control["id"] + "/explicit", proposal, docs, facts,
                               ("conflicting_place_qualifiers",) * 3))
        results.append(observe(control["id"] + "/missing", {**proposal, "joins": []}, docs, facts,
                               ("disconnected_proof", "no_grounded_connection", "no_grounded_connection")))
    return results + constructed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    results = main()
    dump(parser.parse_args().output, results)
    print(f"{len(results)} archived and synthetic controls passed")
