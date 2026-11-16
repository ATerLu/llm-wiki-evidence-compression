"""Audit paired RAGFlow and E2 logs without calling models or changing the index."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median

from cloud_reader import INPUT_PRICE, OUTPUT_PRICE
from dependency_development import SOURCE
from external_compression import dump
from prepare import read_jsonl
from ragflow_native_compare import relay_records, source_spans
from paired import support_sets
from semantic_bridge_experiment import verify_freeze
from wiki_bridge import span_text


def summarize_cost(rows):
    prompt = sum(row["api_prompt_tokens"] for row in rows)
    output = sum(row["api_completion_tokens"] for row in rows)
    return {"input": prompt, "output": output, "nominal_cny": (prompt * INPUT_PRICE + output * OUTPUT_PRICE) / 1e6}


def audit_e2(raw, routed):
    groups = defaultdict(list)
    for row in routed.values():
        for scope in ("all", row["dataset"]):
            groups[scope, row["route"]].append(row)
    audit = {}
    for (scope, route), rows in groups.items():
        reference = [raw[r["query_id"]] for r in rows]
        gate = [r["gate_api"] for r in rows if "api_prompt_tokens" in r["gate_api"]]
        audit.setdefault(scope, {})[route] = {
            "n": len(rows), "short_support_complete": sum(r["short"]["selected_support_complete"] for r in rows),
            "short_all_support_touched": sum(r["short"]["all_support_touched"] for r in rows),
            "final_f1": mean(r["f1"] for r in rows), "raw_f1": mean(r["f1"] for r in reference),
            "final_em": mean(r["em"] for r in rows), "raw_em": mean(r["em"] for r in reference),
            "raw": summarize_cost(reference), "gate": summarize_cost(gate), "final": summarize_cost(rows),
            "gate_p50_ms": median(r["api_ms"] for r in gate) if gate else None,
        }
    if set(audit["all"]) == {"short", "insufficient"}:
        hit, miss = (audit["all"][r] for r in ("short", "insufficient"))
        saving = (hit["raw"]["input"] - hit["final"]["input"]) / hit["n"]
        overhead = (miss["final"]["input"] - miss["raw"]["input"]) / miss["n"]
        audit["all"]["break_even_rate_at_observed_subgroup_means"] = overhead / (saving + overhead)
    return audit


def verified_by_question(rows, expected, arm):
    keyed = {r["query_id"]: r for r in rows if r["method"] == arm}
    if keyed.keys() != expected or sum(r["method"] == arm for r in rows) != len(expected):
        raise ValueError(f"Incomplete or duplicated results for {arm}")
    return keyed


def span_set(row):
    return {(i, j) for i, indices in row["spans"] for j in indices}


def evidence_changes(query, corpus, ref, candidate):
    before, after = span_set(ref), span_set(candidate)
    docs = [corpus[c["id"]] for c in query["candidates"]]

    def as_rows(spans):
        return [{"title": docs[i]["title"], "sentence": j, "text": span_text(docs[i], j).strip()}
                for i, j in sorted(spans)]

    return {"original_spans_removed": as_rows(before - after), "original_spans_added": as_rows(after - before)}


def paired_details(ref, arm, queries, corpus, scope):
    keys = sorted(qid for qid in ref if scope in ("all", ref[qid]["dataset"]))
    deltas = [(qid, arm[qid]["f1"] - ref[qid]["f1"]) for qid in keys]
    details = {
        "questions": len(keys), "reader_inputs_changed": sum(arm[q]["evidence"] != ref[q]["evidence"] for q in keys),
        "input_smaller": sum(arm[q]["api_prompt_tokens"] < ref[q]["api_prompt_tokens"] for q in keys),
        "input_larger": sum(arm[q]["api_prompt_tokens"] > ref[q]["api_prompt_tokens"] for q in keys),
        "em_gained": sum(arm[q]["em"] > ref[q]["em"] for q in keys),
        "em_lost": sum(arm[q]["em"] < ref[q]["em"] for q in keys),
        "strict_gained": sum(arm[q]["selected_support_complete"] > ref[q]["selected_support_complete"] for q in keys),
        "strict_lost": sum(arm[q]["selected_support_complete"] < ref[q]["selected_support_complete"] for q in keys),
        "wiki_questions": sum(arm[q]["wiki_chunks"] > 0 for q in keys),
        "f1_when_wiki_present": mean(arm[q]["f1"] for q in keys if arm[q]["wiki_chunks"] > 0)
        if any(arm[q]["wiki_chunks"] > 0 for q in keys) else None,
        "cases": [],
    }
    changes = sorted(deltas, key=lambda x: x[1])
    examples = [q for q, delta in changes if delta < -1e-9][:3]
    examples += [q for q, delta in reversed(changes) if delta > 1e-9][:3]
    if not examples:
        examples = sorted(keys, key=lambda q: abs(arm[q]["api_prompt_tokens"] - ref[q]["api_prompt_tokens"]),
                          reverse=True)[:2]
    for q in examples:
        before, after = ref[q], arm[q]
        details["cases"].append({"query_id": q, "question": queries[q]["question"],
                                 "dataset": queries[q]["dataset"], "route": after.get("route"),
                                 "before": {k: before[k] for k in ("answer", "f1", "em", "api_prompt_tokens", "wiki_chunks")},
                                 "after": {k: after[k] for k in ("answer", "f1", "em", "api_prompt_tokens", "wiki_chunks")},
                                 **evidence_changes(queries[q], corpus, before, after)})
    return details


def relay_usage(rows, start, end):
    selected = [r for r in rows if start <= r["at"] <= end]
    report = {}
    for route in sorted({r["route"] for r in selected}):
        group = [r for r in selected if r["route"] == route]
        report[route] = {
            "responses": dict(Counter(str(r["status"]) for r in group)),
            "input": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in group),
            "output": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in group),
        }
    return report


def audit_raw_snapshot(output, queries, corpus):
    score_path = output / "native-raw-snapshot-scored.jsonl"
    a = verified_by_question(read_jsonl(score_path), queries.keys(), "A_raw")
    e2 = verified_by_question(read_jsonl(output / "native-raw-snapshot-e2-scored.jsonl"), queries.keys(), "e2_gated")
    captures = {r["query_id"]: r["raw"]["chunks"] for r in read_jsonl(output / "native-raw-snapshot-retrieval.jsonl")}
    if captures.keys() != queries.keys():
        raise ValueError("Missing native raw captures")
    claims = defaultdict(set)
    for batch in read_jsonl(output / "compile-api.jsonl"):
        if batch["status"] != "ok":
            continue
        for fact in batch["claims"]:
            if corpus[fact["doc_id"]]["version"] != fact["source_version"]:
                raise ValueError("Claim source version differs from the frozen corpus")
            claims[fact["doc_id"]].add(fact["sentence"])
    labels = {(ds, r["id"]): r for ds in ("hotpot", "musique")
              for r in read_jsonl(SOURCE / f"labels-{ds}.jsonl")}
    stages = defaultdict(Counter)
    for qid, query in queries.items():
        ds = query["dataset"]
        docs = [corpus[c["id"]] for c in query["candidates"]]
        retrieved = source_spans(captures[qid], docs)
        reference, strict, touched = support_sets(ds, query, docs, labels[ds, qid], retrieved)
        verified = {i: spans & claims[docs[i]["id"]] for i, spans in retrieved.items()}
        _, claim_strict, claim_touched = support_sets(ds, query, docs, labels[ds, qid], verified)
        counts = {"retrieval_strict": reference <= strict, "retrieval_touched": reference <= touched,
                  "A_strict": a[qid]["selected_support_complete"], "A_touched": a[qid]["all_support_touched"],
                  "claim_pool_strict": reference <= claim_strict, "claim_pool_touched": reference <= claim_touched,
                  "short_strict": e2[qid]["short"]["selected_support_complete"],
                  "short_touched": e2[qid]["short"]["all_support_touched"],
                  "E2_strict": e2[qid]["selected_support_complete"], "E2_touched": e2[qid]["all_support_touched"]}
        for key in (ds, "all"):
            stages[key].update({metric: int(value) for metric, value in counts.items()})
    return {"native_e2": audit_e2(a, e2), "evidence_by_stage": dict(stages)}


def run(output, scope):
    _, corpus, frozen = verify_freeze(SOURCE)
    queries = {q["id"]: q for q in frozen}
    raw_e2 = verified_by_question(read_jsonl(SOURCE / "scored.jsonl"), queries.keys(), "raw_full")
    fixed_e2 = verified_by_question(read_jsonl(output / "e2-scored.jsonl"), queries.keys(), "e2_gated")
    audit = {"at": datetime.now(timezone.utc).isoformat(), "fixed_candidate_e2": audit_e2(raw_e2, fixed_e2)}
    if scope == "raw-snapshot":
        audit.update(audit_raw_snapshot(output, queries, corpus))
        dump(output / "native-raw-snapshot-audit.json", audit)
        return
    state = json.loads((output / f"native-{scope}.json").read_text(encoding="utf-8"))
    ids = set(state["questions"])
    native_rows = read_jsonl(output / f"native-{scope}-scored.jsonl")
    e2_rows = read_jsonl(output / f"native-{scope}-e2-scored.jsonl")
    a, b, c = (verified_by_question(native_rows, ids, method) for method in ("A_raw", "B_native_wiki", "C_guarded"))
    e2 = verified_by_question(e2_rows, ids, "e2_gated")
    audit["native_e2"] = audit_e2(a, e2)
    audit["native_comparison"] = {
        split: {"B_vs_A": paired_details(a, b, queries, corpus, split),
                "C_vs_B": paired_details(b, c, queries, corpus, split),
                "C_vs_A": paired_details(a, c, queries, corpus, split)}
        for split in ("all", "hotpot", "musique")
    }
    summary = json.loads((output / f"native-{scope}-summary.json").read_text(encoding="utf-8"))
    if scope == "full":
        usage = relay_records(output)
        audit["native_relay_before_full_retrieval"] = relay_usage(usage, "", summary["start_utc"])
        audit["native_relay_during_full_retrieval"] = relay_usage(usage, summary["start_utc"], summary["end_utc"])
    dump(output / f"native-{scope}-audit.json", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("scope", choices=("pilot", "raw-snapshot", "checkpoint", "full"))
    args = parser.parse_args()
    run(args.output, args.scope)
