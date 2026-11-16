"""Count source, sentence and logged bridge overlaps. This is a retrospective audit."""

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_experiment import data_for
from paired import inputs, labels_for
from prepare import question_groups, read_jsonl

DATASET = "musique"
RUNS = (
    ("cloud-bridge-validation-qwen35", "wiki_bridge", 1024, "validation"),
    ("cloud-guard-heldout", "wiki_guarded", 2560, "guard_heldout"),
)


def signature(row: dict, query: dict, label: dict) -> dict:
    ids = row["candidate_ids"]
    reference = set(label["support"])
    selected = {(ids[i], j) for i, indices in row["selected_blocks"] for j in indices}
    supported = {(ids[i], j) for i, indices in row["selected_blocks"] for j in indices
                 if query["candidates"][i]["key"] in reference}
    links = set()
    for edge in row["wiki_edges"]:
        if edge["decision"] != "added":
            continue
        left, right = [(ids[i], j) for i, j in (edge["anchor"], edge["proof"])]
        if left in selected and right in selected:
            links.add((left, right))
    return {
        "source_version": set(ids), "selected_span": selected, "selected_reference_paragraph_span": supported,
        "logged_target_span": {right for _, right in links}, "logged_bridge": links,
        "logged_bridge_with_both_ends_in_reference_paragraphs": {
            (left, right) for left, right in links if left in supported and right in supported
        },
    }


def audit(run: str, method: str, budget: int, scope: str) -> dict:
    rows = [r for r in read_jsonl(Path("runs") / run / "scored.jsonl")
            if (r["dataset"], r["method"], r["budget"], r["status"]) == (DATASET, method, budget, "ok")]
    queries, corpus = inputs(DATASET, data_for(scope))
    labels = labels_for(DATASET, data_for(scope))
    indexed = {r["query_id"]: signature(r, queries[r["query_id"]], labels[r["query_id"]]) for r in rows}
    groups = {"same_family": Counter(), "different_family": Counter()}
    examples = {}
    for a, b in combinations(rows, 2):
        if a["question"] == b["question"]:
            continue
        left, right = a["query_id"], b["query_id"]
        group = "same_family" if a["family_id"] == b["family_id"] else "different_family"
        counts = groups[group]
        counts["pairs"] += 1
        different_answer = labels[left]["answer"] != labels[right]["answer"]
        counts["different_reference_answer_strings"] += different_answer
        for kind, values in indexed[left].items():
            shared = values & indexed[right][kind]
            counts[kind] += bool(shared)
            counts[f"{kind}_different_answer"] += bool(shared) and different_answer
            example_key = f"{group}_{kind}"
            if shared and different_answer and example_key not in examples:
                item = sorted(shared)[0]
                doc_id = (item if kind == "source_version" else item[1][0] if kind.startswith("logged_bridge")
                          else item[0])
                examples[example_key] = {"queries": [left, right], "shared_page": corpus[doc_id]["title"],
                                         "questions": [a["question"], b["question"]]}
    return {
        "run": run, "method": method, "budget": budget, "questions": len(rows),
        "families": len({r["family_id"] for r in rows}), "groups": groups,
        "examples": examples, "assembly_ms_p50": median(r["assembly_ms"] for r in rows),
        "api_ms_p50_logged": median(r["api_ms"] for r in rows),
    }


def screen_remaining_musique() -> dict:
    # Decomposition metadata is used here only for workload construction.
    rows = read_jsonl(Path("data/raw/musique_ans_v1.0_dev.jsonl"))
    groups = question_groups(rows, DATASET)
    family = {qid: key for key, ids in groups.items() for qid in ids}
    used = {q["family_id"] for scope in ("official_dev", "guard_heldout")
            for q in read_jsonl(data_for(scope) / DATASET / "queries.jsonl")}
    remaining = [r for r in rows if family[r["id"]] not in used]
    step_index = defaultdict(list)
    for row in remaining:
        paragraphs = {p["idx"]: p for p in row["paragraphs"]}
        for step in row["question_decomposition"]:
            if re.search(r"#\d+", step["question"]):
                continue
            paragraph = paragraphs[step["paragraph_support_idx"]]
            step_index[str(step["id"])].append((row, step, paragraph))
    pairs, exact_pairs = set(), {}
    for candidates in step_index.values():
        for (a, step, left), (b, _, right) in combinations(candidates, 2):
            if a["question"] == b["question"] or a["answer"] == b["answer"]:
                continue
            ids = tuple(sorted((a["id"], b["id"])))
            pairs.add(ids)
            if (left["title"], left["paragraph_text"]) == (right["title"], right["paragraph_text"]):
                exact_pairs[ids] = {
                    "family_id": family[a["id"]], "queries": list(ids), "shared_subquestion": step["question"],
                    "shared_page": left["title"], "questions": [a["question"], b["question"]],
                }
    exact_families = Counter(v["family_id"] for v in exact_pairs.values())
    sample = []
    for example in sorted(exact_pairs.values(), key=lambda r: (exact_families[r["family_id"]], r["queries"])):
        if example["family_id"] not in {x["family_id"] for x in sample}:
            sample.append(example)
        if len(sample) == 6:
            break
    return {
        "source": "MuSiQue-Ans official dev, excluding two previously evaluated family sets",
        "remaining_questions": len(remaining), "remaining_families": len({family[r["id"]] for r in remaining}),
        "shared_independent_step_pairs_with_different_answer_strings": len(pairs),
        "families_with_candidate_pairs": len({family[a] for a, _ in pairs}),
        "same_exact_supporting_paragraph_pairs": len(exact_pairs),
        "same_exact_supporting_paragraph_families": len(exact_families),
        "largest_family_pairs": max(exact_families.values(), default=0), "illustrative_pairs": sample,
    }


if __name__ == "__main__":
    results = [audit(*settings) for settings in RUNS]
    path = Path(sys.argv[1])
    with path.open("x", encoding="utf-8") as file:
        json.dump({"retrospective": results, "workload_screen": screen_remaining_musique()},
                  file, ensure_ascii=False, indent=2)
    for result in results:
        print(result["run"], result["questions"], result["groups"])
