"""Verify dependency assembly on two constructed source chains."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline import sentence_blocks
from cloud_experiment import BudgetTokenizer
from prepare import document
from semantic_bridge import dependency_proof, restore_dependency
from wiki_bridge import title_surface
from wiki_compile_v2 import filter_v2


CASES = (
    ("When was the school that Nora Vale went to established?", "Nora Vale", "Nora Vale attended Summit College.",
     "educated_at", "Summit College", "Summit College was founded in 1901.", "founded_date", "1901"),
    ("When did the writer of The Silent Crossing die?", "The Silent Crossing",
     "The Silent Crossing was written by Mara Quinn.", "created_by", "Mara Quinn", "Mara Quinn died in 1998.",
     "death_date", "1998"),
)


def facts_for(doc: dict, predicate: str, obj: str) -> list[dict]:
    facts, rejected = filter_v2(doc, [{"predicate": predicate, "object": obj, "sentence_id": 0, "anchor_id": None}])
    assert facts and not rejected, rejected
    return facts


def check(tokenizer: BudgetTokenizer, case: tuple[str, ...]) -> None:
    question, source, first, link, middle, second, requested, value = case
    docs = [document(source, [first]), document(middle, [second]), document("Unrelated article", [
        "Context one.", "Context two.", "Context three.", "Context four.",
        " ".join(["Unrelated content can fill the unused space."] * 7),
    ])]
    facts = [facts_for(docs[0], link, middle), facts_for(docs[1], requested, value), []]
    titles = {doc["id"]: title_surface(doc["title"]) for doc in docs}
    blocks = {0: {0}, 2: set(range(5))}
    text = sentence_blocks(docs, blocks)
    budget = len(tokenizer.tokenize(text))
    base = blocks, text, budget, []
    order = [(0, 0), *((2, i) for i in range(5)), (1, 0)]
    after = restore_dependency(tokenizer, question, docs, titles, order, facts, budget, base)
    assert after[0] == {0: {0}, 1: {0}, 2: {0, 1, 2, 3}}
    assert after[2] <= budget and after[3][-1]["removed_spans"] == [(2, 4)]
    assert restore_dependency(tokenizer, question, docs, titles, order, facts, budget, after) == after

    # A previous accepted bridge can pin the only remaining removable sentence.
    protected = {"anchor": (0, 0), "required": [(2, 4)], "decision": "added"}
    constrained = restore_dependency(tokenizer, question, docs, titles, order, facts, budget, (*base[:3], [protected]))
    assert constrained[:3] == base[:3] and constrained[3][-1]["decision"] == "over_budget"
    assert dependency_proof(question, docs, titles, facts, {(2, 0)}) is None
    duplicate = document(middle + " (other)", [second])
    assert dependency_proof(question, [*docs, duplicate], {**titles, duplicate["id"]: middle},
                            [*facts, []], {(0, 0)}) is None
    print(f"{link} -> {requested}: restoration, eviction, existing proof, budget fallback and title ambiguity passed")


if __name__ == "__main__":
    tokenizer = BudgetTokenizer()
    for case in CASES:
        check(tokenizer, case)
