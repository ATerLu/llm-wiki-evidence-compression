"""Use two source-grounded works by the same author to fill a missing answer step."""

from collections import defaultdict
from pathlib import Path
import json
import random
import re

from baseline import sentence_blocks
from prepare import adapt, jsonl, question_groups
from wiki_bridge import SEED_ANCHORS, span_text
from wiki_compile_v2 import fold, work_author_phrase

N = 16
SEED = 20261121
AUTHOR = re.compile(r"\b(?:author|writer|wrote|written|authored|playwright|novelist)\b", re.I)
WORK_QUERY = re.compile(r"\b(?:which|what)\b.{0,50}\b(?:work|book|novel|play|poem|story|film|manga|screenplay)\b", re.I)


def asks_work(question: str) -> bool:
    return bool(AUTHOR.search(question) and WORK_QUERY.search(question))


def freeze_authors(output: Path) -> tuple[list[dict], dict, dict]:
    from semantic_bridge_compile import prior_ids, raw_rows

    workload = Path("runs/semantic-restoration-512/reuse-rescreen.json")
    reserved = {q for pair in json.loads(workload.read_text(encoding="utf-8"))["sample"] for q in pair["ids"]}
    queries, corpus, exclusions = [], {}, {}
    for dataset in ("hotpot", "musique"):
        rows = raw_rows(dataset)
        groups = question_groups(rows, dataset)
        old = prior_ids(dataset) | (reserved if dataset == "musique" else set())
        blocked = {f for f, ids in groups.items() if old.intersection(ids)}
        index = {r["id"]: r for r in rows}
        focused, other = [], []
        rng = random.Random(SEED)
        for family, ids in sorted(groups.items()):
            if family in blocked:
                continue
            candidates = [index[q] for q in sorted(ids) if AUTHOR.search(index[q]["question"])]
            work = [r for r in candidates if asks_work(r["question"])]
            if candidates:
                (focused if work else other).append((family, rng.choice(work or candidates)))
        rng.shuffle(focused)
        rng.shuffle(other)
        picked = (focused + other)[:N]
        assert len(picked) == N
        labels = []
        for family, raw in picked:
            query, label, docs = adapt(raw, dataset, family)
            queries.append({"dataset": dataset, **query})
            labels.append(label)
            corpus.update({d["id"]: d for d in docs})
        jsonl(output / f"labels-{dataset}.jsonl", labels)
        exclusions[dataset] = {"earlier_ids": sorted(old), "excluded_official_families": sorted(blocked),
                               "eligible_families": len(focused) + len(other),
                               "work_question_families": len(focused), "selected_work_questions": min(N, len(focused)),
                               "author_sample_seed": SEED}
    jsonl(output / "queries.jsonl", queries)
    jsonl(output / "corpus.jsonl", corpus.values())
    return queries, corpus, exclusions


def author_source(question: str, docs: list[dict], titles: dict[str, str]) -> int | None:
    normalized = fold(question)
    mentions = []
    for i, doc in enumerate(docs):
        full, short = fold(doc['title']), fold(titles[doc['id']])
        for name in {full, short}:
            if len(name) < 4:
                continue
            for match in re.finditer(rf'(?<!\w){re.escape(name)}(?!\w)', normalized):
                mentions.append((match.start(), match.end(), i))
    maximal = {i for a,b,i in mentions if not any(x<=a and b<=y and y-x>b-a for x,y,_ in mentions)}
    return next(iter(maximal)) if len(maximal)==1 else None


def author_pair(question, docs, titles, order, facts):
    if not asks_work(question):
        return None
    source = author_source(question, docs, titles)
    if source is None:
        return None
    ranks = {span: i for i, span in enumerate(order)}
    by_author = defaultdict(list)
    for i, items in enumerate(facts):
        for f in items:
            if f["predicate"] == "created_by" and work_author_phrase(span_text(docs[i], f["sentence_id"]), f["object"]):
                by_author[fold(f["object"])].append((i, f))
    candidates = []
    for items in by_author.values():
        for i, first in items:
            if i != source or ranks.get((i, first["sentence_id"]), len(order)) >= SEED_ANCHORS:
                continue
            for j, second in items:
                target = (j, second["sentence_id"])
                if j != i and fold(titles[docs[j]["id"]]) != fold(titles[docs[i]["id"]]) and target in ranks:
                    candidates.append((ranks[target], target, first, second))
    return min(candidates, key=lambda x: (x[0], x[1]))[1:] if candidates else None


def restore_author_pair(tokenizer, docs, order, budget, baseline, pair):
    selected, evidence, tokens, events = baseline
    if pair is None:
        return baseline
    (target, target_id), first, second = pair
    source = next(i for i, d in enumerate(docs) if d["id"] == first["doc_id"])
    needed = {(i, s["index"]) for i, fact in ((source, first), (target, second)) for s in fact["proof"]}
    selected_spans = {(i, j) for i, js in selected.items() for j in js}
    if needed <= selected_spans:
        return baseline
    packed = {i: js.copy() for i, js in selected.items()}
    for i, j in needed:
        packed.setdefault(i, set()).add(j)
    locked = needed | (selected_spans & set(order[:5]))
    removals = [p for p in reversed(order) if p in selected_spans and p not in locked]
    text = sentence_blocks(docs, packed)
    count = len(tokenizer.tokenize(text))
    for i, j in removals:
        if count <= budget:
            break
        packed[i].remove(j)
        if not packed[i]:
            del packed[i]
        text = sentence_blocks(docs, packed)
        count = len(tokenizer.tokenize(text))
    event = {"kind": "compiled", "link_type": "same_author", "anchor": [source, first["sentence_id"]],
             "target": [target, target_id], "required": sorted(needed), "added_spans": sorted(needed - selected_spans),
             "relations": [first, second], "decision": "added" if count <= budget else "over_budget"}
    if count > budget:
        return selected, evidence, tokens, events + [event]
    event["removed_spans"] = sorted(selected_spans - {(i, j) for i, js in packed.items() for j in js})
    return packed, text, count, events + [event]
