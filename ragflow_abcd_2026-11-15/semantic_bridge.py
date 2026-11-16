"""Assemble evidence from one semantic rank, with optional source-grounded bridges."""

from collections import defaultdict
from dataclasses import dataclass
import re

from baseline import sentence_blocks
from cloud_experiment import BudgetTokenizer
from external_compression import pack_ranked
from wiki_bridge import SEED_ANCHORS, accent_fold, facet_pattern, resolve_mentions, span_text, title_surface
from wiki_compiler_heldout import predicates
from wiki_compile_v2 import fold

Span = tuple[int, int]
ENTITY_RELATIONS = frozenset({"located_in", "educated_at", "created_by", "performed_by", "released_by"})
MAX_BUNDLES = 2
TARGET_WINDOW = 5


@dataclass(frozen=True)
class Bridge:
    target: int
    proof: tuple[Span, ...] = ()
    relations: tuple[dict, ...] = ()


def rule_links(question: str, docs: list[dict], titles: dict, seeds: list[Span]) -> dict[Span, dict[int, Bridge]]:
    direct = resolve_mentions(question, question, docs, titles)
    result = {}
    for source, index in seeds:
        text = span_text(docs[source], index)
        targets = resolve_mentions(text, f"{question} {text}", docs, titles) - direct - {source}
        if targets:
            result[source, index] = {target: Bridge(target) for target in targets}
    return result


def alias_index(docs: list[dict], facts: list[list[dict]]) -> dict[str, tuple[int, dict]]:
    aliases, titles = defaultdict(list), defaultdict(set)
    for i, (doc, items) in enumerate(zip(docs, facts, strict=True)):
        titles[fold(title_surface(doc["title"]))].add(i)
        for fact in items:
            obj = fact["object"]
            if fact["predicate"] == "alias" and (len(fold(obj).split()) > 1 or (obj.isupper() and len(obj) >= 3)):
                aliases[fold(obj)].append((i, fact))
    return {name: entries[0] for name, entries in aliases.items()
            if len({i for i, _ in entries} | titles[name]) == 1}


def relation_spans(doc: int, fact: dict) -> list[Span]:
    return [(doc, span["index"]) for span in fact["proof"]]


def relation_key(doc: int, fact: dict) -> dict:
    return {"doc": doc, **{key: fact[key] for key in ("doc_id", "source_version", "predicate", "object", "sentence_id")}}


def compiled_links(question: str, docs: list[dict], titles: dict, seeds: list[Span],
                   rules: dict[Span, dict[int, Bridge]], facts: list[list[dict]]) -> dict[Span, dict[int, Bridge]]:
    result = {seed: links.copy() for seed, links in rules.items()}
    direct = resolve_mentions(question, question, docs, titles)
    aliases = alias_index(docs, facts)
    target_titles = defaultdict(list)
    for i, doc in enumerate(docs):
        target_titles[fold(titles[doc["id"]])].append(i)
    for source, index in seeds:
        blocked = direct | {source} | set(rules.get((source, index), {}))
        new = find_compiled_edges(source, index, docs, facts, aliases, target_titles, blocked)
        if new:
            result.setdefault((source, index), {}).update(new)
    return result


def find_compiled_edges(source: int, index: int, docs: list[dict], facts: list[list[dict]], aliases: dict,
                        target_titles: dict, blocked: set[int]) -> dict[int, Bridge]:
    found = {}
    text = span_text(docs[source], index)
    folded = f" {fold(text)} "
    for name, (target, fact) in aliases.items():
        surface = fact["object"]
        matches = (bool(re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", accent_fold(text)))
                   if surface.isupper() else f" {name} " in folded)
        if target not in blocked and matches:
            found.setdefault(target, Bridge(target, tuple(relation_spans(target, fact)), (relation_key(target, fact),)))
    for fact in facts[source]:
        if fact["predicate"] not in ENTITY_RELATIONS or index not in (fact["sentence_id"], fact["anchor_id"]):
            continue
        name = fold(fact["object"])
        exact = target_titles[name]
        alias = aliases.get(name)
        if len(exact) == 1:
            target, extra, refs = exact[0], [], []
        elif not exact and alias:
            target, alias_fact = alias
            extra, refs = relation_spans(target, alias_fact), [relation_key(target, alias_fact)]
        else:
            continue
        if target not in blocked:
            found.setdefault(target, Bridge(target, tuple(relation_spans(source, fact) + extra),
                                            tuple([relation_key(source, fact), *refs])))
    return found


def choose_bundles(question: str, docs: list[dict], order: list[Span],
                   links: dict[Span, dict[int, Bridge]]) -> dict[Span, tuple[Span, Bridge]]:
    facet = facet_pattern(question)
    by_doc = defaultdict(list)
    ranks = {span: rank for rank, span in enumerate(order)}
    for i, j in order:
        by_doc[i].append(j)
    choices = {}
    for anchor, edges in links.items():
        options = []
        for target, bridge in edges.items():
            candidates = by_doc[target][:TARGET_WINDOW]
            if candidates:
                cue = [i for i in candidates if facet and facet.search(span_text(docs[target], i))]
                span = (target, (cue or candidates)[0])
                options.append((span, bridge))
        if options:
            choices[anchor] = min(options, key=lambda pair: (ranks[pair[0]], pair[0]))
    return choices


def dependency_proof(question: str, docs: list[dict], titles: dict, facts: list[list[dict]],
                     selected: set[Span]) -> tuple[Span, Bridge] | None:
    """Require a unique target fact reached from a selected, source-grounded link."""
    subjects = resolve_mentions(question, question, docs, titles)
    requested = predicates(question)
    if len(subjects) != 1 or len(requested) != 1:
        return None
    source = next(iter(subjects))
    predicate = next(iter(requested))
    if any(fact["predicate"] == predicate for fact in facts[source]):
        return None

    aliases = alias_index(docs, facts)
    target_titles = defaultdict(set)
    for index, doc in enumerate(docs):
        target_titles[fold(titles[doc["id"]])].add(index)
    paths = defaultdict(list)
    for first in facts[source]:
        anchor = (source, first["sentence_id"])
        if first["predicate"] not in ENTITY_RELATIONS or anchor not in selected:
            continue
        name = fold(first["object"])
        exact = target_titles[name]
        alias = aliases.get(name)
        targets = exact | ({alias[0]} if alias else set())
        if len(targets) != 1 or source in targets:
            continue
        target = next(iter(targets))
        extra = relation_spans(target, alias[1]) if not exact and alias else []
        refs = [relation_key(target, alias[1])] if not exact and alias else []
        for last in facts[target]:
            if last["predicate"] != predicate:
                continue
            proof = tuple(sorted(set(relation_spans(source, first) + extra + relation_spans(target, last))))
            refs_for_path = (relation_key(source, first), *refs, relation_key(target, last))
            paths[target, fold(last["object"])].append((anchor, Bridge(target, proof, refs_for_path)))
    if len(paths) != 1:
        return None
    return min(next(iter(paths.values())), key=lambda pair: (len(set(pair[1].proof) - selected), pair[1].proof))


def restore_dependency(tokenizer: BudgetTokenizer, question: str, docs: list[dict], titles: dict,
                       order: list[Span], facts: list[list[dict]], budget: int,
                       base: tuple[dict, str, int, list]) -> tuple[dict, str, int, list]:
    selected = {(i, j) for i, indices in base[0].items() for j in indices}
    path = dependency_proof(question, docs, titles, facts, selected)
    if path is None:
        return base
    return pack_proof(tokenizer, docs, order, budget, base, *path)


def pack_proof(tokenizer: BudgetTokenizer, docs: list[dict], order: list[Span], budget: int,
               base: tuple[dict, str, int, list], anchor: Span,
               bridge: Bridge) -> tuple[dict, str, int, list]:
    blocks, _, _, events = base
    selected = {(i, j) for i, indices in blocks.items() for j in indices}
    proof = set(bridge.proof)
    if proof <= selected:
        return base

    protected = proof | (selected & set(order[:TARGET_WINDOW]))
    for event in events:
        if event["decision"] == "added":
            protected.update({tuple(event["anchor"]), *map(tuple, event["required"])})
    packed = {i: set(indices) for i, indices in blocks.items()}
    for i, j in proof:
        packed.setdefault(i, set()).add(j)
    removable = [(i, j) for i, j in reversed(order) if (i, j) in selected - protected]
    removed = []
    for count in range(len(removable) + 1):
        text = sentence_blocks(docs, packed)
        tokens = len(tokenizer.tokenize(text))
        if tokens <= budget:
            event = {"kind": "dependency", "anchor": anchor, "target": bridge.target,
                     "required": bridge.proof, "relations": bridge.relations, "decision": "added",
                     "added_spans": sorted(proof - selected), "removed_spans": removed}
            return packed, text, tokens, [*events, event]
        if count < len(removable):
            i, j = removable[count]
            packed[i].remove(j)
            if not packed[i]:
                del packed[i]
            removed.append((i, j))
    event = {"kind": "dependency", "anchor": anchor, "target": bridge.target,
             "required": bridge.proof, "relations": bridge.relations, "decision": "over_budget",
             "added_spans": [], "removed_spans": []}
    return (*base[:3], [*events, event])


def assemble(tokenizer: BudgetTokenizer, docs: list[dict], candidates: list[tuple[int, int, str]],
             ranking: list[int], budget: int, bundles: dict[Span, tuple[Span, Bridge]]) -> tuple[dict, str, int, list]:
    if not bundles:
        selected, evidence, count, _ = pack_ranked(tokenizer, docs, candidates, ranking, budget)
        return selected, evidence, count, []
    selected, evidence, count, events, added = {}, "", 0, [], 0
    for index in ranking:
        i, j = candidates[index][:2]
        if j in selected.get(i, set()):
            continue
        proposed = {**selected, i: selected.get(i, set()) | {j}}
        choice = bundles.get((i, j)) if added < MAX_BUNDLES else None
        if choice:
            target, edge = choice
            required = sorted({target, *edge.proof} - {(i, j)})
            missing = [(doc, span) for doc, span in required if span not in proposed.get(doc, set())]
            if missing:
                expanded = proposed.copy()
                for doc, span in missing:
                    expanded[doc] = expanded.get(doc, set()) | {span}
                packed = sentence_blocks(docs, expanded)
                tokens = len(tokenizer.tokenize(packed))
                events.append({"anchor": [i, j], "target": target, "required": required, "added_spans": missing,
                               "relations": edge.relations, "kind": "compiled" if edge.relations else "rule",
                               "decision": "added" if tokens <= budget else "over_budget"})
                if tokens <= budget:
                    selected, evidence, count = expanded, packed, tokens
                    added += 1
                    continue
        packed = sentence_blocks(docs, proposed)
        tokens = len(tokenizer.tokenize(packed))
        if tokens <= budget:
            selected, evidence, count = proposed, packed, tokens
    return selected, evidence, count, events
