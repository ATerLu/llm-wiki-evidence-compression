"""Protect title-linked evidence within the original candidate pool."""

from pathlib import Path
import re
import sqlite3
import unicodedata

from baseline import sentence_anchors, sentence_blocks, words

SEED_ANCHORS = 24
MAX_BUNDLES = 3
TARGET_SPANS = 2


def accent_fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def title_surface(title: str) -> str:
    return accent_fold(re.sub(r"\s+\([^()]+\)$", "", title)).strip()


def literal(surface: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)")


def build_titles(corpus: dict, destination: Path) -> dict[str, str]:
    titles = {doc_id: title_surface(doc["title"]) for doc_id, doc in corpus.items()}
    with sqlite3.connect(destination) as db:
        db.execute("CREATE TABLE pages (doc_id TEXT PRIMARY KEY, source_version TEXT, title TEXT, surface TEXT)")
        db.executemany("INSERT INTO pages VALUES (?, ?, ?, ?)", (
            (i, doc["version"], doc["title"], titles[i]) for i, doc in corpus.items()
        ))
    return titles


def span_text(doc: dict, index: int) -> str:
    span = doc["spans"][index]
    return doc["text"][span["start"]:span["end"]]


def bridge_candidates(question: str, docs: list[dict], titles: dict[str, str],
                      ranked: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    surfaces = [titles[doc["id"]] for doc in docs]
    unique = {s for s in surfaces if len(s) >= 4 and s[0].isupper() and surfaces.count(s) == 1}
    question_folded = accent_fold(question)
    patterns = {i: literal(s) for i, s in enumerate(surfaces) if s in unique}
    target = {}
    for i, j in ranked:
        target.setdefault(i, []).append(j)
    rank = {pair: position for position, pair in enumerate(ranked)}
    terms = words(question)
    links = {}
    direct = {i for i, pattern in patterns.items() if pattern.search(question_folded)}
    seeds = dict.fromkeys([
        *((i, j) for i in sorted(direct) for j in target.get(i, [])[:2]),
        *ranked[:SEED_ANCHORS],
    ])
    for i, j in seeds:
        text = span_text(docs[i], j)
        title_in_question = i in patterns and bool(patterns[i].search(question_folded))
        if not title_in_question and len(words(text) & terms) < 2:
            continue
        folded = accent_fold(text)
        matches = [
            (k, target[k][0]) for k, pattern in patterns.items()
            if k != i and k in target and not pattern.search(question_folded) and pattern.search(folded)
        ]
        if matches:
            links[i, j] = sorted(matches, key=rank.__getitem__)
    return links


def pack_bridges(tokenizer, question: str, docs: list[dict], titles: dict[str, str],
                 budget: int) -> tuple[dict[int, set[int]], str, int, list[dict]]:
    ranked = sentence_anchors(question, docs)
    links = bridge_candidates(question, docs, titles, ranked)
    per_doc = {}
    for i, j in ranked:
        per_doc.setdefault(i, []).append(j)
    question_folded = accent_fold(question)
    direct = [anchor for anchor in ranked if anchor in links
              and literal(titles[docs[anchor[0]]["id"]]).search(question_folded)]
    order = dict.fromkeys([*direct, *ranked])
    selected, evidence, count, events = {}, "", 0, []
    added = 0
    for i, j in order:
        if j in selected.get(i, set()):
            continue
        proposed = {**selected, i: selected.get(i, set()) | {j}}
        choices = links.get((i, j), [])
        if added < MAX_BUNDLES and choices:
            k, _ = choices[0]
            new = set(per_doc[k][:TARGET_SPANS]) - selected.get(k, set())
        else:
            new = set()
        if new:
            expanded = {**proposed, k: proposed.get(k, set()) | new}
            candidate = sentence_blocks(docs, expanded)
            tokens = len(tokenizer.tokenize(candidate))
            accepted = tokens <= budget
            for span_index in sorted(new):
                events.append({
                    "kind": "explicit_title_mention", "anchor": [i, j], "proof": [k, span_index],
                    "title": docs[k]["title"], "doc_id": docs[k]["id"], "version": docs[k]["version"],
                    "span": docs[k]["spans"][span_index], "decision": "added" if accepted else "over_budget",
                })
            if accepted:
                selected, evidence, count = expanded, candidate, tokens
                added += 1
                continue
        candidate = sentence_blocks(docs, proposed)
        tokens = len(tokenizer.tokenize(candidate))
        if tokens <= budget:
            selected, evidence, count = proposed, candidate, tokens
    return selected, evidence, count, events


FACETS = (
    (r"\bcounty\b", r"\bcounty\b"),
    (r"\b(?:college|university|school|educat\w*)\b", r"\b(?:attend\w*|graduat\w*|universit\w*|college|school|studied)\b"),
    (r"\b(?:founded|established|foundation)\b", r"\b(?:found(?:ed|ing)|establish\w*|incorporat\w*|launch\w*)\b"),
    (r"\b(?:language|spoken|speaks)\b", r"\b(?:language|speak\w*|spoken)\b"),
    (r"\b(?:develop\w*|publish\w*|produc\w*|record label)\b", r"\b(?:develop\w*|publish\w*|produc\w*|record(?:s| label))\b"),
    (r"\b(?:born|birth)\b", r"\b(?:born|birth|baptiz\w*)\b"),
    (r"\b(?:died|die|death)\b", r"\b(?:died|death|passed away)\b|\d\s*[–—-]\s*\w"),
    (r"\b(?:what year|what date|which year|which date|when)\b", r"\b(?:\d{3,4}|January|February|March|April|May|June|July|August|September|October|November|December)\b"),
)


def resolve_mentions(text: str, context: str, docs: list[dict], titles: dict[str, str]) -> set[int]:
    folded = accent_fold(text)
    context_terms = words(accent_fold(context))
    groups = {}
    for i, doc in enumerate(docs):
        surface = titles[doc["id"]]
        if len(surface) >= 4 and surface[0].isupper():
            groups.setdefault(surface, []).append(i)
    matches = []
    for surface, indices in groups.items():
        for found in literal(surface).finditer(folded):
            selected = None
            if len(indices) == 1:
                selected = indices[0]
            else:
                full = [i for i in indices if literal(accent_fold(docs[i]["title"])).search(folded)
                        and accent_fold(docs[i]["title"]) != surface]
                if len(full) == 1:
                    selected = full[0]
                elif not full:
                    scores = [
                        (len((words(accent_fold(docs[i]["title"])) - words(surface)) & context_terms), i)
                        for i in indices
                    ]
                    scores.sort(reverse=True)
                    if scores[0][0] > scores[1][0]:
                        selected = scores[0][1]
            matches.append((found.start(), found.end(), selected))
    # A known longer title also masks shorter matches when the longer title is ambiguous.
    return {i for start, end, i in matches if i is not None and not any(
        other_start <= start and end <= other_end and other_end - other_start > end - start
        for other_start, other_end, _ in matches
    )}


def facet_pattern(question: str) -> re.Pattern | None:
    for query, evidence in FACETS:
        if re.search(query, question, re.IGNORECASE):
            return re.compile(evidence, re.IGNORECASE)
    return None


def pack_ranked_bridges(tokenizer, question: str, docs: list[dict], titles: dict[str, str],
                        budget: int, protected: dict[int, set[int]] | None = None,
                        *, fact_hints: dict[tuple[str, int], tuple[int, ...]] | None = None,
                        ) -> tuple[dict[int, set[int]], str, int, list[dict]]:
    ranked = sentence_anchors(question, docs)
    direct = resolve_mentions(question, question, docs, titles)
    terms = words(question)
    facet = facet_pattern(question)
    hints = fact_hints or {}
    per_doc = {}
    for i, j in ranked:
        per_doc.setdefault(i, []).append(j)
    seeds = set(ranked[:SEED_ANCHORS]) | {
        (i, j) for i in direct for j in per_doc.get(i, [])[:2]
    }
    links = {}
    for i, j in ranked:
        if (i, j) not in seeds:
            continue
        anchor = span_text(docs[i], j)
        if i not in direct and len(words(anchor) & terms) < 2:
            continue
        targets = resolve_mentions(anchor, f"{question} {anchor}", docs, titles) - direct - {i}
        candidates = []
        for k in sorted(targets):
            for index in per_doc.get(k, []):
                text = span_text(docs[k], index)
                overlap = len((terms - words(titles[docs[k]["id"]])) & words(text))
                cue = bool(facet and facet.search(text))
                hinted = (docs[k]["id"], index) in hints
                if hinted or cue or overlap >= (2 if facet else 1):
                    candidates.append((-int(hinted), -int(cue), -overlap, index, k))
        if candidates:
            _, _, _, index, k = min(candidates)
            links[i, j] = (k, index)

    order = dict.fromkeys([*(pair for pair in ranked if pair in links and pair[0] in direct), *ranked])
    selected = {i: spans.copy() for i, spans in (protected or {}).items()}
    evidence = sentence_blocks(docs, selected)
    count, events = len(tokenizer.tokenize(evidence)), []
    if count > budget:
        raise ValueError("Protected spans exceed the evidence budget")
    added = 0
    for i, j in order:
        if j in selected.get(i, set()):
            continue
        proposed = {**selected, i: selected.get(i, set()) | {j}}
        if added < 2 and (i, j) in links:
            k, proof = links[i, j]
            hint_key = docs[k]["id"], proof
            bundle = {proof, *hints.get(hint_key, ())}
            if not bundle <= selected.get(k, set()):
                expanded = {**proposed, k: proposed.get(k, set()) | bundle}
                candidate = sentence_blocks(docs, expanded)
                tokens = len(tokenizer.tokenize(candidate))
                accepted = tokens <= budget
                events.append({
                    "kind": "disambiguated_facet_sentence", "anchor": [i, j], "proof": [k, proof],
                    "title": docs[k]["title"], "doc_id": docs[k]["id"], "version": docs[k]["version"],
                    "span": docs[k]["spans"][proof], "decision": "added" if accepted else "over_budget",
                    "compiled_hint": hint_key in hints, "proof_bundle": sorted(bundle),
                })
                if accepted:
                    selected, evidence, count = expanded, candidate, tokens
                    added += 1
                    continue
        candidate = sentence_blocks(docs, proposed)
        tokens = len(tokenizer.tokenize(candidate))
        if tokens <= budget:
            selected, evidence, count = proposed, candidate, tokens
    return selected, evidence, count, events


def guarded_bridges(tokenizer, question: str, docs: list[dict], titles: dict[str, str], budget: int,
                    *, fact_hints: dict[tuple[str, int], tuple[int, ...]] | None = None,
                    ) -> tuple[dict[int, set[int]], str, int, list[dict], dict]:
    from baseline import pack_sentences

    plain = pack_ranked_bridges(tokenizer, question, docs, titles, budget, fact_hints=fact_hints)
    direct = resolve_mentions(question, question, docs, titles)
    facet = facet_pattern(question)
    protected = {}
    if facet and direct:
        baseline = pack_sentences(tokenizer, question, docs, budget, 0)[0]
        for i, j in sentence_anchors(question, docs):
            if (i in direct and i not in protected and j in baseline.get(i, set())
                    and facet.search(span_text(docs[i], j))):
                protected[i] = {j}
    conflicts = [(i, j) for i, spans in protected.items() for j in spans if j not in plain[0].get(i, set())]
    result = (pack_ranked_bridges(tokenizer, question, docs, titles, budget, protected, fact_hints=fact_hints)
              if conflicts else plain)
    covered = sorted(i for i in direct if result[0].get(i))
    return *result, {"protected": [[i, j] for i, spans in sorted(protected.items()) for j in sorted(spans)],
                     "conflicts": [list(x) for x in conflicts], "direct_pages": sorted(direct),
                     "direct_pages_covered": covered}
