"""Select a source-bound answer path before calling the reader."""

from collections import defaultdict
from dataclasses import dataclass
import re

from baseline import sentence_blocks
from evidence_join_recovery import versions
from intent_evidence import state_qualifiers, validate_join
from wiki_bridge import accent_fold, literal, resolve_mentions, span_text, title_surface
from wiki_compile_pilot import DATE
from wiki_compile_v2 import binding, fold, subject_anchors


@dataclass(frozen=True)
class Relation:
    name: str
    question: str
    evidence: str


# Each cue identifies a requested relation. Sentence cues also require the value
# shape where possible, so a date alone cannot answer a question about a premiere.
RELATIONS = (
    Relation("other_location", r"\b(?:same|other|another|shares?|border|near|nearest|furthest)\b", ""),
    Relation("alias", r"\b(?:nickname|nicknamed|known as|alternate name)\b",
             r"\b(?:nicknamed|known as|also called|nickname (?:is|was|came|derived))\b"),
    Relation("premiere", r"\b(?:first air date|premiered in what|when.{0,60}(?:premier|first air))",
             rf"\b(?:premiered|first aired|debuted|began with.{0,45}airing)\s+(?:on\s+|in\s+)?{DATE}\b"),
    Relation("genre", r"\b(?:what|which) genre\b", r"\b(?:comedy|thriller|drama|horror|fantasy|science fiction|genre)\b"),
    Relation("county", r"\b(?:what|which) county\b|\bwhat administrative territorial entity\b",
             r"\b(?:located|situated|lies|part of|is in|city in|town in|municipality in|village in|includes)\b.{0,130}\b(?:County|Province|District|Prefecture|Department)\b"),
    Relation("founded_date", r"\b(?:when|what year) (?:was|were)\b.{0,85}\b(?:founded|established|incorporated|formed)\b",
             rf"\b(?:founded|established|incorporated|formed)\s+(?:in\s+|on\s+)?{DATE}\b"),
    Relation("birth_date", r"\b(?:when|what year) (?:was|were)\b.{0,85}\bborn\b",
             rf"\bborn\s+(?:in\s+|on\s+)?{DATE}\b"),
    Relation("death_date", r"\b(?:when|what year)\b.{0,65}\b(?:died|die|death)\b",
             rf"\b(?:died|passed away)\s+(?:in\s+|on\s+)?{DATE}\b"),
    Relation("educated_at", r"\bwhere\b.{0,100}\b(?:educated|studied|attended)\b",
             r"\b(?:attended|studied at|educated at|graduated from|enrolled at)\b"),
    Relation("creator", r"\b(?:who created|creator of|created by who|formed by who)\b",
             r"\b(?:created|formed) by\b"),
    Relation("occupation", r"\b(?:what field|which field|what profession|which profession)\b",
             r"\b(?:is|was)\s+(?:an?|the)\s+[^.]{0,70}\b(?:professor|scientist|physicist|writer|artist|economist|historian)\b"),
    Relation("leadership", r"\b(?:who is|who was) the (?:chairman|president|director|leader)\b",
             r"\b(?:chairman|president|director|leader)\b"),
    Relation("headquarters", r"\bwhere\b.{0,50}\bheadquarters\b",
             r"\b(?:headquarter(?:s|ed)|based)\s+in\b"),
)
SOURCE_RELATIONS = (
    (r"\b(?:at|beside|near)\b", r"\b(?:home of|located in|located at|beside|near|adjacent to)\b", ("located_in",)),
    (r"\b(?:author|writ\w*)\b", r"\b(?:author|written by|wrote|novel by|play by|book by)\b", ("created_by",)),
    (r"\b(?:cast|star|actor|actress|played|plays)\b", r"\b(?:starred|starring|stars|played|plays|portrayed)\b", ()),
    (r"\b(?:singer|perform\w*|record\w*)\b", r"\b(?:singer|performed by|released by|recorded by|single by)\b",
     ("performed_by", "released_by")),
    (r"\b(?:acqui\w*|bought)\b", r"\b(?:acqui\w*|bought|purchased)\b", ()),
    (r"\b(?:located|contains|includes|displayed)\b", r"\b(?:located in|situated in|contains|includes|displayed at)\b",
     ("located_in",)),
    (r"\b(?:born|birth)\b", r"\b(?:born in|birthplace)\b", ("located_in",)),
    (r"\b(?:operat\w*)\b", r"\b(?:operated by|operator)\b", ()),
    (r"\b(?:of the movie|in with)\b", r"\b(?:starred|starring|stars|played|appeared in)\b", ()),
    (r"\b(?:preceded|predecessor|previous)\b", r"\b(?:previous|preceded|predecessor|prior)\b", ()),
)


def relation_for(question: str) -> Relation | None:
    matches = [relation for relation in RELATIONS if re.search(relation.question, question, re.I)]
    if matches and not matches[0].evidence:
        return None
    return matches[0] if len(matches) == 1 else None


def verified_facts(doc: dict, compiled: dict) -> list[dict]:
    row = compiled.get(doc["id"], {})
    if row.get("source_version") != doc["version"]:
        return []
    return [fact for fact in row.get("llm", []) if fact["proof"] and all(
        0 <= item["index"] < len(doc["spans"]) and span_text(doc, item["index"]) == item["text"]
        for item in fact["proof"]
    )]


def aliases(docs: list[dict], compiled: dict) -> list[tuple[int, str, set[int]]]:
    entries = []
    for page, doc in enumerate(docs):
        for fact in verified_facts(doc, compiled):
            if fact["predicate"] == "alias" and len(fact["object"]) >= 4:
                entries.append((page, accent_fold(fact["object"]), {item["index"] for item in fact["proof"]}))
    return entries


def bound_spans(doc: dict, index: int) -> set[int]:
    try:
        anchor, _ = binding(doc, index, None)
    except ValueError:
        anchors = subject_anchors(doc)
        title = fold(title_surface(doc["title"]))
        if index == 0 and title and f"{fold(span_text(doc, index))} ".startswith(f"{title} "):
            return {index}
        # A sentence splitter can separate an abbreviation from its continuation.
        # Carry the preceding fragment only when it refers to the page subject.
        if index >= 2 and index - 2 in anchors:
            prior = span_text(doc, index - 1).strip()
            lead = re.match(r"^\s*The ((?:\w+ ){0,2}\w+)\s+(?:is|was|lies|encompasses|includes|contains)\b",
                            prior, re.I)
            if lead and title.endswith(" " + fold(lead[1])) and re.search(r"\b(?:[A-Z][a-z]?|Jr|Sr)\.$", prior):
                return {index - 2, index - 1, index}
        return set()
    return {index} if anchor is None else {anchor, index}


def terminal_proof(doc: dict, pattern: re.Pattern) -> list[set[int]]:
    matches = []
    for index in range(len(doc["spans"])):
        if not pattern.search(span_text(doc, index)):
            continue
        proof = bound_spans(doc, index)
        if proof:
            matches.append(proof)
    return matches


def find_paths(question: str, docs: list[dict], compiled: dict) -> tuple[str, list[dict]]:
    relation = relation_for(question)
    if relation is None:
        return "unsupported_relation", []
    titles = {doc["id"]: title_surface(doc["title"]) for doc in docs}
    named = resolve_mentions(question, question, docs, titles)
    if len(named) != 1:
        return "ambiguous_or_missing_question_entity", []
    source = next(iter(named))
    terminals = {i: terminal_proof(doc, re.compile(relation.evidence, re.I)) for i, doc in enumerate(docs)}
    paths = []
    bridge_context = re.search(r"\b(?:of the|that|whose|from the|which was|who was|who played|preceded)\b", question, re.I)
    if len(terminals[source]) == 1 and not bridge_context:
        paths.append({"spans": {(source, j) for j in terminals[source][0]}, "wiki_alias": False})
    source_cues = [(cue, predicates) for q, cue, predicates in SOURCE_RELATIONS if re.search(q, question, re.I)]
    if not source_cues:
        return relation.name, paths
    wiki_aliases = aliases(docs, compiled)
    verified = verified_facts(docs[source], compiled)
    source_predicates = {p for _, predicates in source_cues for p in predicates}
    for index in range(len(docs[source]["spans"])):
        text = span_text(docs[source], index)
        if any(re.search(cue, text, re.I) for cue, _ in source_cues):
            origin = bound_spans(docs[source], index)
            if origin:
                paths.extend(linked_paths(question, docs, titles, source, index, origin, text, terminals, wiki_aliases))
        for fact in verified:
            if fact["sentence_id"] == index and fact["predicate"] in source_predicates:
                proof = {item["index"] for item in fact["proof"]}
                paths.extend(linked_paths(question, docs, titles, source, index, proof, fact["object"], terminals,
                                          wiki_aliases, from_wiki=True))
    return relation.name, paths


def linked_paths(question: str, docs: list[dict], titles: dict, source: int, index: int, origin: set[int],
                 text: str, terminals: dict, wiki_aliases: list, *, from_wiki: bool = False) -> list[dict]:
    targets = defaultdict(list)
    for page in resolve_mentions(text, f"{question} {text}", docs, titles) - {source}:
        targets[page].append((titles[docs[page]["id"]], set()))
    normalized_titles = [fold(titles[doc["id"]]) for doc in docs]
    for page, doc in enumerate(docs):
        entity = titles[doc["id"]]
        normalized = normalized_titles[page]
        if (page != source and normalized_titles.count(normalized) == 1 and len(normalized.split()) >= 2 and
                literal(normalized).search(fold(text)) and not literal(entity.casefold()).search(accent_fold(text).casefold())):
            targets[page].append((entity, set()))
    for page, alias, proof in wiki_aliases:
        if page != source and literal(alias.casefold()).search(accent_fold(text).casefold()):
            targets[page].append((alias, proof))
    paths = []
    for page, links in targets.items():
        if len(terminals[page]) != 1:
            continue
        for entity, alias_proof in links:
            terminal = terminals[page][0]
            target_span = next((j for j in sorted(terminal | alias_proof)
                                if literal(entity.casefold()).search(accent_fold(span_text(docs[page], j)).casefold())),
                               min(terminal))
            ends = {(source, index), (page, target_span)}
            try:
                validate_join({"left": [source, index], "right": [page, target_span], "entity": entity}, ends, docs)
            except ValueError as error:
                # Punctuation alone must not turn a source-bound mention into a different entity.
                normalized = fold(entity)
                fragments = [fold(docs[d]["title"] + " " + span_text(docs[d], s)) for d, s in sorted(ends)]
                if not (str(error) == "unbound_join_entity" and len(normalized.split()) >= 2
                        and all(literal(normalized).search(f) for f in fragments)
                        and not literal(entity.casefold()).search(accent_fold(span_text(docs[source], index)).casefold())):
                    continue
                regions = [state_qualifiers(accent_fold(docs[d]["title"]).casefold(),
                                            fold(span_text(docs[d], s)), normalized) for d, s in sorted(ends)]
                if all(regions) and regions[0].isdisjoint(regions[1]):
                    continue
            qualifiers = [versions(docs[doc], (doc, span), entity) for doc, span in sorted(ends)]
            if all(qualifiers) and qualifiers[0].isdisjoint(qualifiers[1]):
                continue
            paths.append({"spans": {(source, j) for j in origin} | {(page, j) for j in terminal | alias_proof},
                          "wiki_alias": from_wiki or bool(alias_proof)})
    return paths


def choose_pack(tokenizer, docs: list[dict], budget: int, full: dict, prediction: tuple[str, list[dict]],
                original: dict) -> tuple[dict, str, list[list[int]]]:
    if full["evidence_tokens"] <= budget:
        return full, "already_short", []
    reason, candidates = prediction
    unique = {frozenset(path["spans"]): path for path in sorted(candidates, key=lambda path: path["wiki_alias"],
                                                              reverse=True)}
    if len(unique) != 1:
        return full, "ambiguous_path" if unique else reason if reason.startswith(("unsupported", "ambiguous")) else "no_bound_path", []
    path = next(iter(unique.values()))
    selected = defaultdict(set)
    for page, span in path["spans"]:
        selected[page].add(span)
    text = sentence_blocks(docs, selected)
    count = len(tokenizer.tokenize(text))
    if count > budget:
        return full, "path_over_budget", []
    # Keep the old short context when it already preserves the checked path.
    spans = {(page, j) for page, ids in original["selected_blocks"] for j in ids}
    if path["spans"] <= spans:
        return original, "checked_existing_pack", sorted(map(list, path["spans"]))
    return {**original, "evidence": text, "evidence_tokens": count,
            "selected_blocks": sorted((i, sorted(ids)) for i, ids in selected.items())}, (
                "checked_wiki_alias_pack" if path["wiki_alias"] else "checked_rebuilt_pack"
            ), sorted(map(list, path["spans"]))
