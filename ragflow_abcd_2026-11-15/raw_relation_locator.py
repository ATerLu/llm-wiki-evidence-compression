"""Locate typed evidence directly in the frozen candidate text."""

from dataclasses import dataclass
import re

from evidence_join_recovery import versions
from intent_evidence import validate_join
from relation_gate import bound_spans, find_paths, terminal_proof, verified_facts
from wiki_bridge import accent_fold, literal, span_text, title_surface
from wiki_compile_pilot import DATE
from wiki_compile_v2 import fold, subject_anchors


@dataclass(frozen=True)
class LinkRule:
    name: str
    question: str
    source: str
    target: str
    wiki_predicate: str = ""


PLACE = r"(?:County|District|Prefecture|Province|Department)"
LOCATED = (r"\b(?:(?:is|was) (?:an? |the )?(?:[\w-]+ ){0,5}(?:stadium|museum|city|village|town|gmina|exhibit|facility|building) "
           r"(?:in|within)|(?:located|situated|based|lies) (?:in|within)|"
           r"(?:shopping|retail) (?:mall|center|centre) (?:in|within))\b")
COUNTY_FACT = rf"\b(?:county seat of|is in|city in|town in|in|within|part of)\s+(?:,\s*)?(?:and the county seat of,?\s+)?(?:[\w'-]+[ ,]+){{1,6}}{PLACE}\b"
RULES = (
    LinkRule("county_location", r"\b(?:what|which) county\b(?!.*\b(?:same|other|shares?|border|near)\b)",
             LOCATED + r"|\blicensed to\b", COUNTY_FACT, "located_in"),
    LinkRule("exhibit_location", r"\bwhat city\b.*\b(?:displayed|exhibited|housed)\b",
             r"\b(?:located in|displayed at|exhibited at|housed at|on display at)\b",
             r"\b(?:is|was)\b.{0,90}\b(?:museum|gallery|art center|collection)\b.{0,40}\bin\s+[^.;]+", "located_in"),
    LinkRule("performer_member", r"\bwho\b.*\b(?:member|members)\b.*\b(?:performed|recorded|sang)\b",
             r"\b(?:recorded by|performed by|sung by|single by|song by)\b",
             r"\b(?:composed of|consists? of|consisted of|members (?:are|were|include)|duo of)\b", "performed_by"),
    LinkRule("release_formation", r"\b(?:album|record|single)\b.*\b(?:formed|founded) by who\b",
             r"\b(?:debut album of|album by|album of|recorded by|single by)\b",
             r"\b(?:formed|founded)\s*(?:in\s+\d{4}\s+)?by\b"),
    LinkRule("role_premiere", r"\b(?:plays?|played)\b.*\b(?:premiered|first aired)\b",
             r"\b(?:role as|plays?|played|portrayed)\b",
             rf"\b(?:premiered|first aired|debuted)\s+(?:on\s+|in\s+)?{DATE}\b"),
    LinkRule("series_predecessor", r"\b(?:first air date|when)\b.*\b(?:preceded|predecessor)\b",
             r"\b(?:previous|preceding|earlier)\b",
             rf"\b(?:premiered|first aired|debuted|began with.{{0,45}}airing)\s+(?:on\s+|in\s+)?{DATE}\b"),
    LinkRule("casting_award", r"\b(?:actor|actress)\b.*\bplayed\b.*\bwon\b.*\baward\b",
             r"\b(?:starring|starred|cast|featured)\b",
             r"\b(?:won|winning|received)\b.{0,24}\b(?:Academy Award|Oscar|Tony Award)\b.{0,130}\b(?:in|for)\b"),
    LinkRule("compilation_induction", r"\bwhat band\b.*\bcompiled\b.*\binducted\b",
             r"\b(?:compiled by|compiled the track)\b",
             r"\binducted into\b.{0,85}\bas (?:a )?member of\b"),
    LinkRule("river_chain", r"\bwhat river\b.*\b(?:mouth|flows|tributary)\b",
             r"\b(?:tributary of|flows into)\b", r"\btributary of\s+(?:the\s+)?[\w -]+\bRiver\b"),
)


def titles_mentioned(text: str, titles: list[str]) -> set[int]:
    folded = fold(text)
    hits = [(i, m.start(), m.end()) for i, title in enumerate(titles)
            for m in literal(title).finditer(folded) if len(title) >= 4]
    # A nested title such as Boston inside FleetBoston Financial is not another entity.
    return {i for i, start, end in hits if not any(
        other_start <= start and end <= other_end and (start, end) != (other_start, other_end)
        for _, other_start, other_end in hits
    )}


def question_pages(question: str, titles: list[str]) -> set[int]:
    named = titles_mentioned(question, titles)
    # A bare nested work title is weaker than the explicitly punctuated title.
    punctuated = {i for i in named if re.search(rf"{literal(titles[i]).pattern}!", accent_fold(question).casefold())}
    return punctuated or named


def proof_spans(doc: dict, index: int) -> set[int]:
    spans = bound_spans(doc, index)
    if not spans and index == 0 and re.search(r'"[^".]{2,20}"', span_text(doc, index)):
        name = fold(title_surface(doc["title"])).split()
        lead = fold(span_text(doc, index)).split()[:8]
        if len(name) == 2 and lead and lead[0].startswith(name[0]) and name[0] in lead and name[1] in lead:
            return {index}
    if spans or index != 1:
        return spans
    lead = span_text(doc, index).strip()
    if fold(span_text(doc, 0)) == fold(title_surface(doc["title"])) and re.match(r"^(?:is|was|are|were|has|had)\b", lead, re.I):
        return {0, index}
    return set()


def terminal_spans(doc: dict, pattern: re.Pattern) -> list[set[int]]:
    proofs = terminal_proof(doc, pattern)
    title = fold(title_surface(doc["title"]))
    for j in range(1, len(doc["spans"])):
        text = span_text(doc, j)
        if any(j in proof for proof in proofs) or not pattern.search(text):
            continue
        tail = title.split()[-1:] or [""]
        head = re.match(r"\s*(?:The (?:show|series|district|city|album|film)\b|[\w'’-]+\b)", text, re.I)
        anchors = subject_anchors(doc)
        previous = max((i for i in anchors if 0 < j - i <= 3), default=None)
        # A descriptive reference or surname can bind to an immediately preceding named lead.
        if (previous is not None and head and (fold(head[0]).startswith("the ") or fold(head[0]) == tail[0])
                and (previous == j - 1 or fold(head[0]) == tail[0])):
            proofs.append({previous, j})
    return proofs


def source_objects(docs: list[dict], i: int, j: int, rule: LinkRule, compiled: dict) -> list[tuple[str, set[int], bool]]:
    doc = docs[i]
    sentence = span_text(doc, j)
    pattern = re.compile(rule.source, re.I)
    match = pattern.search(sentence)
    bound = proof_spans(doc, j)
    tail = re.split(r"[;]|\b(?:serving|which|whereas|while)\b", sentence[match.end():], maxsplit=1)[0] if match else ""
    found = [(tail, bound, False)] if tail.strip() and bound else []
    if rule.wiki_predicate:
        found.extend((f["object"], {p["index"] for p in f["proof"]}, True) for f in verified_facts(doc, compiled)
                     if f["sentence_id"] == j and f["predicate"] == rule.wiki_predicate)
    return found


def compatible(docs: list[dict], origin: tuple[int, int], target: tuple[int, int]) -> bool:
    source, j = origin
    page, k = target
    name = title_surface(docs[page]["title"])
    try:
        validate_join({"left": list(origin), "right": list(target), "entity": name}, {origin, target}, docs)
    except ValueError:
        return False
    region = re.search(r",\s*(?:state of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$", name)
    if region:
        qualified = re.search(rf"{literal(name[:region.start()]).pattern},\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
                              span_text(docs[source], j))
        if qualified and fold(qualified[1]) != fold(region[1]):
            return False
    scopes = [versions(docs[d], (d, s), name) for d, s in (origin, target)]
    return not all(scopes) or not scopes[0].isdisjoint(scopes[1])


def explicit_paths(question: str, docs: list[dict], rule: LinkRule, compiled: dict) -> tuple[str, list[dict]]:
    titles = [fold(title_surface(doc["title"])) for doc in docs]
    named = question_pages(question, titles)
    if rule.name == "compilation_induction":
        named = {i for i in named if any(re.search(rule.source, span_text(docs[i], j), re.I)
                                        for j in range(len(docs[i]["spans"])))}
    if len(named) != 1:
        return "ambiguous_source", []
    source = next(iter(named))
    targets = re.compile(rule.target, re.I)
    terminals = {i: terminal_spans(doc, targets) for i, doc in enumerate(docs) if i != source}
    if rule.name == "county_location":
        county_query = r"(?:,\s+and\s+the\s+county seat of,|county seat of)\s+(?:[\w'-]+\s+){1,6}County\b"
        terminals = {i: paths or terminal_spans(docs[i], re.compile(county_query, re.I)) for i, paths in terminals.items()}
    found = {}
    for j in range(len(docs[source]["spans"])):
        for text, proof, wiki in source_objects(docs, source, j, rule, compiled):
            for page in titles_mentioned(text, titles) - {source}:
                if titles.count(titles[page]) != 1 or len(terminals[page]) != 1:
                    continue
                end = terminals[page][0]
                if not any(compatible(docs, (source, j), (page, k)) for k in end):
                    continue
                result = frozenset({(source, k) for k in proof} | {(page, k) for k in end})
                found.setdefault(result, {"spans": result, "wiki_alias": wiki})
    return rule.name, list(found.values())


def locate(question: str, docs: list[dict], compiled: dict) -> tuple[str, list[dict]]:
    for rule in RULES:
        if re.search(rule.question, question, re.I):
            typed = explicit_paths(question, docs, rule, compiled)
            if typed[1] or rule.name == "county_location":
                return typed
    return find_paths(question, docs, compiled)
