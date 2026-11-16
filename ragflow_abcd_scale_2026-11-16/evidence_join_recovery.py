"""Conservatively recover one missing cross-page connection from selected original sentences."""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import re

from intent_evidence import parse_span, validate_join, validate_proof
from semantic_bridge import Bridge, ENTITY_RELATIONS, Span
from wiki_bridge import accent_fold, literal, span_text, title_surface
from wiki_compile_v2 import work_author_phrase


NAME_WORD = r"[A-Z][^\W\d_]*(?:[-'’][A-Za-z]+)*\.?"
PERSON = re.compile(rf"(?<!\w)({NAME_WORD}(?:\s+(?:(?:de|van|von|da|of)\s+)?{NAME_WORD}){{1,4}})(?!\w)")
PLACE = re.compile(rf"\b(?:in|within|of)\s+((?:{NAME_WORD}\s+){{1,5}}(?:County|Province|District|Prefecture))\b")
PLACE_CUE = re.compile(r"\b(?:located|situated|based|born|lies|resides|resided|headquartered|part of)\s+(?:in\s+)?$", re.I)
SCHOOL_CUE = re.compile(r"\b(?:attended|studied at|graduated from|enrolled at)\s+(?:the\s+)?$", re.I)
VERSION = r"(?:version\s+|v\s*)([0-9]+(?:\.[0-9]+){1,3})(?!\w|\.\d)"


@dataclass(frozen=True)
class Connection:
    left: Span
    right: Span
    entity: str
    relation: str

    def as_join(self) -> dict:
        return {"left": list(self.left), "right": list(self.right), "entity": self.entity}


@dataclass(frozen=True)
class Decision:
    path: tuple[Span, Bridge] | None
    status: str
    connection: Connection | None = None


def normalize(text: str) -> str:
    return " ".join(accent_fold(text).casefold().split())


def textual_roles(sentence: str) -> dict[tuple[str, str], str]:
    objects = {("located_in", normalize(m[1])): m[1] for m in PLACE.finditer(sentence)}
    for match in PERSON.finditer(sentence):
        name = match[1].rstrip(".")
        if work_author_phrase(sentence, name):
            objects["created_by", normalize(name)] = name
    return objects


def title_relation(sentence: str, title: str) -> str | None:
    # The original title may disambiguate a mention, but it does not establish the source relation.
    for match in literal(normalize(title)).finditer(accent_fold(sentence).casefold()):
        prefix = accent_fold(sentence).casefold()[:match.start()]
        if PLACE_CUE.search(prefix):
            return "located_in"
        if SCHOOL_CUE.search(prefix):
            return "educated_at"
    return None


def possible_connections(ordered: list[Span], docs: list[dict], wiki_facts: list[list[dict]] | None) -> list[Connection]:
    titles = [title_surface(doc["title"]) for doc in docs]
    title_counts = Counter(map(normalize, titles))
    selected = set(ordered)
    text = {span: span_text(docs[span[0]], span[1]) for span in ordered}
    roles = {span: textual_roles(text[span]) if wiki_facts is None else compiled_roles(span, selected, wiki_facts)
             for span in ordered}
    connections = []
    for left, right in combinations(ordered, 2):
        if left[0] == right[0]:
            continue
        for relation, key in sorted(roles[left].keys() & roles[right].keys()):
            connections.append(Connection(left, right, roles[left][relation, key], relation))
        for origin, target in ((left, right), (right, left)):
            title = titles[target[0]]
            key = normalize(title)
            if len(key) < 4 or title_counts[key] != 1 or not any(c.isupper() for c in title):
                continue
            relations = ([relation for relation, obj in roles[origin] if obj == key] if wiki_facts is not None
                         else [title_relation(text[origin], title)])
            connections.extend(Connection(origin, target, title, relation) for relation in relations if relation)
    return connections


def compiled_roles(span: Span, selected: set[Span], wiki_facts: list[list[dict]]) -> dict[tuple[str, str], str]:
    page, index = span
    objects = {}
    for fact in wiki_facts[page]:
        if fact["sentence_id"] != index or fact["predicate"] not in ENTITY_RELATIONS:
            continue
        if not all((page, proof["index"]) in selected for proof in fact["proof"]):
            continue
        name = fact["object"].strip()
        if len(name) >= 4 and any(c.isupper() for c in name):
            objects[fact["predicate"], normalize(name)] = name
    return objects


def versions(doc: dict, span: Span, entity: str) -> set[str]:
    name = literal(normalize(entity)).pattern
    pattern = re.compile(rf"{name}\s*(?:[,(/-]\s*)?{VERSION}")
    text = normalize(span_text(doc, span[1]))
    return set(pattern.findall(text)) or set(pattern.findall(normalize(doc["title"])))


def choose_connection(proposal: dict, docs: list[dict], ordered: list[Span],
                      wiki_facts: list[list[dict]] | None = None) -> Decision:
    valid, conflicts = {}, set()
    for candidate in possible_connections(ordered, docs, wiki_facts):
        join = candidate.as_join()
        try:
            validate_join(join, set(ordered), docs)
            left, right = [versions(docs[s[0]], s, candidate.entity) for s in (candidate.left, candidate.right)]
            if left and right and left.isdisjoint(right):
                raise ValueError("conflicting_version_qualifiers")
            path = validate_proof({**proposal, "joins": [*proposal["joins"], join]}, docs)
        except ValueError as error:
            if str(error) in ("conflicting_place_qualifiers", "conflicting_version_qualifiers"):
                conflicts.add(str(error))
            continue
        key = (candidate.left[0], candidate.right[0], normalize(candidate.entity), candidate.relation)
        valid.setdefault(key, (candidate, path))
    if len(valid) > 1 or (valid and conflicts):
        return Decision(None, "ambiguous_connection")
    if not valid:
        return Decision(None, "+".join(sorted(conflicts)) if conflicts else "no_grounded_connection")
    candidate, path = next(iter(valid.values()))
    return Decision(path, "recovered_wiki" if wiki_facts is not None else "recovered_raw", candidate)


def recover(row: dict, docs: list[dict], wiki_facts: list[list[dict]]) -> tuple[Decision, Decision, Decision]:
    if row["status"] != "ok":
        result = Decision(None, row["status"])
        return result, result, result
    proposal = row["proposal"]
    try:
        path = validate_proof(proposal, docs)
        result = Decision(path, "validated" if path else "model_abstained")
        return result, result, result
    except ValueError as error:
        result = Decision(None, str(error))
        if result.status != "disconnected_proof":
            return result, result, result
    ordered = [parse_span(value, docs) for value in proposal["spans"]]
    raw = choose_connection(proposal, docs, ordered)
    augmented = choose_connection(proposal, docs, ordered, wiki_facts) if raw.status == "no_grounded_connection" else raw
    return result, raw, augmented
