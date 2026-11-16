"""Validate proposed original spans and joins before budgeted evidence assembly."""

import re

from semantic_bridge import Bridge, Span
from wiki_bridge import accent_fold, literal, span_text


MAX_PROOF_SPANS = 8
STATES = (
    "alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|"
    "indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|"
    "montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|"
    "oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|"
    "virginia|washington|west virginia|wisconsin|wyoming"
)
TITLE_STATE = re.compile(rf"(?:,\s*|\()({STATES})(?=\)|,|$)")


def state_qualifiers(title: str, sentence: str, entity: str) -> set[str]:
    qualified = re.compile(rf"{literal(entity).pattern}(?:\s+(?:borough|township|city|village))?\s*[,\(]\s*"
                           rf"({STATES})(?!\w)")
    explicit = set(qualified.findall(sentence))
    return explicit or set(TITLE_STATE.findall(title))


def validate_join(join: object, spans: set[Span], docs: list[dict], *, check_place_scope: bool = True) -> dict:
    if not isinstance(join, dict) or not isinstance(join.get("entity"), str):
        raise ValueError("invalid_join")
    left, right = (parse_span(join.get(side), docs) for side in ("left", "right"))
    if left not in spans or right not in spans or left[0] == right[0]:
        raise ValueError("invalid_join_endpoints")
    name = join["entity"].strip()
    entity = accent_fold(name).casefold()
    if len(entity) < 3 or not any(char.isupper() for char in name):
        raise ValueError("invalid_join_entity")
    regions = []
    for page, index in (left, right):
        title = accent_fold(docs[page]["title"]).casefold()
        text = accent_fold(span_text(docs[page], index)).casefold()
        if not literal(entity).search(f"{title}\n{text}"):
            raise ValueError("unbound_join_entity")
        regions.append(state_qualifiers(title, text, entity))
    if check_place_scope and all(regions) and regions[0].isdisjoint(regions[1]):
        raise ValueError("conflicting_place_qualifiers")
    return {"left": left, "right": right, "entity": name, "states": [sorted(group) for group in regions]}


def parse_span(value: object, docs: list[dict]) -> Span:
    if not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError("invalid_span")
    page, index = value
    if not 0 <= page < len(docs) or not 0 <= index < len(docs[page]["spans"]):
        raise ValueError("span_outside_candidate_pool")
    return page, index


def validate_proof(proposal: object, docs: list[dict], *, check_place_scope: bool = True) -> tuple[Span, Bridge] | None:
    if not isinstance(proposal, dict) or proposal.get("status") not in ("supported", "abstain"):
        raise ValueError("invalid_proposal")
    spans, joins = proposal.get("spans"), proposal.get("joins")
    if not isinstance(spans, list) or not isinstance(joins, list):
        raise ValueError("invalid_proof_fields")
    if proposal["status"] == "abstain":
        if spans or joins:
            raise ValueError("nonempty_abstention")
        return None
    if not 1 <= len(spans) <= MAX_PROOF_SPANS or len(joins) >= len(spans):
        raise ValueError("invalid_proof_size")
    ordered = tuple(parse_span(span, docs) for span in spans)
    selected = set(ordered)
    if len(selected) != len(ordered):
        raise ValueError("duplicate_proof_span")

    verified = tuple(validate_join(join, selected, docs, check_place_scope=check_place_scope) for join in joins)
    connected = {ordered[0][0]}
    for _ in verified:
        for link in verified:
            pages = {link["left"][0], link["right"][0]}
            if connected & pages:
                connected.update(pages)
    if connected != {page for page, _ in ordered}:
        raise ValueError("disconnected_proof")
    return ordered[0], Bridge(ordered[-1][0], ordered, verified)
