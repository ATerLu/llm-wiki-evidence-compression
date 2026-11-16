"""Conservative local routing from an original two-sentence evidence chain."""

from collections import defaultdict
import re

from baseline import words
from wiki_bridge import span_text
from wiki_compile_v2 import fold

SOURCE_LIMIT = 5
TARGET_LIMIT = 3
ENTITY = re.compile(r"\b[A-Z][\w'’-]+(?:\s+(?:[A-Z][\w'’-]+|of|the|and|[IVX]+)){0,4}\b")
SKIP = {'the','this','in','during','after','before','it','he','she','when','a','an','according','however'}
FACETS = (
    (r'^\s*(?:(?:in|within)\s+)?(?:which|what)\s+(?:administrative\s+)?(?:province|county)\b',
     r'\b(?:located|lies|situated|in|within)\b.{0,100}\b(?:province|county)\b'),
    (r'^\s*how many\s+(?:groups|alliances)\b', r'\b(?:two|three|four|five|\d+)\s+(?:groups|alliances)\b'),
    (r'^\s*who\s+(?:is|was)\s+the\s+permanent representative\b', r'\b(?:permanent representative|incumbent)\b'),
    (r'^\s*(?:what|which)\s+episode\b.{0,90}\b(?:married|wedding)\b', r'\b(?:married|marriage|wedding)\b'),
    (r'^\s*(?:in\s+)?(?:what|which)\s+(?:country|city)\b.{0,90}\b(?:born|birth)\b', r'\b(?:was born|birthplace)\b'),
    (r'^\s*(?:who|which person)\s+(?:directed|(?:was|is) the director)\b', r'\bdirected\s+by\b'),
    (r'^\s*(?:who|which person)\s+(?:wrote|(?:was|is) the author)\b', r'\b(?:written|authored)\s+by\b'),
    (r'^\s*(?:who|which person)\s+(?:created|(?:was|is) the creator)\b', r'\bcreated\s+by\b'),
    (r'^\s*(?:when|(?:in\s+)?what year)\s+(?:was|were)\b.{0,75}\b(?:founded|established|incorporated)\b',
     r'\b(?:founded|established|incorporated)\b.{0,40}\b(?:1[0-9]{3}|20[0-9]{2})\b'),
)


def cue_for(question: str) -> re.Pattern | None:
    return next((re.compile(cue, re.I) for condition, cue in FACETS if re.search(condition, question, re.I)), None)


def anchor_words(question: str, text: str) -> set[str]:
    return {w for w in words(fold(question)) & words(fold(text)) if len(w) >= 4}


def has_anchor(question: str, text: str) -> bool:
    q, body = fold(question).split(), f' {fold(text)} '
    for width in (2, 3):
        for i in range(len(q)-width+1):
            part = q[i:i+width]
            if (sum(len(w)>=4 and w not in SKIP for w in part)>=2
                    and f" {' '.join(part)} " in body):
                return True
    return False


def entities(text: str) -> set[str]:
    names = set()
    for match in ENTITY.finditer(text):
        tokens = fold(match.group()).split()
        while len(tokens) > 1 and tokens[0] in SKIP:
            tokens.pop(0)
        value = ' '.join(tokens)
        if len(value) >= 4 and value not in SKIP:
            names.add(value)
    return names


def sources_from_view(question: str, docs: list[dict], records: list[dict]) -> list[tuple[int, int]]:
    versions = defaultdict(list)
    for i, doc in enumerate(docs):
        versions[doc['version']].append(i)
    picked = set()
    for record in records:
        if not has_anchor(question, record['question']):
            continue
        matches = versions.get(record['version'], [])
        if len(matches) != 1:
            continue
        i = matches[0]
        j = record['span']
        if 0 <= j < len(docs[i]['spans']) and span_text(docs[i], j) == record['text']:
            picked.add((i, j))
    return sorted(picked)


def choose_chain(question: str, docs: list[dict], records: list[dict]) -> dict:
    cue = cue_for(question)
    if cue is None:
        return {'reason':'unsupported_question_type', 'view_hits':0}
    spans = {(i, j):span_text(doc, j) for i, doc in enumerate(docs) for j in range(len(doc['spans']))}
    hits = set(sources_from_view(question, docs, records))
    anchored = {p:text for p, text in spans.items() if has_anchor(question, f'{docs[p[0]]["title"]} {text}')}
    source = sorted(anchored, key=lambda p: (p not in hits, -len(anchor_words(question, anchored[p])), p))[:SOURCE_LIMIT]
    targets = [p for p, text in spans.items() if cue.search(text)]
    chains = defaultdict(set)
    normalized = f' {fold(question)} '
    for i, j in source:
        for value in entities(spans[i, j]):
            if f' {value} ' in normalized:
                continue
            connected = [(k, l) for k, l in targets if k != i
                         and f' {value} ' in f' {fold(docs[k]["title"] + " " + spans[k,l])} ']
            if len(connected) > TARGET_LIMIT:
                continue
            for target in connected:
                # Prefer an explicit named target in the question when the wording names one.
                if any(k != i and has_anchor(question, docs[k]['title']) for k, _ in connected):
                    if not has_anchor(question, docs[target[0]]['title']):
                        continue
                chains[(value, target)].add((i,j))
    if len(chains) != 1:
        return {'reason':'no_unique_chain', 'view_hits':len(hits), 'chains':len(chains)}
    (value, target), sources = next(iter(chains.items()))
    first = min(sources, key=source.index)
    return {'reason':'short', 'view_hits':len(hits), 'used_view':first in hits, 'chain_value':value,
            'spans':[first, target]}
