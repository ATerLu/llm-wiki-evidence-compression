"""Source-grounded Wiki alias entries and a budgeted dependency packer."""

from dataclasses import astuple, dataclass
from pathlib import Path
import re
import sqlite3

from baseline import sentence_anchors, sentence_blocks

MARKER = r"also known as|formerly known as|previously known as|formerly called|renamed to"


@dataclass(frozen=True)
class Alias:
    doc_id: str
    version: str
    sentence_index: int
    start: int
    end: int
    canonical: str
    alias: str
    marker: str


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold())).removeprefix("the ")


def contains(text: str, term: str) -> bool:
    return bool(term) and f" {term} " in f" {text} "


def extract_aliases(doc: dict) -> list[Alias]:
    canonical = re.sub(r"\s*\([^()]+\)$", "", doc["title"]).strip()
    if len(normalize(canonical).split()) < 2:
        return []
    pattern = re.compile(
        rf"^\s*{re.escape(canonical)}(?:\s*\([^()]{{0,35}}\))?[\s,(;–-]+"
        rf"(?:(?:is|was)\s+)?(?P<marker>{MARKER})\s+(?P<tail>.+)",
        re.IGNORECASE,
    )
    entries = []
    for span in doc["spans"]:
        sentence = doc["text"][span["start"]:span["end"]]
        match = pattern.match(sentence)
        if not match:
            continue
        tail = match["tail"].strip()
        quoted = re.match(r'["“]([^"”]+)["”]', tail)
        alias = quoted[1] if quoted else re.split(r"[,;().]|\s+(?:is|was|are|were)\s+", tail, maxsplit=1)[0]
        alias = alias.strip(" \t'’")
        normalized = normalize(alias)
        if (not 2 <= len(normalized.split()) <= 10 or normalized == normalize(canonical)
                or re.search(r"\b(in|outside|during)\b", alias, re.IGNORECASE)
                or not alias[0].isupper() or re.search(r"\b(?:and|or)\b", alias, re.IGNORECASE)):
            continue
        entries.append(Alias(doc["id"], doc["version"], span["index"], span["start"], span["end"],
                             canonical, alias, match["marker"].lower()))
    return entries


def compile_wiki(corpus: dict, path: Path | None = None) -> dict[str, list[Alias]]:
    pages = {}
    for doc_id, doc in corpus.items():
        entries = extract_aliases(doc)
        if entries:
            pages[doc_id] = entries
    if path is not None:
        with sqlite3.connect(path) as database:
            database.execute(
                "CREATE TABLE alias_entries (doc_id TEXT, version TEXT, sentence_index INTEGER, "
                "start INTEGER, end INTEGER, canonical TEXT, alias TEXT, marker TEXT, "
                "PRIMARY KEY (doc_id, sentence_index))"
            )
            database.executemany("INSERT INTO alias_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                 (astuple(entry) for entries in pages.values() for entry in entries))
    return pages


def pack_dependencies(tokenizer, question: str, docs: list[dict], wiki: dict[str, list[Alias]],
                      budget: int) -> tuple[dict[int, set[int]], str, int, list[dict]]:
    question_text = normalize(question)
    relevant = []
    for proof_index, doc in enumerate(docs):
        for entry in wiki.get(doc["id"], []):
            canonical, alias = normalize(entry.canonical), normalize(entry.alias)
            if contains(question_text, alias):
                relevant.append((proof_index, entry, canonical, True))
            elif contains(question_text, canonical):
                relevant.append((proof_index, entry, alias, False))
    selected, evidence, count, events = {}, "", 0, []
    for doc_index, anchor in sentence_anchors(question, docs):
        if anchor in selected.get(doc_index, set()):
            continue
        doc = docs[doc_index]
        span = doc["spans"][anchor]
        text = normalize(doc["text"][span["start"]:span["end"]])
        dependencies = [
            (proof_index, entry) for proof_index, entry, other_name, alias_query in relevant
            if (proof_index, entry.sentence_index) != (doc_index, anchor)
            and (contains(text, other_name) or (alias_query and proof_index == doc_index))
            and entry.sentence_index not in selected.get(proof_index, set())
        ]
        proposed = {**selected, doc_index: selected.get(doc_index, set()) | {anchor}}
        for proof_index, entry in dependencies:
            proposed[proof_index] = proposed.get(proof_index, set()) | {entry.sentence_index}
        candidate = sentence_blocks(docs, proposed)
        candidate_count = len(tokenizer.tokenize(candidate))
        accepted = candidate_count <= budget
        for proof_index, entry in dependencies:
            events.append({
                "anchor": [doc_index, anchor], "proof": [proof_index, entry.sentence_index],
                "doc_id": entry.doc_id, "version": entry.version, "start": entry.start, "end": entry.end,
                "canonical": entry.canonical, "alias": entry.alias, "marker": entry.marker,
                "decision": "added" if accepted else "over_budget",
            })
        if accepted:
            selected, evidence, count = proposed, candidate, candidate_count
    return selected, evidence, count, events
