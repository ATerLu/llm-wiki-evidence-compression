"""Source-bound subject and relation validation shared by both compilers."""

import re

from wiki import normalize
from wiki_bridge import accent_fold, span_text, title_surface
from wiki_compile_pilot import DATE, KINDS, MAX_FACTS, PATTERNS, key

ANAPHOR = re.compile(r"^\s*(?:he|she|it|they|the (?:river|company|town|film|album|band|university|city))\b", re.I)
PERSON_LEAD = re.compile(r"^\s*(?:was|is|born|played|attended|studied|graduated|served|worked|died|became)\b", re.I)
LANGUAGES = set("english spanish french german italian portuguese russian japanese korean chinese arabic "
                "dutch latin greek catalan hindi bengali mandarin welsh gaelic swedish danish norwegian "
                "finnish hungarian polish czech hebrew persian turkish thai urdu ukrainian".split())
RELATION_CUES = {
    "located_in": r"\b(?:located|situated|based|lies|in|part of|county seat|headquarters|location)\b",
    "educated_at": r"\b(?:attend\w*|studied|graduat\w*|alma mater|school career|education|enrolled)\b",
    "created_by": r"\b(?:created|written|directed|developed|composed|designed|invented|lyrics|songwriter)\b",
    "performed_by": r"\b(?:performed|recorded|sung|singing|singer|vocals|single by)\b",
    "released_by": r"\b(?:released|published|distributed|label)\b",
}


def fold(text: str) -> str:
    return normalize(accent_fold(text))


def starts_with_name(text: str, title: str) -> bool:
    return f"{fold(text)} ".startswith(f"{fold(title)} ")


def subject_anchors(doc: dict) -> dict[int, str]:
    title = title_surface(doc["title"])
    anchors = {}
    for span in doc["spans"]:
        i = span["index"]
        text = accent_fold(span_text(doc, i))
        if starts_with_name(text, title):
            anchors[i] = "title"
        if i == 0 and i not in anchors:
            # Geographic and campus qualifiers must still appear in the lead.
            base, *suffix = re.split(r", |\s*[–—]\s*", title, maxsplit=1)
            if suffix and starts_with_name(text, base) and f" {fold(suffix[0])} " in f" {fold(text)} ":
                anchors[i] = "qualified_title"
            names = title.split()
            if 2 <= len(names) <= 3 and all(re.fullmatch(r"[A-Z][A-Za-z'-]+", n) for n in names):
                if re.match(rf"^\s*{re.escape(names[0])}(?: [A-Z][A-Za-z'.-]+){{1,2}} "
                            rf"{re.escape(names[-1])}\b", text):
                    anchors[i] = "full_person_name"
                elif re.match(rf"^\s*{re.escape(names[-1])}\b", text):
                    remainder = re.sub(rf"^\s*{re.escape(names[-1])}\b", "", text, count=1)
                    if PERSON_LEAD.match(remainder):
                        anchors[i] = "lead_surname"
    return anchors


def binding(doc: dict, index: int, claimed: int | None) -> tuple[int | None, str]:
    anchors = subject_anchors(doc)
    if index in anchors:
        return None, anchors[index]
    title = title_surface(doc["title"])
    text = accent_fold(span_text(doc, index))
    possible = [i for i in anchors if 0 < index - i <= 3]
    ordered = sorted(possible, key=lambda i: (i != claimed, -i))
    for i in ordered:
        prior = span_text(doc, i)
        short = re.split(r", |\s*[–—]\s*", title, maxsplit=1)[0]
        acronym = "".join(w[0].upper() for w in short.split() if w and w[0].isalpha())
        if (len(acronym) >= 2 and re.match(rf"^\s*{re.escape(acronym)}\b", text)
                and re.search(rf"\b{re.escape(acronym)}\b", prior)
                and re.search(r"\b(?:known as|referred to|abbreviat\w*)\b", prior, re.I)):
            return i, "explicit_acronym"
        if ANAPHOR.match(text) and index - i <= 2:
            return i, "nearby_anaphor"
    raise ValueError("subject_not_bound")


def work_author_phrase(sentence: str, obj: str) -> bool:
    prefix = sentence[:sentence.index(obj)]
    return bool(re.search(
        r"^.{0,160}?\b(?:is|was)\s+(?:(?:an?|the)\s+)?"
        r"(?:(?!(?:based|adapted|from|on|of|in|about|with|for|and|film|play|novel|book|poem|manga)\b)[\w-]+\s+){0,5}"
        r"(?:(?:play|novel|book|poem|short story|manga|screenplay|work of fiction)\b(?:\s+in\s+\w+\s+acts)?\s+"
        r"(?:(?:written|authored|co-written)\s+)?|film\s+(?:written|authored|co-written)\s+)by\s+"
        r"(?:the (?:[\w-]+ ){0,4}(?:author|writer|playwright|philosopher|novelist),\s*)?$",
        prefix, re.I))


def validate_relation(predicate: str, obj: str, sentence: str, *, author_extension: bool = False) -> None:
    start = sentence.index(obj)
    prefix = sentence[max(0, start - 110):start]
    nearby = sentence[max(0, start - 100):start + len(obj) + 36]
    if predicate == "founded_date":
        # This predicate records formation of the subject, not release, renaming or a third party it founded.
        accepted = bool(re.search(
            r"(?:\b(?:was|were)\s+(?:(?:officially|originally)\s+)?|^\s*)"
            r"(?:founded|established|incorporated|formed)(?:\s+(?:in|on))?\s*$", prefix, re.I
        ))
    elif predicate in ("birth_date", "death_date"):
        cue = r"\bborn\b" if predicate == "birth_date" else r"\b(?:died|passed away)\b"
        accepted = bool(re.search(cue, prefix[-55:], re.I) and re.search(r"\d{3,4}", obj))
        for segment in re.findall(r"\(([^)]{4,130})\)", sentence):
            dates = re.findall(r"\d{4}", segment)
            if len(dates) == 2 and (re.search(r"[–—-]", segment) or re.search(r"\d{4}[A-Z]", segment)):
                accepted |= obj in segment and dates[int(predicate == "death_date")] in obj
        for dates in re.finditer(rf"(?P<birth>{DATE})\s*[–—]\s*(?P<death>{DATE})", sentence[:180]):
            accepted |= obj == dates["birth" if predicate == "birth_date" else "death"]
    elif predicate == "alias":
        accepted = bool(re.search(r"\b(?:known as|known by|known under|referred to|name from|renamed|under the name|born|née)\b",
                                  prefix, re.I))
        accepted &= not bool(re.search(r"\b(?:affiliated with|owned by|subsidiary of)\b", prefix, re.I))
    elif predicate == "language":
        accepted = bool(set(fold(obj).split()) & LANGUAGES and
                        re.search(r"\b(?:language|spoken|speaks|written in|published in)\b", nearby, re.I))
    else:
        accepted = bool(re.search(RELATION_CUES[predicate], nearby, re.I))
        if author_extension and predicate == "created_by":
            accepted |= work_author_phrase(sentence, obj)
        if predicate in ("created_by", "performed_by", "released_by"):
            accepted &= fold(obj) not in {"american", "brazilian", "english", "british", "canadian", "australian"}
    if not accepted:
        raise ValueError(f"relation_guard_{predicate}")


def filter_v2(doc: dict, items: list, *, author_extension: bool = False) -> tuple[list[dict], list[dict]]:
    kept, rejected, seen = [], [], set()
    for item in items:
        try:
            if not isinstance(item, dict):
                raise ValueError("invalid_item")
            pred, obj, index, anchor = (item.get(x) for x in ("predicate", "object", "sentence_id", "anchor_id"))
            if pred not in KINDS or not isinstance(obj, str) or not obj.strip():
                raise ValueError("predicate_or_object")
            if type(index) is not int or not 0 <= index < len(doc["spans"]):
                raise ValueError("invalid_sentence")
            sentence = span_text(doc, index)
            if obj not in sentence:
                raise ValueError("object_not_verbatim")
            anchor, ground = binding(doc, index, anchor)
            validate_relation(pred, obj, sentence, author_extension=author_extension and ground == "title")
            if pred == "alias" and fold(obj) == fold(title_surface(doc["title"])):
                raise ValueError("redundant_alias")
            ids = [index] if anchor is None else [anchor, index]
            row = {"doc_id": doc["id"], "source_version": doc["version"], "subject": doc["title"],
                   "predicate": pred, "object": obj, "sentence_id": index, "anchor_id": anchor,
                   "binding": ground,
                   "proof": [{**doc["spans"][i], "text": span_text(doc, i)} for i in ids]}
            if key(row) in seen or len(kept) >= MAX_FACTS:
                raise ValueError("duplicate_or_cap")
            seen.add(key(row))
            kept.append(row)
        except ValueError as error:
            rejected.append({"reason": str(error), "candidate": item})
    return kept, rejected


def rules_v2(doc: dict) -> tuple[list[dict], list[dict]]:
    from wiki import extract_aliases

    items = [{"predicate": "alias", "object": r.alias, "sentence_id": r.sentence_index, "anchor_id": None}
             for r in extract_aliases(doc)]
    for span in doc["spans"]:
        i = span["index"]
        try:
            anchor, _ = binding(doc, i, None)
        except ValueError:
            continue
        text = span_text(doc, i)
        for pred, pattern in PATTERNS.items():
            if match := re.search(pattern, text):
                items.append({"predicate": pred, "object": match["object"].rstrip(" .,;"),
                              "sentence_id": i, "anchor_id": anchor})
    return filter_v2(doc, items)
