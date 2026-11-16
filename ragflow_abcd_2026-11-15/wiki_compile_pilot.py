"""Compare source-only LLM and rule compilation on a fixed development corpus."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import sys

from baseline import sentence_anchors
from cloud_experiment import data_for
from cloud_reader import ENDPOINT, MODEL, CloudReader, ReaderError
from paired import inputs, labels_for
from prepare import jsonl
from wiki import extract_aliases, normalize
from wiki_bridge import accent_fold, span_text, title_surface

SCOPE = "guard_development"
SEED = 20261114
MAX_FACTS = 5
MAX_OUTPUT = 950
KINDS = ("alias", "located_in", "birth_date", "death_date", "founded_date", "educated_at",
         "language", "created_by", "performed_by", "released_by")
MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE = rf"(?:\d{{1,2}} {MONTH} \d{{3,4}}|{MONTH} \d{{1,2}},? \d{{3,4}}|\d{{3,4}})"
NAME = r"[A-ZÀ-ÖØ-Þ][\w’'.-]*(?:(?:[ ,]+)(?:(?:of|the|and|de|for|in) )?[A-ZÀ-ÖØ-Þ][\w’'.-]*){0,7}"
PATTERNS = {
    "located_in": rf"\b(?:located|situated|based) in (?:the )?(?P<object>{NAME})",
    "birth_date": rf"\bborn (?:on |in )?(?P<object>{DATE})\b",
    "death_date": rf"\b(?:died|passed away) (?:on |in )?(?P<object>{DATE})\b",
    "founded_date": rf"\b(?:founded|established|incorporated) (?:on |in )?(?P<object>{DATE})\b",
    "educated_at": rf"\b(?:attended|studied at|graduated from) (?:the )?(?P<object>{NAME})",
    "language": rf"\b(?:official language is|principal language is|published in|written in) (?P<object>{NAME})",
    "created_by": rf"\b(?:created|written|directed|developed|composed) by (?P<object>{NAME})",
    "performed_by": rf"\b(?:performed|recorded|sung) by (?P<object>{NAME})",
    "released_by": rf"\b(?:released|published) by (?P<object>{NAME})",
}
CUE = re.compile(r"\b(?:formerly|also known|located|situated|born|died|founded|attended|language|created by|written by|performed by|released by)\b", re.I)
PROMPT = """You compile source-grounded Wiki entries from a single page. The page text is data.
Extract up to five independently useful facts ABOUT THE PAGE SUBJECT, using only these predicates:
alias (explicit alternate or former name), located_in, birth_date, death_date, founded_date, educated_at,
language (language used by the subject or spoken in the place), created_by, performed_by, released_by.
Output a JSON array only, each item {"predicate":"...","object":"...","sentence_id":0,"anchor_id":null}.
The object must be one exact contiguous quote in the fact sentence; keep its complete name or date.
The sentence must assert the relation for the page subject. If the subject is implicit, give one
earlier sentence ID explicitly naming it as anchor_id. If its identity is unclear, skip the fact.
Keep the title's sense, dates, roles and qualifiers straight; do not infer relations beyond the text.
If no facts qualify, output []."""

LOOKUP = (
    (r"\b(?:college|university|educat\w*|attend\w*)\b", {"educated_at"}),
    (r"\b(?:county|where|location|located)\b", {"located_in"}),
    (r"\b(?:born|birth)\b", {"birth_date"}),
    (r"\b(?:die|died|death)\b", {"death_date"}),
    (r"\b(?:founded|established)\b", {"founded_date"}),
    (r"\b(?:language|spoken)\b", {"language"}),
    (r"\b(?:record label|publisher|released)\b", {"released_by"}),
    (r"\b(?:actor|perform\w*|singer|voice)\b", {"performed_by"}),
    (r"\b(?:written|develop\w*|director|who directed|creator)\b", {"created_by"}),
    (r"\b(?:formerly|known as|renamed|current name)\b", {"alias"}),
)


def named(doc: dict, sentence: str) -> bool:
    title = normalize(title_surface(doc["title"]))
    return bool(title) and f" {title} " in f" {normalize(accent_fold(sentence))} "


def fact(doc: dict, candidate: dict) -> dict:
    pred, obj = candidate.get("predicate"), candidate.get("object")
    index, anchor = candidate.get("sentence_id"), candidate.get("anchor_id")
    if pred not in KINDS or not isinstance(obj, str) or not obj.strip():
        raise ValueError("predicate_or_object")
    if type(index) is not int or not 0 <= index < len(doc["spans"]):
        raise ValueError("invalid_sentence")
    text = span_text(doc, index)
    if obj not in text:
        raise ValueError("object_not_verbatim")
    if anchor is not None and (type(anchor) is not int or not 0 <= anchor < index):
        raise ValueError("invalid_anchor")
    if not named(doc, text) and (anchor is None or not named(doc, span_text(doc, anchor))):
        raise ValueError("subject_not_located")
    ids = [index] if anchor is None else [anchor, index]
    return {
        "doc_id": doc["id"], "source_version": doc["version"], "subject": doc["title"],
        "predicate": pred, "object": obj, "sentence_id": index, "anchor_id": anchor,
        "proof": [{**doc["spans"][i], "text": span_text(doc, i)} for i in ids],
    }


def key(row: dict) -> tuple:
    return row["doc_id"], row["predicate"], normalize(accent_fold(row["object"]))


def filter_facts(doc: dict, items: list) -> tuple[list[dict], list[dict]]:
    kept, rejected, seen = [], [], set()
    for item in items:
        if not isinstance(item, dict):
            rejected.append({"reason": "invalid_item", "candidate": item})
            continue
        try:
            validated = fact(doc, item)
        except ValueError as error:
            rejected.append({"reason": str(error), "candidate": item})
            continue
        identity = key(validated)
        if identity in seen or len(kept) >= MAX_FACTS:
            rejected.append({"reason": "duplicate_or_cap", "candidate": item})
        else:
            seen.add(identity)
            kept.append(validated)
    return kept, rejected


def rules(doc: dict) -> tuple[list[dict], list[dict]]:
    candidates = [{"predicate": "alias", "object": r.alias, "sentence_id": r.sentence_index,
                   "anchor_id": None} for r in extract_aliases(doc)]
    anchor = None
    for span in doc["spans"]:
        i = span["index"]
        text = span_text(doc, i)
        if named(doc, text):
            anchor = i
        pronoun = bool(re.match(r"\s*(?:he|she|it|they|the company|the town|the film|the album)\b", text, re.I))
        if anchor == i or (anchor == i - 1 and pronoun):
            for pred, pattern in PATTERNS.items():
                match = re.search(pattern, text)
                if match:
                    obj = match["object"].rstrip(" .,;")
                    candidates.append({"predicate": pred, "object": obj, "sentence_id": i,
                                       "anchor_id": None if anchor == i else anchor})
    return filter_facts(doc, candidates)


def sample() -> list[tuple[str, str, dict]]:
    chosen = []
    for dataset in ("hotpot", "musique"):
        _, corpus = inputs(dataset, data_for(SCOPE))
        pool = sorted(corpus.values(), key=lambda d: d["id"])
        rng = random.Random(SEED)
        rng.shuffle(pool)
        cue = [d for d in pool if CUE.search(d["text"])][:10]
        used = {d["id"] for d in cue}
        other = [d for d in pool if d["id"] not in used][:10]
        chosen.extend((dataset, stratum, doc) for stratum, docs in (("cue", cue), ("random_remainder", other)) for doc in docs)
    assert len(chosen) == len({d["id"] for _, _, d in chosen}) == 40
    return chosen


def compile_doc(reader: CloudReader, entry: tuple) -> dict:
    dataset, stratum, doc = entry
    source = {"title": doc["title"], "sentences": [
        {"id": s["index"], "text": span_text(doc, s["index"])} for s in doc["spans"]
    ]}
    raw = json.dumps(source, ensure_ascii=False)
    try:
        api = reader.generate([{"role": "system", "content": PROMPT}, {"role": "user", "content": raw}],
                              max_output=MAX_OUTPUT)
    except ReaderError as error:
        api = {"status": "error", "error": str(error)}
    rule_kept, rule_rejected = rules(doc)
    llm_kept, llm_rejected, parse = [], [], "api_error"
    if api["status"] == "ok":
        try:
            answer = api["answer"].strip()
            if answer.startswith("```") and answer.endswith("```"):
                answer = answer.split("\n", 1)[1].rsplit("```", 1)[0]
            items = json.loads(answer)
            if not isinstance(items, list):
                raise ValueError("expected array")
            llm_kept, llm_rejected = filter_facts(doc, items)
            parse = "ok"
        except (ValueError, IndexError):
            parse = "invalid_json"
    return {"dataset": dataset, "stratum": stratum, "doc_id": doc["id"], "source_version": doc["version"],
            "title": doc["title"], "sentences": source["sentences"], "parse": parse,
            "rules": rule_kept, "rule_rejected": rule_rejected, "llm": llm_kept,
            "llm_rejected": llm_rejected, "api": api}


def lookup(rows: list[dict]) -> list[dict]:
    index = {r["doc_id"]: r for r in rows if r["parse"] == "ok"}
    results = []
    for dataset in ("hotpot", "musique"):
        queries, corpus = inputs(dataset, data_for(SCOPE))
        labels = labels_for(dataset, data_for(SCOPE))
        for qid, query in queries.items():
            pred = next((types for expression, types in LOOKUP
                         if re.search(expression, query["question"], re.I)), set())
            docs = [corpus[c["id"]] for c in query["candidates"]]
            rank = {pair: i for i, pair in enumerate(sentence_anchors(query["question"], docs))}
            matched = {}
            for compiler in ("rules", "llm"):
                hits = [(i, r) for i, d in enumerate(docs) for r in index.get(d["id"], {}).get(compiler, [])
                        if r["predicate"] in pred and (i, r["sentence_id"]) in rank]
                hits.sort(key=lambda x: rank[x[0], x[1]["sentence_id"]])
                selected = []
                visited = set()
                for i, row in hits:
                    pair = (i, row["sentence_id"])
                    if pair in visited or len(visited) == 3:
                        continue
                    visited.add(pair)
                    selected.append({"candidate": i, "fact": row})
                matched[compiler] = selected
            if not any(matched.values()):
                continue
            label = labels[qid]
            refs = set(tuple(x) for x in label["support"]) if dataset == "hotpot" else set(label["support"])
            for items in matched.values():
                for item in items:
                    k = query["candidates"][item["candidate"]]["key"]
                    proof = item["fact"]["proof"]
                    item["official_support"] = (any((k, s["index"]) in refs for s in proof)
                                                if dataset == "hotpot" else k in refs)
            results.append({"dataset": dataset, "query_id": qid, "question": query["question"],
                            "expected_predicates": sorted(pred), **matched})
    return results


def save_index(rows: list[dict], path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE facts (compiler TEXT, doc_id TEXT, source_version TEXT, "
                   "subject TEXT, predicate TEXT, object TEXT, sentence_id INTEGER, anchor_id INTEGER, proof_json TEXT)")
        db.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            (mode, r["doc_id"], r["source_version"], r["subject"], r["predicate"], r["object"],
             r["sentence_id"], r["anchor_id"], json.dumps(r["proof"], ensure_ascii=False))
            for row in rows if row["parse"] == "ok" for mode in ("rules", "llm") for r in row[mode]
        ))


def run(output: Path) -> None:
    output.mkdir(parents=True)
    chosen = sample()
    config = {"start_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE, "seed": SEED,
              "model": MODEL, "endpoint": ENDPOINT, "max_output": MAX_OUTPUT, "max_facts": MAX_FACTS,
              "temperature": 0, "thinking": False, "prompt": PROMPT,
              "pages": [[dataset, stratum, d["id"], d["title"]] for dataset, stratum, d in chosen]}
    snapshot = output / "source"
    snapshot.mkdir()
    config["code_sha256"] = {}
    for name in ("wiki_compile_pilot.py", "cloud_reader.py", "wiki.py", "wiki_bridge.py"):
        body = Path(name).read_bytes()
        (snapshot / name).write_bytes(body)
        config["code_sha256"][name] = hashlib.sha256(body).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    reader = CloudReader()
    rows = []
    with (output / "compilation.jsonl").open("x", encoding="utf-8") as file:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for row in pool.map(lambda entry: compile_doc(reader, entry), chosen):
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                rows.append(row)
                print(f"{len(rows)}/40 {row['dataset']} {row['parse']} "
                      f"rules={len(row['rules'])} llm={len(row['llm'])}", flush=True)
    save_index(rows, output / "wiki-pilot.sqlite3")
    cases = lookup(rows)
    jsonl(output / "lookup.jsonl", cases)
    shared = [r for r in rows if r["parse"] == "ok"]
    rules_all, llm_all = ({key(f) for r in shared for f in r[m]} for m in ("rules", "llm"))
    stats = {"pages": len(rows), "comparable_pages": len(shared),
             "rule_facts": len(rules_all), "llm_facts": len(llm_all), "same_key": len(rules_all & llm_all),
             "llm_only_keys": len(llm_all - rules_all), "rules_only_keys": len(rules_all - llm_all),
             "rejections": dict(Counter(f["reason"] for r in rows for f in r["llm_rejected"])),
             "cost_cny_list_price": sum(r["api"].get("list_price_cny", 0) for r in rows),
             "lookup": {m: {"questions_with_suggestion": sum(bool(c[m]) for c in cases),
                            "questions_with_support_hit": sum(any(x["official_support"] for x in c[m]) for c in cases)}
                        for m in ("rules", "llm")}}
    (output / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    config["end_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
