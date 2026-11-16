"""Preserve the exact span differences and qualitative notes for the 36 old validation cases."""

import json
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired import inputs, labels_for
from prepare import jsonl, read_jsonl
from wiki_bridge import span_text

# Notes refer to the observed delta, not to everything already present in the input.
NOTES = {
    "5ac04b035542996f0d89cb81": "Added Hammerstein identity rather than an award count; displaced unrelated song text.",
    "5a7c08ab5542990527d5544b": "Added Beatles attribution for Drive My Car; displaced an unrelated single's chart position.",
    "5a7e195955429965cec5ea52": "Added Frederick population rather than county affiliation; displaced another courthouse's history.",
    "5a7bdea35542997c3ec9729d": "Added 1914 film footage and generic Garrett career; displaced 1949 film production evidence.",
    "5adff2d155429925eb1afb72": "Added relevant railroad alias plus unrelated Pennsylvania railroad history.",
    "5ae7b6ed5542993210983f03": "Added coat-of-arms detail and geographic Bald Eagle Valley for a question about an individual eagle.",
    "5ac1371f55429964131be198": "Added relevant match participant support; displaced years and unrelated finals.",
    "5ae2c45c5542991a06ce98cf": "Added a different actor and generic Tim Conway career; no added birth date.",
    "5ae7187f554299572ea546d4": "Added Northern Cheyennes battle context; displaced Apache and Crow scout distractors.",
    "5ab5bd255542997d4ad1f1cd": "Added rebellion organizers plus a similarly named baronet; displaced mostly family-history distractors.",
    "5abf9b815542994516f454d8": "Replaced sequel material with context from the correct film.",
    "5a7337e25542994cef4bc4d2": "Added family mention from another series and Kyle Massey career; no new Raven-Symoné identification.",
    "5a826f7b55429966c78a6a1d": "Added a winner's single and X Factor Vietnam; presenter evidence was not restored.",
    "5a77d8fc55429967ab1052d1": "Added the Ottawa University location in Kansas plus an unrelated founding date.",
    "5a8c6f775542996e8ac88aab": "Added Darren Criss birth year plus another album's release date.",
    "2hop__58747_82551": "Added finance and BRICS keyword matches for unrelated countries; no change to correct answer.",
    "2hop__80185_55948": "Added Super Bowl XLII winner evidence but not the requested 0–5 date; also added distractor football spans.",
    "3hop1__858730_386977_851569": "Added MC Eiht identity plus two unrelated district fragments.",
    "2hop__157343_112595": "Restored Qian Yuanguan death date; model gave an exact day where label is a year.",
    "2hop__352332_112595": "Added spouse death date and generic dynasty detail; answer became over-specific.",
    "2hop__434251_77790": "Added generic Portuguese exploration; displaced Gold Coast's dated 1482 sentence.",
    "2hop__159025_82551": "Added a Central African politician and a finance minister in India, neither the requested South African minister.",
    "2hop__251426_88653": "Added another team's history and a player-team list; no last-win date recovered.",
    "2hop__136671_91469": "Added film-series connection, but also unrelated regulatory text.",
    "2hop__56754_77790": "Added 1484 Congo exploration and a different Gold Coast; locality and answer granularity remained wrong.",
    "3hop1__128640_255609_605023": "Added Red Bay founding year, irrelevant to country; removed a WRMG fragment.",
    "2hop__107184_3836": "Added generic neuroscience from Brain and a different torch-relay fact; no setting date recovered.",
    "2hop__815999_49441": "Added Gossip Girl scene and unrelated Babylon 5 fragment; no paternity statement added.",
    "2hop__128849_406512": "Restored Brandon's county and displaced unrelated Polish municipality.",
    "3hop2__30654_199039_89329": "Restored headquarters location but not the Bethel founder; also added unrelated Black Bear Ranch.",
    "2hop__56754_76652": "Added Namibia history and incomplete Somalis fragment; returned numeral 8 against label eight.",
    "2hop__242144_88653": "Added Cowboys and a quarterback fact; failed to establish the requested team and date.",
    "2hop__9922_20556": "Added unrelated North Carolina Reconstruction history; answer unchanged after normalization.",
    "2hop__80185_64290": "Added the Giants' victory over the Patriots plus Super Bowl LII distractor; no last-win date.",
    "2hop__153348_49441": "Added a Vampire Diaries character because of the generic title Pilot.",
    "2hop__128603_397470": "Restored Rock Hill's York County statement and displaced other station details.",
}


def run(output: Path) -> None:
    rows = read_jsonl(Path("runs/cloud-bridge-validation-qwen35/scored.jsonl"))
    off = {(r["dataset"], r["query_id"]): r for r in rows if r["method"] == "lexical_sentence"}
    audited, counts = [], {}
    for dataset in ("hotpot", "musique"):
        queries, corpus = inputs(dataset)
        labels = labels_for(dataset)
        counts[dataset] = Counter()
        for new in rows:
            if new["dataset"] != dataset or new["method"] != "wiki_bridge":
                continue
            qid = new["query_id"]
            old = off[dataset, qid]
            if old["evidence"] == new["evidence"]:
                continue
            query, label = queries[qid], labels[qid]
            docs = [corpus[c["id"]] for c in query["candidates"]]
            ref = {tuple(x) for x in label["support"]} if dataset == "hotpot" else set(label["support"])

            def spans(row):
                return {(i, j) for i, indices in row["selected_blocks"] for j in indices}

            def details(positions):
                output_spans = []
                for i, j in sorted(positions):
                    key = query["candidates"][i]["key"]
                    output_spans.append({
                        "candidate": i, "title": docs[i]["title"], "source_version": docs[i]["version"],
                        **docs[i]["spans"][j], "text": span_text(docs[i], j),
                        "in_annotated_support": ((key, j) if dataset == "hotpot" else key) in ref,
                    })
                return output_spans

            gained, lost = details(spans(new) - spans(old)), details(spans(old) - spans(new))
            audited.append({"dataset": dataset, "query_id": qid, "question": query["question"],
                            "gold": label["answer"], "off_answer": old["answer"], "on_answer": new["answer"],
                            "added": gained, "displaced": lost, "note": NOTES[qid]})
            counter = counts[dataset]
            counter.update(cases=1, spans_added=len(gained), spans_displaced=len(lost),
                           annotated_added=sum(s["in_annotated_support"] for s in gained),
                           annotated_displaced=sum(s["in_annotated_support"] for s in lost),
                           f1_up=new["f1"] > old["f1"], f1_down=new["f1"] < old["f1"])
    assert {r["query_id"] for r in audited} == NOTES.keys()
    output.mkdir(parents=True)
    jsonl(output / "audit.jsonl", audited)
    (output / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
