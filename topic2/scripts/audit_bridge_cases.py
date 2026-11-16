"""Audit all discovery disagreements and test five targeted original-span restorations."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import SYSTEM, sentence_blocks
from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader, ReaderError
from paired import inputs, labels_for
from prepare import jsonl, read_jsonl
from score import official_answer

# Candidate and sentence positions refer to the frozen discovery workload.
HOT = {
    "5a78ebfd55429974737f7901": ("date_format", [], "Same date, different order and punctuation"),
    "5adea72b55429975fa854f48": ("terminal", [(5, 1)], "Myst → developer"),
    "5a85caf95542997175ce2020": ("terminal", [(7, 1)], "Shine → Starbucks → founding year"),
    "5add3aef5542995b365faac9": ("bridge_and_terminal", [(9, 1), (9, 3)], "Documentary participant and production year"),
    "5a8ce1a955429941ae14df2f": ("terminal", [(9, 3)], "Cain-Sloan → renamed Dillard's"),
    "5ae1fd425542997283cd2310": ("bridge_and_terminal", [(7, 0), (7, 1)], "Publisher and magazine"),
    "5a74c2c85542996c70cfaddc": ("terminal", [(9, 3)], "Miss Piggy → performer"),
}
MUS = {
    "2hop__834440_121865": ("terminal", [(15, 0)], "Havana → Kenny G → college"),
    "2hop__127505_68782": ("terminal", [(15, 0)], "Spanish name of Mexico City"),
    "2hop__142548_303417": ("terminal", [(3, 0)], "Dwight Yoakam → Reprise Records"),
    "2hop__605693_121865": ("terminal", [(6, 0)], "Heart and Soul → Kenny G → college"),
    "2hop__583487_121865": ("terminal", [(14, 0)], "Duotones → Kenny G → college"),
    "2hop__143889_121865": ("terminal", [(9, 0)], "Hi, How Ya Doin'? → Kenny G → college"),
    "2hop__185142_6660": ("terminal", [(8, 1)], "Catalan region → Iberian Peninsula"),
    "3hop1__4264_92106_30601": ("bridge_only", [(13, 1)], "Galapagos → Spanish; newspaper answer already present"),
    "2hop__26947_59695": ("terminal", [(16, 0), (16, 5)], "Unambiguous singular abbreviation examples missing"),
    "2hop__659879_121865": ("terminal", [(7, 0)], "Rhythm & Romance → Kenny G → college"),
    "2hop__142888_303417": ("terminal", [(11, 0)], "It Won't Hurt → Dwight Yoakam → label"),
    "2hop__496399_121865": ("terminal", [(6, 0)], "Faith → Kenny G → college"),
    "3hop1__4264_92106_6713": ("bridge_and_terminal", [(7, 1), (5, 0)], "Galapagos → Spanish → 80% lexical similarity"),
    "2hop__407343_121865": ("terminal", [(6, 0)], "Don't Make Me Wait for Love → Kenny G → college"),
    "2hop__6649_6713": ("terminal", [(14, 0)], "Spanish → 80% lexical similarity"),
}
INTERVENE = {
    "5adea72b55429975fa854f48", "5a85caf95542997175ce2020", "5a8ce1a955429941ae14df2f",
    "2hop__834440_121865", "3hop1__4264_92106_30601",
}


def run(output: Path) -> None:
    a = read_jsonl(Path("runs/cloud-a-smoke-qwen35/scored.jsonl"))
    b = read_jsonl(Path("runs/cloud-b-smoke-qwen35/scored.jsonl"))
    off = {(r["dataset"], r["query_id"]): r for r in b if r["method"] == "lexical_sentence"}
    output.mkdir(parents=True)
    records, plans = [], []
    for dataset, manual in (("hotpot", HOT), ("musique", MUS)):
        queries, corpus = inputs(dataset)
        labels = labels_for(dataset)
        disagreements = {r["query_id"]: r for r in a if r["dataset"] == dataset and r["em"]
                         and not off[dataset, r["query_id"]]["em"]}
        assert disagreements.keys() == manual.keys()
        for qid, (kind, missing, explanation) in manual.items():
            query = queries[qid]
            docs = [corpus[c["id"]] for c in query["candidates"]]
            compact = off[dataset, qid]
            selected = {i: set(indices) for i, indices in compact["selected_blocks"]}
            assert all(j not in selected.get(i, set()) for i, j in missing)
            records.append({
                "dataset": dataset, "query_id": qid, "family_id": query["family_id"],
                "question": query["question"], "category": kind, "explanation": explanation,
                "a_answer": disagreements[qid]["answer"], "off_answer": compact["answer"],
                "missing": [{
                    "candidate": i, "title": docs[i]["title"], "doc_id": docs[i]["id"],
                    "source_version": docs[i]["version"], **docs[i]["spans"][j],
                    "text": docs[i]["text"][docs[i]["spans"][j]["start"]:docs[i]["spans"][j]["end"]],
                } for i, j in missing],
            })
            if qid in INTERVENE:
                for i, j in missing:
                    selected.setdefault(i, set()).add(j)
                plans.append((dataset, qid, query["question"], compact["evidence"],
                              sentence_blocks(docs, selected), labels[qid]))
    jsonl(output / "audit.jsonl", records)
    reader, tokenizer = CloudReader(), BudgetTokenizer()
    with (output / "restoration.jsonl").open("x", encoding="utf-8") as file:
        for dataset, qid, question, original, restored, label in plans:
            for mode, text in (("repeat_off", original), ("restore_only_missing", restored)):
                try:
                    result = reader.generate([{"role": "system", "content": SYSTEM}, {
                        "role": "user", "content": f"Evidence:\n{text}\n\nQuestion: {question}",
                    }])
                except ReaderError as error:
                    result = {"answer": "", "status": "error", "error": str(error)}
                em, f1 = official_answer(dataset, result["answer"], label)
                row = {"dataset": dataset, "query_id": qid, "mode": mode, "em": em, "f1": f1,
                       "evidence": text, "tokens": len(tokenizer.tokenize(text)), **result}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                print(qid, mode, result["status"], em, flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
