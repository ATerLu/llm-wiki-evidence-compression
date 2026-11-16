"""Drain the 100-question RAGFlow compilation queue and evaluate the four arms."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from time import sleep

from external_compression import append
from ragflow_native_compare import run as compare
from ragflow_native_e2 import run as compare_e2
from ragflow_native_setup import authenticate, documents
from ragflow_result_audit import run as audit


def wait_for_index(output):
    state = json.loads((output / "native-full.json").read_text(encoding="utf-8"))
    expected = set(state["documents"].values())
    dataset = state["dataset_id"]
    client = authenticate()
    retries = Counter()
    last = None
    with (output / "native-full-progress.jsonl").open("a", encoding="utf-8") as log:
        while True:
            rows = list(documents(client, dataset))
            if {r["id"] for r in rows} != expected:
                raise RuntimeError("RAGFlow document IDs no longer match the frozen upload")
            statuses = Counter(r["ingestion_status"] for r in rows)
            compile_status = client.call("GET", f"/datasets/{dataset}/compilation/status")
            artifacts = client.call("GET", f"/datasets/{dataset}/artifacts", params={"page_size": 1})["total"]
            snapshot = {"documents": dict(statuses), "wiki_artifacts": artifacts,
                        **{k: compile_status[k] for k in ("state", "inflight", "backlog", "error", "current_phase")
                           if k in compile_status}}
            if snapshot != last:
                append(log, {"at": datetime.now(timezone.utc).isoformat(), **snapshot})
                print(json.dumps(snapshot, ensure_ascii=False), flush=True)
                last = snapshot
            if statuses == {"COMPLETED": len(expected)} and not snapshot["inflight"] and not snapshot["backlog"]:
                if not artifacts:
                    raise RuntimeError("RAGFlow finished indexing with zero Wiki artifacts")
                return
            failed = [r["id"] for r in rows if r["ingestion_status"] in ("FAILED", "CANCELED")]
            exhausted = [doc for doc in failed if retries[doc] >= 2]
            if exhausted:
                raise RuntimeError(f"{len(exhausted)} documents failed twice after retries")
            for start in range(0, len(failed), 50):
                batch = failed[start:start + 50]
                client.call("POST", "/documents/ingest", {"doc_ids": batch, "run": 1})
                retries.update(batch)
            sleep(20)


def write_results(output):
    native = json.loads((output / "native-full-summary.json").read_text(encoding="utf-8"))
    e2 = json.loads((output / "native-full-e2-summary.json").read_text(encoding="utf-8"))["comparison"]
    labels = {"A_raw": "A 原文检索", "B_native_wiki": "B 原生 Wiki", "C_guarded": "C 原句核验",
              "e2": "E2 同后端适配"}
    lines = ["# RAGFlow 原生 Wiki 100 题四组对照结果", "",
             "HotpotQA 和 MuSiQue 各 50 道已使用过的开发题，固定候选池。下表的输入仅计阅读器，E2 计入判断及回读。",
             "", "| 范围 | 方法 | F1 | EM | 必要证据完整 | 阅读输入 | 相对 A 的输入变化 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for scope, group in native["results"].items():
        reference = group["A_raw"]["api_prompt_total"]
        for arm, result in {**group, "e2": e2[scope]["e2"]}.items():
            total = result["api_prompt_total"]
            lines.append(f"| {scope} | {labels[arm]} | {result['f1']:.2%} | {result['em']:.2%} | "
                         f"{result['full_support']}/{group['A_raw']['n']} | {total:,} | {total / reference - 1:+.2%} |")
    all_rows = native["results"]["all"]
    lines.extend(["", "配对 F1 比较以百分点计，区间为 95%。",
                  "", "| 比较 | F1 差值 | 区间 | 提高题数 | 下降题数 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    pairs = {"B vs A": all_rows["B_native_wiki"]["vs_A"], "C vs A": all_rows["C_guarded"]["vs_A"],
             "C vs B": all_rows["C_guarded"]["vs_B"], "E2 vs A": e2["all"]["paired_f1"]}
    for label, row in pairs.items():
        lo, hi = row["f1_delta_ci95"]
        lines.append(f"| {label} | {row['delta'] * 100:+.2f} | [{lo * 100:+.2f}, {hi * 100:+.2f}] | "
                     f"{row['improved']} | {row['worsened']} |")
    routes = all_rows["C_guarded"]["route"]
    lines.extend(["", f"C 路由分布 `{json.dumps(routes, ensure_ascii=False)}`。",
                  f"E2 路由分布 `{json.dumps(e2['all']['routes'], ensure_ascii=False)}`。",
                  f"数据集生成 {native['wiki_artifacts']:,} 个 Wiki 产物。",
                  f"检索模型用量记录 `{json.dumps(native['online_query_embeddings'], ensure_ascii=False)}`。", ""])
    with (output / "native-full-results.md").open("w", encoding="utf-8") as report:
        report.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    wait_for_index(args.output)
    compare(args.output, "full")
    compare_e2(args.output, "full")
    write_results(args.output)
    audit(args.output, "full")
