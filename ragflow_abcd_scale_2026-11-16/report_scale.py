"""Render reproducible 200/300/500 paired comparisons from completed results."""

import json
from pathlib import Path

from prepare import read_jsonl


DATA = Path("data")
RESULTS = Path("results")
LABELS = {"A": "A 原文 RAG", "B": "B 原生 Wiki＋RAG", "C": "C Wiki＋在线组件",
          "D": "D Wiki＋在线组件＋原句关系"}


def render():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    freeze = json.loads((DATA / "freeze.json").read_text(encoding="utf-8"))
    scores = {(r["query_id"], r["method"]): r for r in read_jsonl(RESULTS / "scored.jsonl")}
    refs = {r["id"]: r for ds in ("hotpot", "musique") for r in read_jsonl(DATA / f"labels-{ds}.jsonl")}
    lines = ["# RAGFlow A/B/C/D 扩规模实验结果", "", f"冻结 {freeze['queries']} 道官方开发题，"
             f"共 {freeze['pages']:,} 份源文档。HotpotQA 和 MuSiQue 各 250 道。200 题为预先冻结的子集，"
             "新增 300 题单列。全程共用同一个最终检索索引。历史题族按本地记录排除。", "",
             "总在线 token 为阅读输入、阅读输出和两组检索实测查询向量输入的平均分摊之和。离线编译另计。",
             "F1 变化区间按题族配对重采样。", ""]
    for key, title in (("200", "200 题"), ("additional300", "新增 300 题"), ("500", "500 题")):
        group = summary[key]
        lines.extend([f"## {title}", "", "| 通道 | F1 | EM | 阅读输入 | 总在线 token | 完整支持 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for arm, r in group["arms"].items():
            lines.append(f"| {LABELS[arm]} | {r['f1']:.2%} | {r['em']:.2%} | {r['reader_input']:,} | "
                         f"{r['total_tokens']:,} | {r['strict_support']} |")
        lines.extend(["", "| 比较 | Δ F1，百分点 | 95% 区间 | Δ 总在线 token | 材料改变题数 |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for label in ("C_vs_B", "D_vs_C", "C_vs_A", "D_vs_A"):
            r = group["comparisons"][label]
            low, high = r["f1_delta_ci95"]
            lines.append(f"| {label.replace('_vs_', ' 对 ')} | {100*r['delta']:+.2f} | "
                         f"[{100*low:+.2f}, {100*high:+.2f}] | {r['total_token_delta']:+,} "
                         f"({r['total_token_delta_rate']:+.2%}) | {len(r['changed_query_ids'])} |")
        lines.append("")
    lines.extend(["## 500 题逐题例子", ""])
    for new, old in (("C", "B"), ("D", "C")):
        ids = summary["500"]["comparisons"][f"{new}_vs_{old}"]["changed_query_ids"]
        lines.extend([f"### {new} 对 {old}", ""])
        if not ids:
            lines.extend(["两组阅读材料逐题相同。", ""])
            continue
        ids.sort(key=lambda qid: (scores[qid, new]["f1"] - scores[qid, old]["f1"], qid))
        for qid in dict.fromkeys((ids[-1], ids[0])):
            a, b = scores[qid, old], scores[qid, new]
            lines.extend([f"- **{a['question']}**（{a['dataset']}，`{qid}`）  ",
                          f"  参考答案「{refs[qid]['answer']}」。{old} 答「{a['answer']}」，{new} 答「{b['answer']}」。"
                          f"阅读输入 {a['api_prompt_tokens']} → {b['api_prompt_tokens']}，"
                          f"F1 {a['f1']:.2%} → {b['f1']:.2%}。"])
        lines.append("")
    (RESULTS / "扩规模实验结果.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    render()
