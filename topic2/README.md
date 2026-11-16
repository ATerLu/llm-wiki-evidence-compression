# 选题二实验工程

## 当前入口

答案生成统一调用阿里云百炼北京接入点的固定模型 `qwen3.5-plus-2026-04-20`。进程从 `DASHSCOPE_API_KEY` 读取授权。使用公开 Qwen 分词器文件在本地核算证据预算，模型生成在云端完成。

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.lock
& .venv/Scripts/python.exe -X utf8 cloud_experiment.py A runs/cloud-a-new --workers 4
& .venv/Scripts/python.exe -X utf8 cloud_experiment.py B runs/cloud-b-new --reference-a runs/cloud-a-new --workers 6
& .venv/Scripts/python.exe -X utf8 cloud_compare.py runs/cloud-a-new runs/cloud-b-new
```

A 按数据原序发送完整候选。B 在 1,024 证据 token 下比较整段、独立句选、邻句和 Wiki 依赖规则。A 与 B 共用冻结题集、模型和提示。B 的输出中有独立的 SQLite Wiki 索引及逐题触发记录。[云端 A/B 实验报告](reports/云端AB_原始与Wiki对照.md)

新版跨页依赖验证使用 `A --scope validation` 和 `V --scope validation --reference-a <本次 A 运行目录>`。V 比较独立句选与标题桥接，并对相同输入复用一次回答。[必要证据审计与保留验证](reports/必要证据审计与云端保留验证.md)

三档预算验证使用 `A --scope official_dev` 与 `S --scope official_dev --reference-a <本次 A 运行目录>`，用 `sweep_compare.py` 汇总。新版 B 在标题消歧后，按问题属性选择一条目标证据。[证据替换审计与三档预算结果](reports/证据替换审计与三档预算结果.md)

`external_compression.py` 在同一官方 dev 子集上比较完整候选、B 开关、百炼语义重排抽取和 LLM 直接抽取。运行命令与首轮、结构化重跑结果见[同预算外部压缩对照与研究脉络](reports/同预算外部压缩对照与研究脉络.md)。

`semantic_bridge_compile.py` 冻结新题并编译 Wiki，`semantic_bridge_experiment.py` 比较同一语义重排上的规则与编译依赖边。入口、结果和出处复核见[语义重排与 Wiki 依赖桥接保留题对照](reports/语义重排与Wiki依赖桥接保留题对照.md)。

`data_use_audit.py` 核对历史题号和筛选范围。`dependency_development.py` 在已见问题上复用归档排名和编译结果，验证共用原句补证。`semantic_bridge_experiment.py` 提供独立的 `--dependency-extension` 条件。[数据使用审计与开发回放](reports/数据使用审计与依赖补证开发回放_2026-11-11.md)

`scripts/probe_question_intent.py` 对开发回放中缺编译首跳的题做只读题面的百炼解析，保存原文审核、响应与成本。[首跳证据与问题意图诊断](reports/首跳证据与问题意图诊断_2026-11-11.md)

`intent_evidence_experiment.py` 复用冻结解析、语义排名和候选，比较直接找原句与加入 Wiki 出处。`intent_evidence.py` 在装配前核验原句编号、共同实体和州名冲突。[问题解析接入原句装配对照](reports/问题解析接入原句装配对照_2026-11-11.md)

`wiki_window_experiment.py` 生成等 token 上限的语义候选与 Wiki 出处候选，并保存逐步原句覆盖。运行阶段复用 `intent_evidence_experiment.py`。[Wiki 覆盖和原文缩窗对照](reports/Wiki必要原句覆盖与等预算原文缩窗_2026-11-11.md)

证据保护与预算路由使用 `G` 阶段。先在 `guard_development` 调节 `guard_policy.py`，冻结后在 `guard_heldout` 验证。[直接证据保护与自适应预算验证](reports/直接证据保护与自适应预算验证.md)

LLM-Wiki 首轮编译入口为 `wiki_compile_pilot.py`。主体和关系校验改造后，用 `wiki_compiler_heldout.py` 比较同一 B 接入规则索引和 LLM 索引。[主体校验与保留题对照](reports/LLM-Wiki主体校验与保留题对照.md)

`data/processed/<dataset>/` 下有 `corpus.jsonl`、`queries.jsonl`、`labels.jsonl` 和 `splits.json`。原始数据、评分脚本、许可和历史资源来源见 [sources.json](sources.json) 与 [downloads.json](downloads.json)。云端分词器为 [Qwen3.5 固定版 tokenizer.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/e2ef7817335642e36630277239ea21d48f4e9fb4/tokenizer.json)，路径为 `vendor/tokenizer/qwen3.5.json`。各次实验的配置、源码、输入、结果与模型用量保存在各自运行目录。

## 历史实验

旧本地权重、推理器和启动脚本已移出项目。已有运行的源文件、启动参数、校验值和逐题结果保留在历史归档。[第一周基线](reports/第一周_实验准备与进度.md) · [第二周同题装配对照](reports/第二周_同题装配对照.md)
