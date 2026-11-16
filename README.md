# 面向多跳问答的 LLM-Wiki 依赖保持型证据压缩

**Dependency-Preserving Evidence Compression for Multi-Hop QA with LLM-Wiki**

研究组件部署在候选检索和阅读器之间。在线程序尝试从候选材料定位能相互支持的跨页原句，核验实体、关系和限定条件，在找到唯一可核验且更短的证据链时替换完整上下文。可选的离线索引用版本和原句绑定关系，SQLite 复用单独验证。

## 当前版本

| 路径 | 内容 |
| --- | --- |
| [100 题已完成实验](ragflow_abcd_2026-11-15/实验报告.md) | RAGFlow 检索快照、冻结输入、四条通道的逐题请求、评分和汇总。 |
| [扩规模实验](ragflow_abcd_scale_2026-11-16/建库断点与恢复手册.md) | 500 道冻结题及其 200 题子集，已编译的原句关系、流水线映射和建库进度。最终索引和 500 题答题评测待完成。 |
| [原文关系定位](ragflow_abcd_2026-11-15/raw_relation_locator.py) · [在线装配](ragflow_abcd_2026-11-15/run_abcd.py) | 100 题版本的研究组件。 |
| [历史实验与 SQLite 原型](topic2/README.md) | 实验代码与选定的历史报告。全量历史运行目录保存在本地工作区。 |

100 道开发题上，C 相对 B 总在线 token 减少 3.60%，F1 为 63.73% 对 63.23%，EM 为 52% 对 51%。只有 5 道题采用了短证据。D 与 C 的证据、答案、用量逐题相同。C 相对纯原文 A 多 0.14% 总在线 token。数据和统计范围见[报告](ragflow_abcd_2026-11-15/组件设计与实验结果汇报.md)及[汇总 JSON](ragflow_abcd_2026-11-15/results/summary.json)。

## 回放 100 题

用 Python 3.11 安装 `ragflow_abcd_2026-11-15/requirements.lock`，将该实验目录复制到独立的工作目录。在实验目录中启动命令。以 Windows PowerShell 为例。

```powershell
python -m pip install -r requirements.lock
$env:PYTHONPATH = (Resolve-Path vendor).Path
```

可直接读取已有 `results/scored.jsonl` 和 `results/summary.json`。如要重新发出答题请求，在副本内按下列命令运行，并在环境中提供 `DASHSCOPE_API_KEY`。同一份检索快照可脱离运行中的 RAGFlow 回放。

```powershell
Rename-Item -LiteralPath results -NewName archived-results
New-Item -ItemType Directory -Path results
python run_abcd.py prepare
python -u run_abcd.py run
```

扩规模的 RAGFlow 数据卷在本地单独冷备。GitHub 版本保存代码、实验文件和建库断点说明。恢复运行依赖原 RAGFlow 实例或本地冷备。第三方数据与依赖见[来源与许可](THIRD_PARTY_NOTICES.md)。
