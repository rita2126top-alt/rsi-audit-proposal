# 基线、版本与证据

核实日期：2026-10-28。此文件区分原论文报告、已核实版本信息和尚未运行的本项目结果。

## 已报告成绩

[Qwen2.5-Coder Technical Report v3](https://arxiv.org/html/2409.12186v3#S7.T16)，2024-11-12，第 7.1 节表 16，对 Qwen2.5-Coder-7B-Instruct 报告：

| 完整数据集 | 原论文报告 |
| --- | ---: |
| HumanEval | 88.4% |
| HumanEval+ | 84.1% |
| MBPP | 83.5% |
| MBPP+ | 71.7% |

这些数字仅作可追溯参考，不是 RSI-Audit 的成绩、留出分区基线或可保证的复现目标。主实验两次调用上限和分区与原始单次生成并不相同。第一周只在开发分区测量本地 B0；正式审计分区的 B0 和所有其他版本统一在搜索冻结后评估。

## 拟冻结版本

| 组件 | 固定值 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| 权重及 tokenizer revision | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| 模型卡 | [固定 revision](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/c03e6d358207e414f1eca0bb1891e29f1db0e242/README.md) |
| 评测源代码 | [evalplus/evalplus](https://github.com/evalplus/evalplus) |
| 评测 tag | `v0.3.1` |
| 评测 commit | `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2` |
| HumanEval+ | `v0.1.10`，164 道任务 |
| MBPP+ | `v0.2.0`，378 道任务 |

推理引擎、CUDA/驱动、Python 依赖解析结果、容器摘要、prompt adapter、停止字符串、sanitizer 和 timeout 策略将在试跑阶段兼容验证后一起固定。当前没有伪造一份已经验证可用的环境锁文件。

## 历史 recipe 与本项目设置的差异

Qwen 官方历史路径 `qwencoder-eval/instruct/eval_plus/` 的最后编辑 commit 为 `cd315a6a969b87fb0a75d59d51f8468ea856514c`；原仓库链接现重定向至 Qwen3-Coder，但这里研究的仍是固定 Qwen2.5 模型。

该 recipe 的 requirements 写 `evalplus==0.2.1`，vendored EvalPlus commit 为 `899b2b31bbe8d6e12337b865a8aa03fcd57c1121`。可核查脚本使用 greedy、temperature=0、n_samples=1、bf16、Qwen ChatML。原报告并未完整绑定所有环境与权重快照，因此不能断言 v0.3.1 必然逐位复现 84.1%。

本项目统一重跑 B0，记录与原报告的差值及可识别配置差异，不借原数字替代新数据上的对照。首次全量参考复现若包含正式审计题，必须延至所有搜索结束；在此之前仅检查开发分区。

## 数据核验

以下公开 release 文件已实际读取并仅用于任务计数和 SHA-256 核验；没有用它们运行模型或生成研究结果。完整校验字段见 [baseline-lock.json](../evidence/baseline-lock.json)。基准答案和测试不会打包进 proposal 仓库或搜索工作区。

| 数据文件 | 压缩文件 SHA-256 |
| --- | --- |
| [HumanEvalPlus.jsonl.gz](https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz) | `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101` |
| [MbppPlus.jsonl.gz](https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz) | `af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63` |

个别排行榜页面仍保留旧版 MBPP 题数；本申请使用下载核实的 378。542 是原始任务总数，不是已经完成去重的独立任务族数量。

## 对照命名

- `B0`：相同资源限制下的人工初始固定 harness，实际分区成绩待测。
- `Reuse`：固定面板与经验门槛，是简化受控策略，不能标成对完整 VALVE 的复现。
- `Fresh`：一次性分片与相同经验门槛。
- `Terminal`：开发阶段搜索后的一次较大确认。

论文里“reported reference”、本项目“measured B0”及待运行计划必须分列。任何表格都不预填假想性能提升。
