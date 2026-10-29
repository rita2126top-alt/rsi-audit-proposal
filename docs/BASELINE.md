# 基线、版本与证据

外部来源核实日期：2026-10-28；实现版本：**0.2.0**。此文件区分原论文报告、实际核实的版本/数据、已有 CPU 软件验证和尚未运行的 Docker/GPU 研究结果。

## 已报告成绩

[Qwen2.5-Coder Technical Report v3](https://arxiv.org/html/2409.12186v3#S7.T16)，2024-11-12，第 7.1 节表 16，对 Qwen2.5-Coder-7B-Instruct 报告：

| 完整数据集 | 原论文报告 |
| --- | ---: |
| HumanEval | 88.4% |
| HumanEval+ | 84.1% |
| MBPP | 83.5% |
| MBPP+ | 71.7% |

这些数字仅作可追溯参考，不是 RSI-Audit 的成绩、留出分区基线或可保证的复现目标。主实验两次调用上限和分区与原始单次生成并不相同。第一周只在开发分区测量本地 B0；正式审计分区的 B0 和所有其他版本统一在搜索冻结后评估。

## 已固定的模型与上游来源

| 组件 | 固定值 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| 权重及 tokenizer revision | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| 模型卡 | [固定 revision](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/c03e6d358207e414f1eca0bb1891e29f1db0e242/README.md) |
| 数据恢复与 oracle 语义来源 | [evalplus/evalplus](https://github.com/evalplus/evalplus) |
| 上游 tag | `v0.3.1` |
| 上游 commit | `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2` |
| HumanEval+ | `v0.1.10`，164 道任务 |
| MBPP+ | `v0.2.0`，378 道任务 |

## v0.2.0 实际实现与未验证项

| 项目 | 已实现约束 | 当前验证状态 |
| --- | --- | --- |
| 推理后端 | `transformers==4.51.3`、`torch==2.6.0`；版本不符即报错 | 真实 GPU 加载与生成未运行 |
| 加载方式 | 固定 40 位模型/tokenizer revision；`local_files_only=True`、`trust_remote_code=False`、safetensors；要求 CUDA | 离线调用逻辑已有；GPU 环境待验证 |
| 推理精度与调度 | bfloat16，单 GPU 串行生成；无批量推理实现 | 无实测吞吐或数值一致性结论 |
| Prompt adapter | 固定 tokenizer 的 chat template；策略和执行约定进入 system，当前题面进入 user | CPU 提示构造与预算测试已有 |
| 生成/修复 | greedy solver；提议器 temperature 0.7、top_p 0.95；至多两次 solver 调用、总输入 8192/输出 2048 token | CPU 受控 backend 测试已有，不是模型质量测量 |
| 程序提取 | 第一段 Python/py 代码块，未找到则取原始文本；主机 AST 检查后交 Docker | 本项目固定规则，不等同上游 sanitizer |
| 候选运行 | Docker worker，仅 Python 标准库；单次任务批次 30 秒、512 MiB、1 CPU | 命令/协议 CPU 测试通过；真实 Docker 集成跳过 |
| 计分 | 主机比较 wire 编码参考值与候选值，包含固定上游特殊 oracle | CPU 构造案例测试通过；全量 canonical 兼容尚未验证 |
| 环境身份 | 运行记录不可变 Docker image ID、模型 revision、软件版本、CUDA runtime/GPU 名称 | 当前没有已验证的目标镜像摘要、GPU 驱动或完整依赖锁 |

模型使用冻结的 tokenizer/chat template 与模型默认 EOS 行为，没有额外自定义停止字符串。`repair_order` 只控制修复提示中的诊断块先后；执行语法检查始终先于公开示例。结构化公开示例只从可安全理解的题面断言/doctest 提取，不用隐藏结果修复。

项目要求主机 Python ≥3.10；基础 NumPy 依赖范围为 `>=1.26,<3`，真实运行会记录具体版本，但这还不是完整环境锁。Dockerfile 当前基于 `python:3.12-slim` 标签，构建前仍须登记基础镜像摘要；构建后执行器解析并记录不可变 image ID。主机 Python、CUDA/驱动、完整依赖解析和目标容器须在试跑中验证并归档，不能把代码中的版本要求写成已完成的 GPU 复现。

## 历史 recipe 与本项目设置的差异

Qwen 官方历史路径 `qwencoder-eval/instruct/eval_plus/` 的最后编辑 commit 为 `cd315a6a969b87fb0a75d59d51f8468ea856514c`；原仓库链接现重定向至 Qwen3-Coder，但这里研究的仍是固定 Qwen2.5 模型。

该 recipe 的 requirements 写 `evalplus==0.2.1`，vendored EvalPlus commit 为 `899b2b31bbe8d6e12337b865a8aa03fcd57c1121`。可核查脚本使用 greedy、temperature=0、n_samples=1、bf16、Qwen ChatML。原报告并未完整绑定所有环境与权重快照，因此不能断言 v0.3.1 必然逐位复现 84.1%。

本项目按上游固定 commit 还原数据类型和特殊 oracle，自行实现 Docker worker 与可信主机计分，没有原样运行 `evalplus.evaluate`。30 秒是一次任务全部调用与容器开销的墙钟限额，上游按参考耗时设定的规则与之不同；内存、可用依赖、代码提取和两次调用修复也存在差异。特殊输出兼容范围、fail-closed 行为与限制见 [EXECUTION_SECURITY.md](EXECUTION_SECURITY.md)。这些差异都可能改变通过率。

本项目必须在同一实现下重跑 B0，记录与原报告的差值及可识别配置差异，不借原数字替代新数据上的对照。首次全量模型参考复现若包含正式审计题，必须延至所有搜索冻结；此前只在开发分区检查模型行为。可信准备阶段对全部参考代码预计算期望输出不属于模型 B0 评测，但其全量 Docker 兼容也必须先验证。

## 数据核验

以下公开 release 文件已实际读取，完成 SHA-256、任务计数、加载与划分相关 CPU 校验；没有执行 GPU 模型研究，也没有完成全量 Docker canonical 预计算。完整校验字段见 [baseline-lock.json](../evidence/baseline-lock.json)。原始测试、参考代码和预计算期望输出存于运行者生成的私有 bundle，不提交到本仓库；可信协调器在搜索时读取开发/确认内容，在 freeze 后读取审计内容。因此不能笼统声称“整个搜索进程无法接触测试”。

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
