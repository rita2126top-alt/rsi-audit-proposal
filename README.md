# RSI-Audit

## 有限验证预算下，自改进代码智能体的可信评测

**TOKEN-IGNITION 方向：RSI 评测（`eval`） · 版本：0.2.0 · 申请账号：rita2126top-alt**

研究问题：代码智能体逐轮修改持久提示与修复策略时，在相同的验证机会和资源上限下，反复检查同一批题、每轮更换验证题、只在最后做一次验证，哪种策略能提高独立审计上的最终表现？

本仓库同时提供研究 proposal 和可运行的实验代码：数据校验与分区、受限策略搜索、离线模型适配器、Docker 执行器、外部计分、冻结与审计、成本记录和统计报告。

**当前状态：CPU 单元测试与合成数据全流程已验证；真实 Docker 集成、全量参考程序执行、GPU 推理和正式研究尚未完成。合成演示不是性能证据，项目也尚未获得资助通过的确认。** 详细验证记录见 [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)。

## 先在 CPU 上运行

在 Linux / WSL 的仓库根目录执行。Python 3.12 是本次验证环境；离线演示不需要模型、GPU 或 Docker。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m unittest discover -s tests -v
python -m rsi_audit demo --out private-runs/demo-01
python -m rsi_audit verify --run private-runs/demo-01/run
```

演示会实际运行协调器的三组策略、3个种子和3轮搜索，然后冻结、审计16道合成任务、生成配对统计。输出目录包含 `bundle/`、`run/`、`report/`；可打开 `report/report.md` 和 `report/analysis.json`。分数来自明确标注的确定性测试夹具，模型调用数和 GPU 卡时均为零。输出目录不能重复使用。

已生成的展示样例在 [examples/synthetic-report.md](examples/synthetic-report.md)。其中所有数值仅展示报告格式。

## 代码已经实现什么

| 模块 | 实际行为 |
| --- | --- |
| [data.py](rsi_audit/data.py) | 验证固定数据归档双 SHA-256；恢复特殊输入类型；按题目族划分；保存私密数据和公开划分承诺；试跑仅取正式开发分区 |
| [policy.py](rsi_audit/policy.py) | 三字段严格 JSON；冻结 tokenizer 检查提示长度；拒绝额外工具、代码插件、预算修改和非法枚举 |
| [backend.py](rsi_audit/backend.py) | Qwen 固定完整 revision；Transformers 4.51.3 / PyTorch 2.6.0；离线 safetensors；真实模式要求 CUDA |
| [solver.py](rsi_audit/solver.py) | 最多两次调用、累计 token 上限；只有语法和题面公开示例能触发修复；隐藏评分不回流 |
| [engine.py](rsi_audit/engine.py) | Reuse / Fresh / Terminal 搜索；统一精确缓存；逐轮 checkpoint；完成所有组后冻结；冻结后拒绝继续搜索 |
| [execution.py](rsi_audit/execution.py)、[worker.py](docker/worker.py) | 候选代码只在无网络、只读根文件系统、有限资源的 Docker 内执行；没有主机执行回退 |
| [scoring.py](rsi_audit/scoring.py) | 宿主计分器持有 expected，比较受限协议输出；兼容固定 EvalPlus 特殊 oracle；不采信候选自报分数 |
| [ledger.py](rsi_audit/ledger.py) | 哈希链与 HMAC 记录、状态锚点和冻结清单；区分缓存的逻辑额度与物理消耗；异常中断保留记录 |
| [analysis.py](rsi_audit/analysis.py) | 只比较最终部署版本；来源分层、种子与任务族交叉配对 bootstrap；晋升偏差、描述性退步、成本报告 |
| [cli.py](rsi_audit/cli.py) | `doctor / prepare / pilot-bundle / search / freeze / audit / analyze / verify / demo` |

`repair_order` 控制修复提示中两类公开诊断块的先后顺序；运行代码之前始终先检查语法。策略不能改动计分、资源限制或数据访问方式。

## 在 GPU 主机上执行真实试跑

完整安装、数据准备和运行命令见 [RUN_GUIDE.md](docs/RUN_GUIDE.md)。先运行 `doctor`，再完成 Docker 集成检查与 canonical 输出预计算。模型下载是单独的显式命令，不会在搜索时自动联网。

```bash
python -m rsi_audit doctor
python -m pip install -e '.[gpu]'
docker build -f docker/Dockerfile -t rsi-audit-worker:0.2.0 .

RSI_AUDIT_TEST_IMAGE=rsi-audit-worker:0.2.0 \
  python -m unittest discover -s tests -p test_execution.py -v

python tools/download_assets.py --output private-assets
python tools/prefetch_model.py --cache-dir private-model --download

python -m rsi_audit prepare --assets private-assets --out private-bundle \
  --image rsi-audit-worker:0.2.0
python -m rsi_audit pilot-bundle --bundle private-bundle --out private-pilot

python -m rsi_audit search --mode pilot --bundle private-pilot \
  --run private-runs/pilot-01 --image rsi-audit-worker:0.2.0 --model-cache private-model
python -m rsi_audit freeze --run private-runs/pilot-01
python -m rsi_audit audit --run private-runs/pilot-01 --bundle private-pilot \
  --image rsi-audit-worker:0.2.0 --model-cache private-model
python -m rsi_audit analyze --run private-runs/pilot-01 --out private-runs/pilot-01-report
```

试跑使用从主开发分区整族抽出的40题：dev16、两块 confirmation 各8题、audit8；三组各2种子×2轮。它不能替代正式研究。所有 Docker 结果和 GPU 吞吐都需要在该机器上实际确认。

正式设置见 [configs/study.json](configs/study.json)：Qwen2.5-Coder-7B-Instruct，542道原始任务，目标128/192/222分区，三组各6种子×6轮，每轮2个候选。先完成试跑、核对预算、外部登记配置与数据承诺，再在新目录以 `--mode formal` 执行。正式审计只能在全部搜索冻结后启动。

## 研究边界与实现决定

- 主比较是最终版本的 Fresh − Reuse，不能选择历史最高审计分。Terminal 的开发更新必须严格改善，平分保留父版本。
- 同预算指相同逻辑评测额度和 token 上限。实际 token、缓存命中及 GPU 占用分别记录；不为配平而空耗 GPU。
- 可信协调器持有数据和日志，提议模型只接收开发反馈。候选程序的隔离由 Docker 提供；主机管理员与协调器属于信任边界，哈希/HMAC 不是远程执行证明。
- 当前统计实现对历史晋升给出描述性差异，**不提供原草案计划的 Holm 调整支持标签**；此修订已经同步至实验协议。主区间不构成无退步保证。
- 阶段时间检查和模型 `max_time` 是软门禁。硬运行时限需要外部调度器；所有试跑、准备、失败与新建 run 的资源应一起核对480卡时总申请预算。
- Docker 集成、参考程序全量兼容性与真实 GPU 数值仍待验证。公开旧题也不能证明模型训练数据无污染。

## Proposal 与引用

| 文档 | 用途 |
| --- | --- |
| [PROPOSAL.zh-CN.md](PROPOSAL.zh-CN.md) | 研究问题、价值、方法、计划与验收标准 |
| [SUBMISSION.zh-CN.md](SUBMISSION.zh-CN.md) | 官网七项填写稿；私密联系方式保持留空 |
| [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | 三组规则、终态选择、统计及实现修订 |
| [DATA_PREPARATION.md](docs/DATA_PREPARATION.md) | 数据版本、整族隔离、准备与试跑抽样 |
| [EXECUTION_SECURITY.md](docs/EXECUTION_SECURITY.md) | 候选执行边界、特殊 oracle 与上游差异 |
| [ANALYSIS.md](docs/ANALYSIS.md) | 估计量、重采样、成本与缺失结果规则 |
| [BASELINE.md](docs/BASELINE.md)、[COMPUTE.md](docs/COMPUTE.md) | 固定版本、外部基线及预算推导 |
| [RELATED_WORK.md](docs/RELATED_WORK.md)、[references.bib](references.bib) | 9篇已核实论文及具体关系 |
| [CITATION.cff](CITATION.cff) | 引用本项目代码与研究协议；不表示已有发表论文 |

```bash
python tools/validate_package.py
```

该检查核对申请字段、文献键、链接与计划预算，不运行 GPU 实验。原创贡献采用 MIT；改编的第三方规则、数据和模型见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。仓库不包含模型权重、基准答案、真实私密实验数据或私人联系方式。
