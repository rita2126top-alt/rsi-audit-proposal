# RSI-Audit 运行指南

本指南对应仓库 `0.2.0` 实现，命令从仓库根目录执行。它覆盖离线演示、可信数据准备、7 天开发集试跑，以及 28 天研究期内的正式搜索和独立审计。**实现交付不等于已获得研究结果：本次交付没有下载模型权重，没有完成 Docker/GPU 正式实验，也没有实测吞吐量或效果提升。**

实验定义见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)、[study.json](../configs/study.json)；数据语义和保密边界见 [DATA_PREPARATION.md](DATA_PREPARATION.md)；统计输出的实际含义见 [ANALYSIS.md](ANALYSIS.md)。

## 1. 环境检查与离线演示

完整研究需要 Linux、Python 3.10 或更新版本、可用的 NVIDIA CUDA GPU，以及能够启动受限容器的 Docker 守护进程。真实推理后端严格要求 `torch==2.6.0`、`transformers==4.51.3`，使用 BF16；缺少 CUDA 时直接失败。具体依赖以 [pyproject.toml](../pyproject.toml) 为准。

先建立独立环境并检查依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m rsi_audit doctor
```

`doctor` 返回包版本、Docker CLI 路径、CUDA 可用性等信息。它不会启动研究，也不会认证 Docker 守护进程、容器隔离或模型正确性。初始开发环境没有 Docker CLI；正式容器执行仍需在目标机器验证。

无需模型、GPU、Docker 或联网即可运行合成演示：

```bash
python -m rsi_audit demo --out private-runs/demo-01
python -m rsi_audit verify --run private-runs/demo-01/run
```

演示输出分别位于 `private-runs/demo-01/bundle`、`run`、`report`。其中 `report/report.md` 和 `report/analysis.json` 演示完整的搜索、冻结、审计与报告流程。40 条记录都明确标为 `source="synthetic"`；分数来自人工测试夹具，不代表生成代码的能力，也不能用于论文效果表或申请中的实验结果。

## 2. 下载锁定的数据与模型

联网准备与后续离线实验分开执行。先安装真实推理依赖：

```bash
python -m pip install -e '.[gpu]'
python -m rsi_audit doctor
```

### 两份 EvalPlus 数据

```bash
python tools/download_assets.py --output private-assets
```

下载工具只接受 [baseline-lock.json](../evidence/baseline-lock.json) 中锁定的 EvalPlus 发布地址，检查压缩文件及解压 JSONL 的 SHA-256、字节数和题数。目录必须是新目录。正式输入文件名为 `HumanEvalPlus.jsonl.gz` 与 `MbppPlus.jsonl.gz`，共 542 题。已有归档可以直接放入同名目录，但仍须通过 `prepare` 的哈希检查；不要用最新版数据替换锁定版本。

### 精确模型版本

模型固定为 `Qwen/Qwen2.5-Coder-7B-Instruct`，提交为：

```text
c03e6d358207e414f1eca0bb1891e29f1db0e242
```

先打印离线下载计划；这条命令不会联网、创建缓存或下载任何文件：

```bash
python tools/prefetch_model.py --cache-dir private-model
```

在已准备好磁盘与网络的机器上，显式加 `--download` 才会下载：

```bash
python tools/prefetch_model.py --cache-dir private-model --download
```

工具从锁文件读取模型名称和完整提交 SHA，确认 Hub 返回相同提交；只下载必要配置、tokenizer 文件、safetensors 索引及索引指定的 safetensors 分片。它不下载 pickle 权重或远程 Python 文件，不加载模型，不发送已缓存的 Hugging Face 用户凭证。下载后核对远端内容哈希，生成 `private-model/rsi-prefetch-manifest.json`，记录每个文件的字节数、SHA-256、精确提交和下载工具版本。

该固定版本的[官方文件目录](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242)列出约 15.2 GB 文件；磁盘还需容纳缓存、环境与实验输出。工具使用标准 Hugging Face **缓存根目录**，后续 `--model-cache` 应填写 `private-model`，不要改填内部 `snapshots/...` 目录。中断后的同版本下载可以再次执行；已有目录必须包含工具写入的同模型、同提交标记，否则拒绝复用。

真实后端仅从本地缓存加载，设置 `local_files_only=True`、`trust_remote_code=False`、`use_safetensors=True`，不会在实验中隐式下载缺失文件。下载机制参考 Hugging Face 的[官方缓存和筛选下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)。

## 3. 构建受限执行容器

```bash
docker build -f docker/Dockerfile -t rsi-audit-worker:0.2.0 .
```

该命令适合首次环境验证。正式冻结前，应按 [Dockerfile](../docker/Dockerfile) 注释给 `PYTHON_BASE` 提供已确认的镜像 digest，并保存构建记录。执行器会把最终镜像解析成不可变 image ID，运行身份包含该 ID；同一研究中搜索和审计必须使用相同执行器身份。

参考实现和模型生成的代码均在受限容器内执行。模型推理在宿主 GPU 进行。不要将完整私密数据目录、运行目录或审计分区挂载给候选代码；执行器按次发送完成当前调用所必需的数据。容器的具体限制以 [execution.py](../rsi_audit/execution.py) 和冻结的运行配置为准。

## 4. 可信准备：参考输出、分区与公开承诺

```bash
python -m rsi_audit prepare \
  --assets private-assets \
  --out private-bundle \
  --image rsi-audit-worker:0.2.0 \
  --seed 20261028 \
  --config configs/study.json
```

`prepare` 完整验证锁定数据，在 Docker 内逐题执行 canonical reference，预计算所有输入的期望输出并做评分器自检，再按整族规则分区。目标是开发 128 题、6 个各 32 题的验证分片、剩余 222 题正式审计。人工检查得到的近重复映射可通过 `--family-map reviewed-families.json` 提供，格式为任务 ID 到共同家族标签的 JSON 映射。自动精确重复检测不能保证语义去重。

**所有题目的 canonical 输出预计算属于可信准备阶段。** 因此这个阶段可以读取将来的审计题目；其输出用于可信评分器，不提供给搜索模型。模型生成只接收题面和从题面提取的公开示例，不能收到 canonical reference、私有测试输入或期望输出。搜索期间不加载正式 `audit.json`；正式审计须等全部搜索运行冻结以后开始。

任何 canonical 执行失败、类型无法支持、哈希不符或整族精确分配不可行都会中止准备，不会静默删题。需要修改运行限制、数据解释或分配规模时，应先记录修订、重新准备，再冻结新研究。

准备阶段资源写入 `private-bundle/metadata.private.json` 的 `preparation`：墙钟时间、执行器累计运行时间、题数和本步骤 GPU 小时。当前准备过程不做模型推理，因此 GPU 小时为 0；`cpu_seconds` 目前为 `null`，不要把墙钟时间冒充 CPU 时间。模型下载、环境安装、镜像构建以及进程启动失败的额外资源另行记录；它们不自动包含在搜索账本内。

完成后保存 `private-bundle/public_manifest.json` 的外部时间戳，例如在正式搜索前将这一份公开清单提交到单独的证据位置或 Git 提交。清单只含计数和哈希承诺，不含审计题目 ID、题面或盐。以下文件保持私密：`dev.json`、`panels.json`、`audit.json`、`secret.private.json`、`metadata.private.json`。

目录权限为 `0700`、文件权限为 `0600`，它们不能隔离同一操作系统用户控制的进程。可信协调器负责访问边界；外部时间戳负责证明承诺先于实验存在。本地哈希检查本身不能防止管理员同时替换全部文件。

## 5. 7 天试跑：仅从主开发集取 40 题

```bash
python -m rsi_audit pilot-bundle \
  --bundle private-bundle \
  --out private-pilot \
  --seed 20261028
```

该命令从主开发集按整族规则联合选出 16 题开发、2 个各 8 题的验证分片和 8 题试跑审计，共 40 题。没有选中的主开发集家族不进入试跑。CLI 读取父 bundle 的已认证准备元数据以继承 canonical oracle 身份与限制，但不会读取父验证题或父审计题。

`--mode pilot` 自动冻结以下覆盖值，无需手动修改主配置：2 个种子 `[11, 29]`、2 轮、每轮 2 个候选、每个验证 panel 8 题、terminal 验证 16 题。reuse/fresh 门槛为净增 1 题，terminal 门槛为净增 2 题；试跑搜索和审计分别有 36 与 12 GPU 小时上限。

```bash
python -m rsi_audit search \
  --config configs/study.json \
  --bundle private-pilot \
  --run private-runs/pilot-01 \
  --mode pilot \
  --image rsi-audit-worker:0.2.0 \
  --model-cache private-model

python -m rsi_audit freeze --run private-runs/pilot-01

python -m rsi_audit audit \
  --run private-runs/pilot-01 \
  --bundle private-pilot \
  --image rsi-audit-worker:0.2.0 \
  --model-cache private-model

python -m rsi_audit analyze \
  --run private-runs/pilot-01 \
  --out private-reports/pilot-01

python -m rsi_audit verify --run private-runs/pilot-01
```

试跑用于确认容器执行语义、缓存行为、token 账本、显存和吞吐量、时间预算及审计闭环是否可行。8 题试跑审计分数只作流程调试，不能作正式效果结论。保留失败与资源记录；不要为了得到更好看的结果反复换试跑 seed。根据试跑确定必须的协议修订后，在正式搜索之前重新登记并锁定。

## 6. 正式搜索、冻结与独立审计

正式使用主 bundle，另开新的运行目录：

```bash
python -m rsi_audit search \
  --config configs/study.json \
  --bundle private-bundle \
  --run private-runs/formal-01 \
  --mode formal \
  --image rsi-audit-worker:0.2.0 \
  --model-cache private-model

python -m rsi_audit freeze --run private-runs/formal-01
```

初次搜索会冻结有效配置、源码指纹、模型/执行器身份、数据承诺和策略快照。冻结操作要求全部预定 arm/seed/round 完成；生成的 `freeze.json` 应在审计开始前另存外部时间戳。不要在搜索、冻结和审计之间更新本地实现、配置、模型版本或 worker 镜像。

```bash
python -m rsi_audit audit \
  --run private-runs/formal-01 \
  --bundle private-bundle \
  --image rsi-audit-worker:0.2.0 \
  --model-cache private-model

python -m rsi_audit analyze \
  --run private-runs/formal-01 \
  --out private-reports/formal-01

python -m rsi_audit verify --run private-runs/formal-01
```

审计在所有搜索 checkpoint 已固定后才读取正式审计分区；审计结果不反馈给 proposer，也不用于挑选历史最优 checkpoint。主要对比使用每个 seed 最终部署策略的 **fresh − reuse 最终审计 solve rate**。验证增益减去审计增益的 optimism gap，以及历史晋升的审计退步是诊断性输出。

`analyze` 生成 `report.md` 和结构化 `analysis.json`。当前实现提供按 seed/题目家族配对的 bootstrap 区间，对晋升退步只标记样本中的观察结果，不提供多重比较校正后的“确定退步”认证。低功效下不显著不能解释为无效果；区间也不是无退步保证。需要新增推断规则时，应先修订预注册并实现验证，而不是在看过结果后改变判定。

`verify` 检查本地运行的身份、配置、账本链、策略、冻结承诺与审计文件一致性，不认证真实硬件执行或研究结论。完整报告还应同时提供准备阶段资源、外部时间戳、固定源码版本和模型下载清单。

## 7. 预算、恢复与发布边界

申报资源为单张 A100 80 GB 口径，试跑 48 GPU 小时、全程最多 480 GPU 小时并包含试跑；这是申请上限，尚未通过实测证明足够。当前运行配置给正式搜索 240、正式审计 144 GPU 小时上限，额外资源仍需在总额度内统筹。GPU 小时按记录区间的墙钟时间乘 GPU 数计算；token 总数、解码秒数或“命中缓存”次数不能直接代替 GPU 小时。参见 [COMPUTE.md](COMPUTE.md)。

正常返回后的未冻结搜索可以用相同命令继续，配置、源码与后端身份必须一致。`command.inprogress` 现在保留持久状态：正常结束写入 completed，并绑定账本最后一条经过HMAC认证的完成记录；仅文件存在不代表中断。若硬中断留下 active 或完成记录不匹配，执行器拒绝自动重放，以免发生未计费调用或状态错配；保留该运行和资源证据，另行登记恢复方案。达到阶段预算时不会把剩余未执行记录补成失败，也不能把不完整运行写成完整正式研究。

运行目录与数据目录保持私密。论文和开源材料发布时应区分合成演示、开发集试跑与正式实验，给出实际完整性、资源覆盖范围和失败情况。公开协议、代码、分析脚本、引用及允许再分发的结果；任何公开审计任务与盐的动作都应发生在正式研究终止后，并遵循原数据许可。不要把缓存权重、私密日志或未审计的数据目录一起加入 Git。

## 常见失败及处理

正式分析的重采样次数与随机种子已在创建 run 时写入配置（默认2000次、20261028），CLI 拒绝使用与冻结设置不符的正式分析参数。已完成的 search 再次调用会直接返回状态，不重新加载模型。

| 现象 | 检查与处理 |
| --- | --- |
| `docker_cli` 为空或容器无法启动 | 在目标机器安装并检查 Docker；演示可以先运行，真实代码执行不能跳过容器。 |
| CUDA 不可用或依赖版本不符 | 检查 `doctor` 与 GPU 驱动，安装锁定版本；真实研究不提供静默 CPU 回退。 |
| 离线模型加载提示缺文件 | 用同一缓存根目录重新执行显式预下载，核对精确模型 SHA 和清单。 |
| 下载工具拒绝已有目录 | 归档下载需要新目录；模型缓存只能复用同模型同 SHA 的工具标记目录，不能覆盖其他缓存。 |
| 数据哈希、题数、类型或 oracle 自检失败 | 保留错误，核对锁定归档、执行器版本与输入语义；不可删去失败题继续研究。 |
| 精确整族分区不可行 | 复核人工家族映射和预注册规模，记录修订；不可把家族拆开或静默减少审计题。 |
| 冻结后身份或文件校验失败 | 使用原始源码、镜像、缓存和完整记录，不能修改已冻结文件来消除错误。 |
| 运行因预算或中断不完整 | 报告未完成范围和已消耗资源；缺失观测不作失败值填充，也不从分析中静默删除。 |
