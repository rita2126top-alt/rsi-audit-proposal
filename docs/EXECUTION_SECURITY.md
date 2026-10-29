# 候选代码执行与可信计分

本实现把候选 Python 放在独立 Docker 容器内运行，把正确性判定留在可信主机进程。它提供可检查的工程隔离，**没有证明容器不可逃逸，也没有完成针对恶意 Python 的安全认证**。容器隔离依赖主机内核、Docker daemon 和运行时配置；研究实验适合使用可重置的专用机器。高价值凭据和其他项目数据不应成为该执行主机的工作负载。Docker 官方对这一边界的说明见 [Docker Engine security](https://docs.docker.com/engine/security/)。

## 数据和权限边界

`DockerExecutor.execute(code, entry_point, inputs)` 发给 worker 的请求严格只有三个字段：代码、入口函数名、参数。`expected`、判分规则、实验分组、晋升阈值、研究日志和任何“已通过”标记均不进入这个请求。容器没有宿主目录挂载、Docker socket、GPU 设备或继承宿主凭据的环境变量。

候选函数在容器 supervisor 派生的子进程中执行。普通 stdout/stderr 指向 `/dev/null`，supervisor 通过独立结果管道接收值，并输出有界 JSON 响应。主机只接受明确的字段与类型；`print('{"passed": true}')`、多余响应字段、缺失输出、非法类型标签均不能代替正确性比较。即使恶意候选设法伪造容器内的“输出值”，它也没有获得标准答案；这些值仍须在主机与独立保存的 canonical 输出比较。这里不把容器内可由候选影响的状态当作可信分数。

`wire.encode` / `wire.decode` 使用显式 JSON 类型标签保留 `tuple`、`set`、`frozenset`、`dict`、`complex` 和非有限浮点数。不使用 pickle，不根据字符串导入类，也不执行表达式。字典键和集合元素的编码顺序确定；循环、非法形状、重复字典键、过深结构和过大值被拒绝。主机解码仅构造已允许的内建类型及一个限定用途的 `non_none` 哨兵。候选代码不会在主机 `exec`、`eval` 或以其他回退路径执行。

上述隔离覆盖候选 Python 的执行。搜索过程能看到哪些任务、最终 audit 何时解封，以及运行日志是否被冻结，由上层分区与实验控制流程负责；仅启动一个容器不能证明这些研究协议已经满足。

## 镜像身份与资源限制

初始化执行器时先检查 Docker CLI/daemon，再将本地镜像引用解析为 `sha256:...` image ID。之后的 `docker run` 使用该不可变 ID，并设置 `--pull never`。`executor.identity` 用于实验元数据。协议 label 检查接口版本，**不证明镜像来源可信**；应从本仓库 Dockerfile 和受审阅源码构建镜像。跨机器复现需要保留该镜像或可获取的 registry digest。构建时还可以通过 `PYTHON_BASE` 固定经过核实的基础镜像 digest。

每次 `execute` 启动一个新容器，默认限制如下；实际值以记录的运行配置为准。

| 范围 | 实现 |
|---|---|
| 网络与 IPC | `--network none`、`--ipc none` |
| 文件系统 | `--read-only`；唯一主动配置的可写空间为容器自己的 16 MiB `/tmp` tmpfs，带 `noexec,nosuid,nodev`；没有宿主 bind mount |
| 身份和权限 | UID/GID `65534:65534`、`--cap-drop ALL`、`no-new-privileges=true` |
| CPU、内存、进程 | 默认 1 CPU、512 MiB；`--memory-swap` 与 memory 相同；最多 32 个进程；不启用 privileged 或宿主 PID namespace |
| 文件与描述符 | 64 个文件描述符、单个文件大小上限 1 MiB、core dump 关闭 |
| 源码和输入 | 源码最多 256 KiB；1–4096 个调用记录；每条最多 128 个位置/关键字参数；请求最多 16 MiB |
| 结果通道 | stdout 响应最多 16 MiB，Docker stderr 最多 64 KiB；禁用 Docker 日志驱动；主机流式读取并检查字节预算 |
| 参数和值结构 | 执行请求/单结果最多 1,000,000 个 wire 节点，深度最多 64；额外字符串、整数长度限制见 `wire.py` |

这些选项的含义见 [Docker run reference](https://docs.docker.com/reference/cli/docker/container/run/) 和 [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)。资源选项依赖运行主机的内核与 cgroup 支持；本次没有在实际 Docker 主机上验证它们的执行效果。

默认 `timeout_seconds=30` 是**一次任务批次**的宿主墙钟期限，包括容器启动、所有输入调用及响应传输。它不是“每条测试 30 秒”。到期、输出超限、异常或中断后，执行器终止客户端并在 `finally` 中执行 `docker rm --force`。清理失败会抛 `RuntimeError` 并中止研究；不会给一个不确定的模型失败分数。返回的 `wall_seconds` 包含清理，所以可能略超过任务期限。`cpu_seconds` 来自容器 supervisor 对子进程的资源统计，外部强制终止时为 `null`；它不是 GPU 时间，也不是完整宿主资源账单。

本执行器不创建跨任务并行调度，也不实现全局并发信号量。外部若并行发起 N 次执行，将创建 N 个各有限额的容器，主机累计资源可能接近这些限额的和。冻结实验配置时应记录调用方的并发设置。

## Canonical 输出预计算接口

准备数据时，canonical 程序使用同一个容器执行路径。标准答案仅由这一独立准备步骤产生，然后保存在可信侧：

```python
execution = executor.execute(canonical_code, task["entry_point"], task["inputs"])
if execution["status"] != "ok":
    raise RuntimeError("canonical execution failed; stop preparation")

task["expected"] = execution["outputs"]
verification = score_outputs(task, execution)
if not verification["passed"]:
    raise RuntimeError("canonical oracle verification failed; stop preparation")
```

`execution["outputs"]` 已经是 wire 编码，赋给 `task["expected"]` 时不要再次编码每个值。将完整 task bundle 落盘时，可以对整个 task 再调用 `wire.encode`；读回整个 task 后，`expected` 仍是原本的 wire 编码输出列表。

超时、错误 canonical 代码、缺少输入、无法表示的 canonical 输出、unsupported oracle 都必须阻止 prepare 完成。不得删除这些任务后继续声称使用了完整固定数据集。预计算的整个 expected 列表、输入次序、模型无关评测配置和数据版本应共同冻结，candidate worker 只获得其中的输入部分。

## EvalPlus oracle 的兼容范围

比较规则依据固定提交 `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2` 的三个一手源码文件：

- [eval/__init__.py](https://github.com/evalplus/evalplus/blob/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2/evalplus/eval/__init__.py)：等值、容差、特殊 oracle 的调用顺序。
- [eval/_special_oracle.py](https://github.com/evalplus/evalplus/blob/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2/evalplus/eval/_special_oracle.py)：特殊任务列表及替代定义。
- [gen/util/__init__.py](https://github.com/evalplus/evalplus/blob/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2/evalplus/gen/util/__init__.py)：canonical 的 `output_not_none` 转换。

本项目重新实现了数据边界和宿主比较，没有把 `evalplus.evaluate` 或其 Python reliability guard 当作沙箱。

| 规则或任务 | 当前实现 |
|---|---|
| 普通输出 | 首先采用内建类型的 Python 等值语义；不把 tuple 无条件转换成 list |
| 数值容差 | 对类型相同的输出使用主机 `numpy.allclose(..., rtol=1e-7, atol=...)`；序列外层长度须一致；保留上游对 float 参考值引入 `1e-6` 默认 atol 及其跨测试延续行为 |
| HumanEval `find_zero` / 32 | 按输入多项式的残差和任务 atol 判断，允许不同合法根；不要求与 canonical 返回的那个根相等 |
| MBPP 8 个集合等价函数 | `similar_elements`、`find_char_long`、`common_in_nested_lists`、`extract_singly`、`larg_nnum`、`intersection_array`、`find_dissimilar`、`Diff` 使用集合等价规则 |
| MBPP `surface_Area` / 581 | 原答案或上游“垂直高度”的替代面积定义均可；保留四舍五入和 atol 规则 |
| MBPP `digit_distance_nums` / 558 | 原答案或补齐前导零后的逐位距离均可；先检查原答案，避免替代计算改变已正确的结果 |
| MBPP 3 个非 None 任务 | `check_str` / 737、`text_match_three` / 787、`text_starta_endb` / 794：canonical 转为是否非 None；候选 bool 按 bool 比较，其他输出按是否非 None 比较 |
| 旧任务 `sum_div` / 295 | 保留上游接受原答案或 0 的规则；此题已不在所固定的 MBPP+ v0.2.0 中 |
| 旧任务 `are_equivalent` / 164 | 明确保留上游任何正常返回值都接受的规则；这是一个无判别力的旧 oracle，不能作为抗作弊证据，此题也不在 MBPP+ v0.2.0 中 |

三个非 None 任务的真实固定 canonical 实现均返回 `re.search(...)` 的 Match 或 `None`。Match 不在主机重建，而编码为 `non_none`。因此，“canonical execution → expected 原样保存 → 宿主验证”的接口与上游 `trusted_exec(output_not_none=True)` 的含义一致，无需调用方额外转 bool。普通可序列化返回值仍保留原值，便于公开示例评分；只有这些特殊入口的不可序列化非 None 结果才使用哨兵。`source='generic'` 不接受不明哨兵作为普通正确答案。

兼容性边界是明确的：worker 只带 Python 标准库，任意用户类、生成器、第三方数组对象不传回主机；代码导入未安装第三方包会失败。固定两份数据的 canonical 源码所需导入均为标准库，但尚未完成全部 canonical 的 Docker 实跑。canonical NaN 返回 `unsupported_oracle`；候选 NaN 对普通有限参考值判错，有符号 infinity 只有满足比较语义才通过。缺少 NumPy 且需要容差比较时返回 `unsupported_numeric_backend`。这些 unsupported 状态必须停止并报告，不能悄悄改成精确等值或从分母剔除。

宿主的任务总超时和标准库运行环境与上游逐测试 reference-time 倍率及其依赖环境不同。因此这里声称的是**已实现并逐项测试的 oracle 规则及明确限制**，没有声称运行环境或论文成绩与上游逐位一致。相关源码的许可证和改写说明应随分发保留；上游仓库为 Apache-2.0，`evalplus/eval/` 另含 OpenAI HumanEval 的 MIT notice。

## 本次验证与可复现测试命令

本模块本次运行了 28 个 CPU 单元测试，均通过：包括 wire 类型/大小/形状边界、不可变镜像命令构造、请求字段隔离、超时清理路径、基础设施错误、拒绝自报通过，以及上述全部特殊 oracle 和数值比较。执行器的 CPU 测试使用受控 Docker 命令替身，**不能证明实际容器隔离已经生效**。

另有 4 个包含真实断言的 Docker 集成测试，覆盖真实执行与 stdout 伪造、类型传输与错误 arity、UID/只读文件系统/无网络、无限循环超时及容器清理。本次工作环境没有可用 Docker，且未配置测试镜像，**这 4 项被明确 skip，没有被记为通过**。也没有在本次环境执行完整 GPU 研究或全量 canonical 准备。

从仓库根目录，在专用的 Docker 主机上运行：

```bash
docker build -f docker/Dockerfile -t rsi-audit-worker:local .

RSI_AUDIT_TEST_IMAGE=rsi-audit-worker:local \
  python -m unittest discover -s tests -p 'test_execution.py' -v

python -m unittest discover -s tests -p 'test_scoring.py' -v
```

`RSI_AUDIT_TEST_IMAGE` 显式选择本地已构建镜像。测试不会代为拉取镜像或把候选源码改在主机执行。配置了镜像后，daemon、镜像或 worker 启动错误会让测试失败；未配置时集成测试仍显示 skip。只有真实集成测试和 canonical 准备均成功后，才能冻结该执行环境并开始主研究。
