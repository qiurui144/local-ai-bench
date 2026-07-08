# K3 真实业务混合流量压力测试方案

本文定义 `vlm-llm-benchmark` 下一轮 K3 压力测试依据。它不是另一轮
ModelZoo 全矩阵跑分，而是模拟一个生产应用如何通过 scheduler / inference
gateway 使用 32 GB SpacemiT K3 节点：前台短检索、受控回答生成、文档/VLM 抽取、
异步后台摄入，以及共享同一个受限 A100 service 的健康检查与恢复操作。

English version: [k3-realistic-stress-plan.md](k3-realistic-stress-plan.md)

## 范围

### 被测系统

主要被测链路：

```text
product-like client -> k3-scheduler / inference gateway -> model workers
```

首选目标端点是 scheduler-facing API，而不是裸模型 server：

| 能力 | 首选路由 |
|---|---|
| 同步模型请求 | `POST /infer/{model}` |
| 异步模型请求 | `POST /infer/{model}:async` |
| 异步轮询 | `GET /jobs/{id}` |
| 健康检查 | `GET /ready` |
| 容量状态 | `GET /capacity` |
| 事件流 | `GET /events` |
| 指标 | `GET /metrics` |

裸 OpenAI-compatible 模型 server 可以作为对照组，但不足以代表本轮压力测试：
它们不会覆盖 scheduler 的资源准入、排队、异步任务、A100/TCM 恢复行为。

### 非目标

- 不把所有缓存的 ModelZoo 模型同时拉起来并发跑。
- 不把 20B+ 长上下文请求当普通同步流量处理。
- 不只测单模型 TPS；本轮关注混合服务稳定性和用户可见行为。
- 不在未记录的 dirty A100/TCM 状态下开始测试。stale TCM 是测试发现，不是合格基线。

## 真实业务流量模型

压力负载应是 trace mix，不是扁平 RPS benchmark。每个虚拟用户执行下列产品流之一：

| 流程 | service_class | 典型路径 | 期望行为 |
|---|---|---|---|
| 前台查询 | `realtime_retrieval` -> `realtime_answer` | query embedding -> vector-search stub -> rerank -> 短 LLM 回答 | 检索高优先级；回答受 deadline/token limit 约束 |
| 文档上传 | `realtime_vlm_compact` -> `background_index` | compact VLM 抽取 -> chunk -> embedding | 小文档可同步；较大工作转异步 |
| 重文档抽取 | `user_async_vlm` | balanced/high-spec VLM 文档抽取 | 快速返回 job；不能阻塞短检索 |
| 长上下文分析 | `long_context` | 20B+ 长上下文回答或审计 | 仅异步；必须有 TTL/cancel |
| 后台摄入 | `background_index` | 批量 embedding/rerank/summary | 只在短前台队列健康时吃空闲窗口 |
| ASR 笔记 / 会议片段 | `realtime_x100` 或 `user_async` | ASR -> 可选 summary | X100 预算不能饿死 scheduler/SSH/system |
| 控制 / 恢复 | `control` | ready/capacity/events/TCM health/unquarantine | 永远不被模型队列阻塞 |

向量数据库可以用 stub。重要压力来自模型服务序列和资源共享，而不是数据库排序质量。

## 第一轮模型角色

以 2026-07-06 K3 32 GB 校准结果作为起点。

| 角色 | 候选模型 | 调度规则 |
|---|---|---|
| 高规格 LLM primary | `Qwen3-30B-A3B-Q4_0` | 仅短回答可同步；其它默认异步 |
| 高吞吐 LLM 对照 | `LFM2-24B-A2B-Q4_0` | 性能对照；质量未合格前不做默认答案模型 |
| Compact VLM | `Qwen3.5-2B.tar.gz` | deadline 足够时可同步；否则异步 |
| Balanced VLM | `Qwen3.5-4B.tar.gz` | 默认异步 |
| High-spec VLM | `qwen30ba3b-mm-q4_1.tar.gz` | 仅异步 |
| VLM fallback/control | `Qwen3VL-4B + mmproj` | 异步 fallback/control |
| Embedding | BGE-Zh / Jina / Nomic / Qwen3-Embedding | `realtime_retrieval` |
| Reranker | `Bge-Reranker-V2-M3-Q4_0` | `realtime_retrieval` |
| ASR | `qwen3-asr-0.6B.tar.gz` | 按音频长度同步或异步 |

不要把“运行很快但质量未过”的 VLM 提升为生产文档抽取模型。例如 FastVLM 上轮很快，
但没有通过文档抽取质量。

## 负载级别

测试应按顺序执行所有级别。前一层健康、正确性或清理失败时，后一层结果无效。

### L0：预检与干净基线

时长：5-10 分钟。

必须检查：

- `GET /ready` 成功。
- `GET /capacity` 成功，并且没有意外 busy holder。
- 任何 A100 工作开始前记录 `spacemit-tcm-smi` snapshot。
- 如果存在 stale TCM block，记录并停止，除非本轮明确是恢复测试。
- 每个选定模型角色跑一个请求，成功或返回预期 unsupported/incompatible 错误。
- 没有计划 hot set 以外的模型被意外启动。

通过标准：

- Scheduler / inference gateway 保持可达。
- A100 要么干净，要么在压测开始前显式标记为 degraded。
- 没有 crash loop、重复 init storm 或无法解释的 worker restart。

### L1：单用户产品流

时长：15 分钟。

执行一个真实的顺序工作流：

1. 上传一张文档图像。
2. 运行 compact VLM 抽取。
3. 对抽取文本 chunk 并 embedding。
4. 执行一次前台查询。
5. rerank 候选片段。
6. 生成一个短且有界的回答。
7. 提交一个更重的 VLM 抽取为 async。
8. 轮询 async job 直到完成。

通过标准：

- 用户可见同步操作在配置 deadline 内完成。
- 异步接收快速返回 job id。
- job 完成后系统仍健康。
- 文档抽取质量保持在所选 compact VLM 的已校准预期范围内。

### L2：正常业务混合

时长：10 分钟 ramp 后运行 60 分钟。

建议到达流量比例：

| 流程 | 占比 | 初始速率目标 |
|---|---:|---|
| 前台查询 | 60% | 总计 2 requests/min |
| Compact VLM 文档上传 | 15% | 0.3 requests/min |
| Heavy VLM async | 5% | 0.05 requests/min |
| ASR 笔记 | 5% | 0.05 requests/min |
| 后台摄入 | 15% | 只填充空闲窗口 |

具体速率可在 L1 后调整，但混合形态要保持真实：短检索频繁，重 VLM 稀少，后台任务机会式运行。

通过标准：

- `control` 端点 p99 保持响应。
- 前台检索不被长 VLM/context 工作堵在后面。
- Heavy VLM 被接收为 async，或返回明确 backpressure 原因。
- 后台摄入只在前台队列健康时推进。
- 无 A100 quarantine、TCM stale leak、节点 hang、SSH/network 饥饿。

### L3：忙时突发

时长：30 分钟。

模式：

- 5 分钟 ramp 到 L2 前台查询到达率的 3 倍。
- 注入文档上传 burst：2 分钟内 3 个 compact VLM job。
- 注入 1 个 high-spec VLM async job。
- 保持后台摄入开启，但低优先级。

期望行为：

- 系统可以限流或返回 `queue_full`、`async_required`、`busy_retry`、
  `deadline_exceeded`。
- 系统不能让长 VLM 先于短检索执行。
- 系统不能把 backpressure 变成 worker crash。

通过标准：

- 无进程 restart storm。
- scheduler 不崩溃。
- 无节点级 SSH 或 `/ready` 失联。
- 错误响应被分类且可执行。

### L4：长上下文隔离

时长：直到计划 job 完成或 TTL 过期。

20B+ 长上下文 job 只能以 async 运行。同时保持低速前台检索流。

通过标准：

- 长上下文 job 永不进入普通同步 `/infer`。
- 检索流量仍排在长上下文工作前面。
- job TTL/cancel 路径可用。
- 每个长 job 都记录 `A100-SVC` slot hold time。

### L5：Soak

时长：至少 8 小时，优先 24 小时。

使用 L2 速率，并每 2 小时插入一次 L3 burst。如果 scheduler 支持维护窗口，还应覆盖：

- 低负载窗口允许后台摄入。
- preload/reindex/cleanup 不阻塞前台流量。
- soak 前、中、后都采样 TCM state。

通过标准：

- 无持续增长且无法恢复的 queue age。
- 无可预测 OOM 的内存增长趋势。
- A100 job 完成后无 stale TCM 积累。
- 无无法解释的模型退化或质量漂移。

### L6：恢复与故障注入

仅在正常路径通过后执行。

故障：

- 以 stale TCM blocks 开始，或在安全条件下模拟。
- kill 一个 A100 worker。
- kill 一个 X100 worker。
- 配置一个不兼容模型包。
- 填满 async backlog。

期望行为：

- 正常负载开始前检测到 stale TCM。
- worker crash 影响范围限制在对应模型/资源。
- 不兼容模型包只禁用该模型，除非有 A100 device poisoning 证据。
- A100 quarantine 只用于设备级失败，不用于普通模型不兼容。

## SLO 与判定

### Hard FAIL

任一项发生即为 hard FAIL：

- scheduler 或节点不可达。
- SSH/network 被模型负载饿死。
- A100/TCM stale 状态未被检测。
- 长 VLM/context job 在正常或 burst 阶段阻塞 realtime retrieval。
- 同步 `/infer` 在无显式 override 时接受已知长上下文 job。
- 出现 worker restart storm。
- 发生 kernel OOM 或需要设备重启。

### 正常混合目标

| 指标 | 目标 |
|---|---|
| `/ready` 和 `/capacity` p99 | < 500 ms |
| 前台检索 queue wait p95 | L2 < 2 s，L3 < 10 s |
| Compact VLM sync p95 | 在配置 deadline 内，初始可设 30 s |
| Heavy VLM async accept p95 | < 1 s |
| Async job poll 正确性 | 100% 最终状态可见 |
| 分类拒绝率 | burst 下允许，但必须有可执行原因 |
| 未分类 5xx | L2/L3 为 0 |
| 运行后 A100 stale TCM | 0 block，或显式 degraded verdict |

### 质量护栏

只通过性能不够。

- Compact VLM 文档抽取的 case pass 与 field accuracy 必须保持在上轮校准范围内。
- 快但弱的 VLM 必须保持 runtime-only，除非质量通过。
- LLM answer generation 必须使用必要的 no-think/chat-template 控制；只有
  reasoning 输出而 `content` 为空，对生产使用是 FAIL。

## 必采遥测

每次运行必须采集：

- request trace：`request_id`、flow、service class、model、sync/async、
  arrival time、start time、finish time、status、error reason。
- 按 service class 和 model 统计 queue wait。
- A100 service holder 与 hold time。
- X100 core usage / load average。
- DRAM used/reserved 和关键 worker RSS。
- 每个 phase 前后的 TCM snapshot。
- `/capacity`、`/metrics`，可选 `/events` stream。
- async work 的 job lifecycle events。
- 文档抽取和 LLM 回答的质量样本输出。

raw artifact 保留在 ignored `output/` 或 `reports/runs/` 下。整理后的结论进入固定名报告，
例如 `reports/k3-riscv-32g.en.md`。

## Harness 要求

下一步实现应新增 product-like scenario runner，不要把它硬塞进单模型 benchmark dimension。

推荐未来命令形态：

```bash
REALISTIC_BASE_URL=http://${K3_HOST}:8090 \
TARGET=k3-riscv-32g \
PROFILE=normal,busy,long_context,soak \
python3 scripts/run_k3_32g_realistic_stress.py
```

runner 应支持：

- trace-driven arrivals；
- service-class-aware assertions；
- sync/async workflow steps；
- 带 idle-window gating 的后台摄入；
- TCM preflight/postflight hooks；
- 可 resume 的输出目录；
- summary JSON 和 Markdown 输出。

在 runner 实现前，本文就是测试合同。任何临时脚本都必须保留上述 phase、pass/fail gate、
telemetry 和 artifact placement 规则。

## 与现有测试的关系

| 现有测试 | 继续用于 | 不足以覆盖 |
|---|---|---|
| ModelZoo full matrix | 模型可用性和基线速度 | 混合业务流量 |
| VLM full document suite | 单模型文档质量和延迟 | 队列 / 背压 |
| long-context suite | 模型能力上限 | 产品准入行为 |
| single-model concurrency | 端点吞吐 | 多服务资源公平性 |
| k3-scheduler pair/stress tests | scheduler 硬资源门证明 | 产品工作流真实性 |

真实业务压力测试应把这些已有测试校准出的模型知识，与产品形态的 arrival 和
scheduler-facing 行为结合起来。
