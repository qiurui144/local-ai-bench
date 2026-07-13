# K3 参数矩阵补测计划

日期：2026-07-08

## 1. 目的

本计划用于补齐 K3 32GB 上 LLM、RAG、VLM、OCR、ASR 的参数曲线。目标不是证明
runtime 能否跑通，而是判断某个参数组合是否适合产品同步路径、异步队列、离线任务或
不推荐部署。

典型问题：

```text
模型支持 1K-4K context，但 3K/4K 回答可能需要数分钟。
```

这种情况不能只标为 PASS。报告必须同时给出 latency、memory、quality 和产品调度
verdict。

### 1.1 硬规则

- `PASS` 只能表示单项运行检查通过，不能作为产品化结论。
- 禁止再只标 `1K-4K context PASS`。每个 context/output/RAG/VLM 参数组都必须单独
  输出 latency、memory、quality、startup、queue、resource hold 和产品化 verdict。
- verdict 只能取 `sync_default`、`sync_bounded`、`async_default`、`async_only`、
  `offline_only`、`not_recommended`、`blocked`。
- 只有参数矩阵覆盖完整，且 p95/p99、峰值内存、质量、启动状态、队列等待和资源持有
  都有记录时，才能把某个区间写成产品默认或同步边界。
- `k3-scheduler` 的详细默认值必须等待参数矩阵结果后再定，包括
  `max_context_tokens_sync`、`max_output_tokens_sync`、ETA 权重、hot-set、
  RAG chunk/top-k/evidence 默认值、VLM sync/async 边界和 UI timeout。
- 测试项目、参数轴和后续输出物以 `docs/evaluation-contract.json` 为准；
  后续 JSON 输出物必须分别符合 `docs/parameter-matrix.schema.json`、
  `docs/run-summary.schema.json`、`docs/model-profile.schema.json` 和
  `docs/scheduler-contract.schema.json`。

## 2. 必测参数

### 通用 admission 维度

所有 LLM/RAG/VLM/OCR/ASR 样本都必须带上以下调度维度。它们不是模型输入本身，
但会直接进入 `k3-scheduler` 的 admission control、ETA 和 UI timeout。

| 参数 | 取值 / 说明 |
|:---|:---|
| `task_class` | `llm_chat`、`llm_summary`、`embedding`、`reranker`、`rag_answer`、`rag_search_only`、`vlm_qa`、`vlm_doc_extract`、`ocr`、`asr`、`mixed_nas_load`、`soak` |
| `priority_class` | `foreground`、`interactive`、`background`、`maintenance` |
| `deadline_ms` | UI/调用方给出的硬截止时间；无显式值时使用 §7 默认 |
| `sync_requested` | 用户是否请求同步返回；scheduler 可降级为 async |
| `startup_state` | `hot_resident`、`warm_process`、`cold_start` |
| `hot_set_id` | 命中的 hot-set 名称；未命中填 `none` |
| `resource_class` | `x100_cpu`、`a100_ime2`、`gpu_vpu`、`mixed` |
| `memory_tier` | `k3_16g`、`k3_32g` |
| `storage_state` | `page_cache_hot`、`page_cache_cold`、`direct_io`、`unknown` |
| `nas_pressure` | `idle`、`normal`、`scrub_or_fio`、`low_mem` |
| `fallback_allowed` | 是否允许 cloud / 小模型 / search-only fallback |

### LLM / RAG answer

| 参数 | 取值 |
|:---|:---|
| context tokens | 512、1024、2048、3072、4096 |
| max output tokens | 64、128、256、512 |
| startup state | hot resident、cold start |
| RAG evidence chunks | 3、5、8、12 |
| rerank top-n | 10、20、50 |
| prompt cache | off、small、default、large |
| answer mode | direct answer、RAG grounded answer、summary、structured JSON |

同步路径必须额外覆盖以下边界，生成 scheduler 字段：

| 字段 | 评测方式 |
|:---|:---|
| `max_context_tokens_sync` | 在 p95 latency、质量和内存均达标的最大 context |
| `max_output_tokens_sync` | 在 p95 latency、质量和内存均达标的最大 output |
| `sync_deadline_ms` | 同步路径建议 deadline，必须小于 UI hard timeout |
| `sync_queue_wait_budget_ms` | 同步请求允许排队的最大时间 |
| `sync_resource_hold_budget_ms` | 同步请求允许持有 A100/IME2/X100 的最大时间 |
| `async_cutover_reason` | 超过边界时转 async 的原因：`latency`、`memory`、`quality`、`queue` |

### Embedding / reranker

| 参数 | 取值 |
|:---|:---|
| embedding batch size | 1、4、8、16、32 |
| query length | 16、64、128、256 tokens |
| rerank candidate count | 10、20、50、100 |
| document chunk length | 128、256、512、1024 tokens |

### VLM / OCR

| 参数 | 取值 |
|:---|:---|
| image count | 1、2、4 |
| image resize | default、1024px、1536px、2048px |
| document pages | 1、2、5、10 |
| output schema | short JSON、full JSON |
| vision token budget | auto、512、1024、1536、2048 |
| VLM mode | image QA、document extraction、OCR+LLM postprocess |

不要把固定 `--image-min-tokens 1024 --image-max-tokens 1024` 作为默认路径；该参数组合已有异常慢和 warning 风险，只能作为受控负向实验。

VLM 必须输出同步/异步边界：

| 字段 | 含义 |
|:---|:---|
| `vlm_sync_max_images` | 同步路径允许的最大图片数量 |
| `vlm_sync_max_pages` | 同步路径允许的最大文档页数 |
| `vlm_sync_max_resize_px` | 同步路径允许的最大长边 resize |
| `vlm_sync_output_schema` | 同步路径允许 `short_json` 还是 `full_json` |
| `vlm_async_boundary` | 超过 images/pages/resize/schema/output/deadline 任一边界即进入 async |

### ASR

| 参数 | 取值 |
|:---|:---|
| audio duration | 10s、30s、60s、300s |
| concurrency | 1、2、4 |
| language | zh、en、mixed |

## 3. 必采指标

每条样本必须记录：

| 指标 | 说明 |
|:---|:---|
| `ttft_ms` | 首 token 或首响应时间 |
| `prefill_tps` | prompt processing 吞吐 |
| `decode_tps` / `tpot_ms` | 输出阶段速度 |
| `e2e_latency_ms` | 用户可感知总耗时 |
| `queue_wait_ms` | scheduler 队列等待 |
| `startup_wait_ms` | 冷启动或 warming 等待 |
| `startup_state` | hot、warming、cold |
| `prompt_tokens` | 实际输入 tokens |
| `completion_tokens` | 实际输出 tokens |
| `rss_peak_mb` | worker 峰值 RSS |
| `mem_available_min_mb` | 系统最低可用内存 |
| `cma_free_min_kb` | 若平台可采集，记录连续内存低水位 |
| `tcm_state_before/after` | K3 SpacemiT 路径必须记录 |
| `resource_hold_ms` | A100/IME2/SMT session 持有时间 |
| `quality_score` | 任务质量指标 |
| `error_class` | timeout、OOM、context_overflow、queue_full 等 |
| `runtime_version` | llama.cpp / ORT / SMT 版本 |
| `model_artifact_id` | 模型包版本、hash、量化 |

统计必须输出 p50、p90、p95、p99、max。平均值只能作为辅助。

### 3.1 参数组汇总输出

每组参数必须输出以下 profile。原始指标可以更细，但汇总层必须具备这些字段，供
`k3-scheduler`、UI 和报告消费。测试项目与参数轴以
`docs/evaluation-contract.json` 为准，完整 JSON 输出必须符合
对应 schema。

| profile | 必填内容 |
|:---|:---|
| `latency_profile` | `ttft_ms`、`prefill_tps`、`decode_tps` / `tpot_ms`、`e2e_latency_ms`，并给出 p50/p90/p95/p99/max |
| `memory_profile` | `rss_peak_mb`、`mem_available_min_mb`、`cma_free_min_kb`、OOM/kill/restart 记录、NAS 保护阈值命中情况 |
| `quality_profile` | 任务质量分、判定阈值、失败原因；RAG/VLM 必须记录 evidence 或结构化字段是否可用 |
| `startup_profile` | `startup_state`、`startup_wait_ms`、`model_io_ms`、page-cache/direct I/O 状态、warm 命中情况 |
| `queue_profile` | `queue_wait_ms`、队列深度、`priority_class`、deadline 命中情况、是否触发 async cutover |
| `resource_profile` | `resource_hold_ms`、`resource_class`、lease 互斥组、是否被抢占或降级 |
| `error_profile` | `error_class`、`retryable`、`blocked_reason` |
| `product_verdict` | 固定 verdict 之一，不允许写普通 `PASS` |

建议 `parameter-matrix.json` 的单行结构：

```json
{
  "target": "k3-riscv-32g",
  "test_item_id": "rag_answer_defaults",
  "task_class": "rag_answer",
  "priority_class": "interactive",
  "deadline_ms": 45000,
  "sync_requested": true,
  "hot_set_id": "k3_32g_qwen30b_chat",
  "memory_tier": "k3_32g",
  "storage_state": "direct_io",
  "nas_pressure": "normal",
  "fallback_allowed": true,
  "model_artifact_id": "Qwen3-30B-A3B-Q4_0.gguf@sha256:...",
  "model_profile": {
    "name": "Qwen3-30B-A3B",
    "format": "gguf",
    "quantization": "Q4_0",
    "artifact_size_bytes": 0,
    "artifact_hash": "sha256:...",
    "load_mode": "direct_io"
  },
  "runtime": {
    "name": "llama.cpp-tools-spacemit",
    "version": "unknown",
    "resource_class": "a100_ime2"
  },
  "dataset_profile": {
    "dataset_id": "rag.zh.default",
    "sample_count": 0,
    "seed": 0
  },
  "params": {
    "context_tokens": 2048,
    "max_output_tokens": 128,
    "chunk_tokens": 512,
    "retrieve_top_k": 50,
    "rerank_top_k": 20,
    "evidence_chunks": 5,
    "startup_state": "hot_resident"
  },
  "latency_profile": {
    "e2e_latency_ms": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0},
    "ttft_ms": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0},
    "prefill_tps": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0},
    "decode_tps": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0},
    "tpot_ms": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
  },
  "memory_profile": {
    "rss_peak_mb": 0,
    "mem_available_min_mb": 0,
    "cma_free_min_kb": 0,
    "tcm_state_before": null,
    "tcm_state_after": null,
    "oom": false,
    "worker_restarted": false
  },
  "quality_profile": {
    "score": 0.0,
    "threshold": 0.0,
    "passed": false,
    "reason": "not_measured",
    "evidence_status": null
  },
  "startup_profile": {
    "startup_state": "hot_resident",
    "startup_wait_ms": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0},
    "model_io_ms": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0},
    "storage_state": "direct_io"
  },
  "queue_profile": {
    "queue_wait_ms": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0},
    "queue_depth_p95": 0,
    "deadline_hit_rate": 0.0,
    "async_cutover_reason": "blocked"
  },
  "resource_profile": {
    "resource_hold_ms": {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0},
    "resource_class": "a100_ime2",
    "lease_conflict_rate": 0.0,
    "preempted": false
  },
  "error_profile": {
    "error_class": "not_measured",
    "retryable": null,
    "blocked_reason": "not_measured"
  },
  "product_verdict": "blocked",
  "product_verdict_reason": "not_measured",
  "confidence": "low"
}
```

## 4. 产品化判定

每个参数组合必须输出一个 verdict：

| verdict | 含义 |
|:---|:---|
| `sync_default` | 可作为默认同步路径 |
| `sync_bounded` | 仅在指定 context/output/deadline 内同步 |
| `async_default` | 质量可用但不适合同步 |
| `async_only` | 不允许同步 |
| `offline_only` | 仅适合离线/维护窗口 |
| `not_recommended` | 质量或稳定性不达标 |
| `blocked` | runtime、shape 或 provider 不支持 |

示例：

```text
context=4096, output=512 若质量 PASS 但 p95 > 60s，应判为 async_default 或 async_only，
而不是普通 PASS。
```

### 4.1 同步 admission 规则

`sync_default` 和 `sync_bounded` 必须同时满足：

- `p95_e2e_latency_ms <= sync_deadline_ms`。
- `p95_queue_wait_ms <= sync_queue_wait_budget_ms`。
- `p95_resource_hold_ms <= sync_resource_hold_budget_ms`。
- `rss_peak_mb` 和 `mem_available_min_mb` 不触发 NAS 保护阈值。
- `quality_score` 达到该任务最低线。
- 没有 `OOM`、`context_overflow`、`queue_full`、`worker_restart`。

若质量可用但任一同步条件不满足，判定必须降为 `async_default`、`async_only`
或 `offline_only`。不要用 `PASS` 替代产品 verdict。

### 4.2 初始同步边界候选

以下值只是 scheduler 开发期 bootstrap，不是发布默认。B1-B4 数据出来后必须回填覆盖。

| task/profile | `max_context_tokens_sync` | `max_output_tokens_sync` | `sync_deadline_ms` | 初始 verdict |
|:---|---:|---:|---:|:---|
| `llm_chat` 16G 3B/7B | 2048 | 128 | 30000 | `sync_bounded` |
| `llm_summary` 16G 3B/7B | 2048 | 256 | 45000 | `sync_bounded` |
| `llm_chat` 32G Qwen3-30B | 1024 | 64 | 45000 | `sync_bounded` 待证据确认；不达标即 `async_default` |
| `rag_answer` 16G 3B/7B | 2048 | 128 | 45000 | `sync_bounded` |
| `rag_answer` 32G Qwen3-30B | 1024 | 128 | 60000 | `sync_bounded` 待证据确认；长上下文默认 async |
| `vlm_qa` compact VLM | 1024 | 64 | 60000 | 仅 1 图同步 |
| `vlm_doc_extract` | 1024 | 64 | 90000 | 仅 1 页 short JSON 同步，其余 async |
| `asr` short clip | n/a | n/a | 60000 | 10-30s 音频可尝试同步 |

## 5. 第一轮 K3 补测批次

### B0：框架补齐

- 增加参数矩阵输入文件。
- 测试项目与参数轴固定在 `docs/evaluation-contract.json`。
- 支持 scheduler `/benchmark/contract` 发现模型限制。
- 支持 hot/cold 分桶。
- 支持 RSS、MemAvailable、CMA/TCM、resource hold 采集。
- 支持 `hot_set_id`、`deadline_ms`、`priority_class`、`nas_pressure`、`storage_state`。
- 输出 `parameter-matrix.json`、`run-summary.json`、`verdict-table.tsv`、`model-profile.json`、`scheduler-contract.json`。
- 后续输出物必须声明 `contract_id=vlm-llm-nas-evaluation-contract` 和
  `contract_version`。
- `parameter-matrix.json`、`run-summary.json`、`model-profile.json`、
  `scheduler-contract.json` 必须符合各自 schema。

### B1：单模型扫描

- `llm-summary`：context 512-4096 × output 64-512。
- `llm-chat` / 30B：context 512-4096 × output 64-512。
- embedding：batch size × query length。
- reranker：candidate count × chunk length。
- compact VLM：image size/page count/schema。
- ASR：duration × concurrency。

### B2：RAG 参数扫描

```text
chunk_size: 256, 512, 768, 1024
retrieve_top_n: 20, 50, 100
rerank_top_n: 10, 20, 50
evidence_chunks: 3, 5, 8, 12
answer_output_tokens: 128, 256
```

输出推荐默认 RAG 参数。

默认候选必须按任务拆分输出，不能只有一个全局 top-k：

| 场景 | `chunk_tokens` | `chunk_overlap_tokens` | `retrieve_top_k` | `rerank_top_k` | `evidence_chunks` | `answer_context_budget_tokens` | `answer_output_tokens` |
|:---|---:|---:|---:|---:|---:|---:|---:|
| quick answer sync | 512 | 64 | 20 | 10 | 3 | 1024 | 128 |
| grounded answer sync | 512 | 64 | 50 | 20 | 5 | 1536 | 128 |
| long answer async | 768 | 96 | 50 | 20 | 8 | 3072 | 256 |
| research/offline | 1024 | 128 | 100 | 50 | 12 | 4096 | 512 |

若 reranker 延迟 p95 超过同步 deadline 预算，sync 路径必须降级为：

```text
retrieve_top_k=20
rerank_top_k=10
evidence_chunks=3
answer_output_tokens=128
```

### B3：混合负载扫描

- daily LLM + RAG。
- 文档上传 OCR/VLM + daily LLM。
- 会议 ASR + daily LLM。
- project VLM + daily LLM。
- long-context async + foreground retrieval。

### B4：soak

对 B1-B3 推荐参数跑 8-24h，记录内存趋势、worker restart、quarantine、质量漂移和
trim 后恢复情况。

## 6. scheduler 输入

本补测完成后，需要把以下字段回填给 scheduler 配置或 profile importer：

| 字段 | 用途 |
|:---|:---|
| `max_context_tokens_sync` | sync admission |
| `max_output_tokens_sync` | sync admission |
| `estimated_runtime_ms` by bucket | ETA |
| `cold_start_ms` | cold-start avoidance |
| `resident_gb` / `rss_peak_mb` | memory governor |
| `resource_hold_ms` | ResourceLease ETA |
| `quality_profile` | model selection |
| `failure_profile` | quarantine/fallback |
| `recommended_task_defaults` | `/kb/tasks` defaults |
| `eta_weights` | ETA 公式权重 |
| `hot_set_profile` | 常驻/预热模型集合 |
| `rag_defaults` | RAG chunk/top-k/evidence 默认值 |
| `vlm_sync_boundary` | VLM sync/async admission |
| `ui_timeout_profile` | 前端、API、proxy timeout 建议 |

在这些结果出来前，scheduler 详细开发拆解不应固化默认 context、RAG top-k、batch size、
VLM sync 边界和 UI timeout。

### 6.0 scheduler 取值冻结规则

下表字段必须由 B1-B4 参数矩阵推导，不能在数据不足时写死为发布默认。若样本数不足或
质量/稳定性未闭环，只能输出 bootstrap 候选并标记 `confidence=low`。

| scheduler 字段 | 数据来源 | 可固化条件 |
|:---|:---|:---|
| `max_context_tokens_sync` | LLM/RAG context × output 矩阵 | 最大 context 的 p95 latency、p95 queue、p95 resource hold、内存和质量全部达标 |
| `max_output_tokens_sync` | output token 扫描 | 最大 output 的 p95 latency、质量和资源持有不超过同步预算 |
| ETA 权重 | 每个 bucket 的 queue/startup/prefill/decode/resource/model I/O 误差 | p95 ETA 误差进入可接受范围，且 cold/hot 分桶均有样本 |
| hot-set | 常驻内存、warm TTL、命中率、驱逐次数、NAS 压力 | 不触发低内存保护，不影响存储任务，互斥组明确 |
| RAG chunk/top-k/evidence 默认值 | B2 RAG 参数扫描 | 质量、延迟和 evidence 可解释性共同最优；同步和异步默认分开 |
| VLM sync/async 边界 | VLM image/page/resize/schema/output 扫描 | p95 不超过 UI hard timeout，warning 和异常慢组合已排除 |
| UI timeout | 实测 ETA、async cutover、用户可见状态 | UI soft/hard、API、proxy timeout 分层验证后再固化 |

因此 scheduler 详细拆解的顺序是：

1. 先产出参数矩阵和 `product_verdict`。
2. 再生成 `model-profile.json` 与 `scheduler-contract.json`。
3. 最后由 scheduler 消费 profile，决定同步默认、异步默认、hot-set 和 UI timeout。

### 6.1 ETA 权重范本

ETA 必须按 bucket 估算，不允许只用全局平均值。建议公式：

```text
eta_ms =
  queue_weight       * p95_queue_wait_ms +
  startup_weight     * p95_startup_wait_ms +
  prefill_weight     * ceil(prompt_tokens / p50_prefill_tps * 1000) +
  decode_weight      * ceil(max_output_tokens / p50_decode_tps * 1000) +
  resource_weight    * p95_resource_hold_ms +
  io_weight          * p95_model_io_ms +
  safety_margin_ms
```

初始权重：

| 权重 | 默认 | 说明 |
|:---|---:|:---|
| `queue_weight` | 1.00 | admission 已知排队时间 |
| `startup_weight` | 1.15 | cold/warming 波动大，略放大 |
| `prefill_weight` | 1.15 | 长 context 更容易抖动 |
| `decode_weight` | 1.25 | 输出阶段直接影响 UI 等待 |
| `resource_weight` | 1.00 | 用于 ResourceLease ETA |
| `io_weight` | 1.10 | cold page-cache/direct I/O 对冷启动敏感 |
| `memory_pressure_multiplier` | 1.20-1.50 | `nas_pressure=low_mem` 或 PSI 异常时启用 |
| `safety_margin_ms` | 1500 | 小任务固定保护边界；长任务可用 5%-10% |

输出 profile 时必须同时给 `p50_eta_ms`、`p95_eta_ms`、`p99_eta_ms` 和
`eta_confidence`。若样本数不足，`eta_confidence=low`，scheduler 只能用于 async
admission，不可作为 sync 默认。

### 6.2 hot-set 范本

hot-set 是 scheduler 可主动保温或允许常驻的模型集合。每个 hot-set 必须有内存、资源和互斥关系：

| 字段 | 说明 |
|:---|:---|
| `hot_set_id` | 例如 `k3_16g_foundation`、`k3_32g_qwen30b_chat` |
| `models` | 模型 artifact 列表 |
| `resident_policy` | `always`、`opportunistic`、`ttl`、`manual` |
| `warm_ttl_s` | 空闲后保持 warm 的秒数 |
| `resident_gb_budget` | 常驻内存预算 |
| `prompt_cache_mb` | prompt cache 上限 |
| `mutex_groups` | 互斥集合，如 `qwen30b_full` 与 `vlm_doc_extract` |
| `preemptible` | 是否可被 NAS 保护或高优先任务驱逐 |

初始 hot-set 候选：

| hot-set | 目标 | 常驻策略 | 备注 |
|:---|:---|:---|:---|
| `k3_16g_foundation` | embedding/rerank/OCR/ASR 小模型 | `opportunistic` | NAS 优先，按需拉起 |
| `k3_16g_chat_7b` | 3B/7B chat | `ttl` | 不与重 OCR/VLM 并发 |
| `k3_32g_qwen30b_chat` | Qwen3-30B chat/RAG | `manual` 或 `ttl` | 与 full-GGUF CMA、VLM 文档抽取互斥 |
| `k3_32g_vlm_compact` | 单图/单页 VLM | `opportunistic` | 只允许小 VLM 走同步 |

### 6.3 scheduler profile 输出范本

`model-profile.json` 至少包含：

```json
{
  "schema_version": 1,
  "target": "k3-riscv-32g",
  "generated_at": "2026-07-08T00:00:00+08:00",
  "model_artifact_id": "Qwen3-30B-A3B-Q4_0.gguf@sha256:...",
  "runtime": {
    "name": "llama.cpp-tools-spacemit",
    "version": "unknown",
    "resource_class": "a100_ime2"
  },
  "task_profiles": {
    "rag_answer": {
      "verdict": "sync_bounded",
      "max_context_tokens_sync": 1024,
      "max_output_tokens_sync": 128,
      "sync_deadline_ms": 60000,
      "sync_queue_wait_budget_ms": 3000,
      "estimated_runtime_ms": {
        "p50": 0,
        "p95": 0,
        "p99": 0,
        "confidence": "low"
      },
      "eta_weights": {
        "queue_weight": 1.0,
        "startup_weight": 1.15,
        "prefill_weight": 1.15,
        "decode_weight": 1.25,
        "resource_weight": 1.0,
        "io_weight": 1.1,
        "safety_margin_ms": 1500
      },
      "rag_defaults": {
        "chunk_tokens": 512,
        "chunk_overlap_tokens": 64,
        "retrieve_top_k": 50,
        "rerank_top_k": 20,
        "evidence_chunks": 5,
        "answer_context_budget_tokens": 1536,
        "answer_output_tokens": 128
      },
      "fallback": {
        "on_timeout": "async_job",
        "on_queue_full": "search_only",
        "on_low_memory": "smaller_model"
      }
    }
  },
  "hot_sets": [
    {
      "hot_set_id": "k3_32g_qwen30b_chat",
      "resident_policy": "ttl",
      "warm_ttl_s": 900,
      "resident_gb_budget": 20,
      "prompt_cache_mb": 2048,
      "mutex_groups": ["vlm_doc_extract", "full_gguf_cma"],
      "preemptible": true
    }
  ]
}
```

`scheduler-contract.json` 至少包含 endpoint 级别边界：

| endpoint | 必填字段 |
|:---|:---|
| `/capacity` | hot-set、队列深度、resource lease、memory tier、degraded reason |
| `/benchmark/contract` | 模型 artifact、task profile、sync limits、ETA bucket、verdict |
| `/v1/chat/completions` | sync admission、async cutover、timeout reason、fallback reason |
| `/kb/tasks` | RAG defaults、job timeout、poll interval、cancel policy |

## 7. UI timeout 指南

UI 不应直接等模型跑完所有长任务。前端、attune API、scheduler 三层 timeout 必须分开：

| 场景 | UI soft timeout | UI hard timeout | scheduler action | UI 行为 |
|:---|---:|---:|:---|:---|
| quick chat sync | 8s | 30s | 超过 ETA 转 async | 8s 显示“处理中”，30s 给后台任务入口 |
| RAG answer sync | 12s | 45s | 超过 sync boundary 转 async | 展示检索到的 evidence，再等待回答 |
| Qwen3-30B short chat | 15s | 60s | 只允许 `sync_bounded` | 超时自动转后台，不重试同一 worker |
| VLM one image QA | 20s | 60s | 只允许 1 图 short output 同步 | 超时转后台 |
| VLM document extraction | 30s | 90s | 1 页 short JSON 可同步，其余 async | 默认后台任务 |
| ASR short clip | 15s | 60s | 10-30s 音频可同步 | 长音频后台 |
| offline / maintenance | n/a | n/a | async/offline only | 只显示任务进度和完成通知 |

Proxy/API 建议：

- UI HTTP request hard timeout 必须小于 nginx/proxy timeout。
- attune -> scheduler 的 sync request hard timeout 应比 UI hard timeout 少 `2-5s`，给 UI 留错误处理时间。
- async job poll 初始 `2s`，30s 后改为 `5s`，5 分钟后改为 `15s`。
- 所有 timeout 错误必须带 `job_id` 或 `retry_policy`，禁止让用户重复提交造成队列放大。

## 8. VLM sync/async 边界初稿

| VLM 任务 | 同步允许 | 异步边界 |
|:---|:---|:---|
| 单图问答 | `image_count=1`、`resize<=1024px`、`output<=64`、short answer | 多图、`resize>1024px`、`output>64` |
| 文档抽取 | `pages=1`、short JSON、字段数小于 20 | `pages>1`、full JSON、表格/多区域抽取 |
| OCR + LLM 后处理 | OCR 可同步，LLM 整理按 ETA 判定 | 多页、长表格、需结构化校验 |
| VLM + RAG 混合 | 不做默认同步 | 默认 async |

若 VLM runtime 报 warning、图像 token 固定 1024 导致异常慢、或 p95 超过 UI hard timeout，
该组合必须标记为 `async_only` 或 `not_recommended`。

## 9. 跨平台重测计划（暂不执行）

本阶段只补齐计划和评测范本，不启动 Intel、AMD 或 K3 测试。后续重测统一采用本文件的
参数矩阵与 verdict 合同，不能回到只输出 measured/PASS/blocked 的报告形式。

### 9.1 重测目标

| target | 系统 | 目标 |
|:---|:---|:---|
| `amd-win-x86` | Windows | 复测 AMD Windows LLM/VLM/embedding/reranker/OCR/ASR，补齐 profile 与 verdict |
| `amd-linux-x86` | Linux | 复测 AMD Linux 加速路径和必要 CPU baseline |
| `intel-win-x86` | Windows | 复测 Intel Windows OpenVINO/DirectML/CPU baseline |
| `intel-linux` | Linux | 复测 Intel Linux OpenVINO 加速路径 |
| `k3-riscv-16g` | Bianbu/K3 | 复测 3B/7B、embedding、reranker、OCR/ASR、compact VLM |
| `k3-riscv-32g` | Bianbu/K3 | 补测 Qwen3-30B GGUF、RAG、VLM 边界、混合负载和 CMA/direct I/O 形态 |

### 9.2 批次

| 批次 | 内容 | 输出 | 当前状态 |
|:---|:---|:---|:---|
| P0 | 补齐矩阵 schema、报告模板、profile importer、verdict 规则 | `parameter-matrix.json` schema、报告范本、导入规则 | 仅开发，不跑模型 |
| P1 | AMD/Intel Windows 重测 | Windows 每模型 profile、verdict-table、平台差异 | 待执行 |
| P2 | AMD/Intel Linux 重测 | Linux 每模型 profile、verdict-table、平台差异 | 待 Windows 完成或显式失效后执行 |
| P3 | K3 补测 | 16G/32G 参数矩阵、Qwen3-30B 边界、RAG/VLM/ASR/OCR profile | 待执行 |
| P4 | scheduler 汇总 | `model-profile.json`、`scheduler-contract.json`、hot-set 与 timeout 建议 | 依赖 P1-P3 结果 |

### 9.3 执行约束

- 每台机器一次只跑一个模型，避免模型加载、judge、service probe 互相污染。
- Windows 与 Linux 的同平台结果不可混用，CPU baseline 不进入加速路径推荐。
- K3 32GB 的 Qwen3-30B 必须把 GGUF 大小、page-cache/direct I/O、CMA/TCM 低水位和
  cold/hot startup 分桶记录完整。
- 若某参数组缺少 latency、memory、quality、startup、queue 或 resource hold 任一 profile，
  verdict 必须为 `blocked`，不能写 `sync_default` 或 `sync_bounded`。
- 重测结果先进入 `output/reports/` 和内部汇总，确认后再选择性同步到公开文档。
