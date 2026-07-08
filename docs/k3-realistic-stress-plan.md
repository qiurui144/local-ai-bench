# K3 Realistic Mixed-Traffic Stress Test Plan

This document defines the next K3 stress round for `vlm-llm-benchmark`. It is
not another ModelZoo full-matrix run. The goal is to simulate how a production
application is likely to use a 32 GB SpacemiT K3 node through a scheduler or
inference gateway:
short foreground retrieval, controlled answer generation, document/VLM
extraction, async background ingestion, and recovery/health operations sharing
one constrained A100 service.

Chinese version: [k3-realistic-stress-plan.zh.md](k3-realistic-stress-plan.zh.md)

## Scope

### System Under Test

Primary SUT:

```text
product-like client -> k3-scheduler / inference gateway -> model workers
```

The preferred target endpoint is the scheduler-facing API, not raw model
servers:

| Capability | Preferred route |
|---|---|
| Sync model request | `POST /infer/{model}` |
| Async model request | `POST /infer/{model}:async` |
| Async poll | `GET /jobs/{id}` |
| Health | `GET /ready` |
| Capacity | `GET /capacity` |
| Events | `GET /events` |
| Metrics | `GET /metrics` |

Raw OpenAI-compatible model servers may still be used as controls, but they are
not sufficient for this stress round because they do not exercise scheduler
resource admission, queueing, async jobs, or A100/TCM recovery behavior.

### Non-Goals

- Do not run every cached ModelZoo model concurrently.
- Do not treat long-context 20B+ requests as normal sync traffic.
- Do not measure only single-model TPS; this round is about mixed-service
  stability and user-visible behavior.
- Do not start from a dirty A100/TCM state without recording it. A stale TCM
  state is a test finding, not an acceptable baseline.

## Product-Like Traffic Model

The stress workload is a trace mix, not a flat RPS benchmark. Each virtual user
executes one of the following product flows.

| Flow | Service class | Typical path | Expected behavior |
|---|---|---|---|
| Foreground query | `realtime_retrieval` -> `realtime_answer` | embed query -> vector-search stub -> rerank -> short LLM answer | Retrieval stays high priority; answer is bounded by deadline/token limit |
| Document upload | `realtime_vlm_compact` -> `background_index` | compact VLM extraction -> chunk -> embed | Small document may sync; larger work becomes async |
| Heavy document extraction | `user_async_vlm` | balanced/high-spec VLM document extraction | Accept quickly as job; never block short retrieval |
| Long-context analysis | `long_context` | 20B+ long-context answer or audit | Async only, TTL/cancel required |
| Background ingestion | `background_index` | batch embedding/rerank/summary | Runs only when short foreground queues are healthy |
| ASR note / meeting clip | `realtime_x100` or `user_async` | ASR -> optional summary | X100 budget must not starve scheduler/SSH/system |
| Control / recovery | `control` | ready/capacity/events/TCM health/unquarantine | Never blocked by model queues |

The vector database can be a stub. The important pressure is the model service
sequence and resource sharing, not database ranking quality.

## Model Roles For The First Round

Use the 2026-07-06 K3 32 GB calibration as the starting point.

| Role | Candidate | Scheduling rule |
|---|---|---|
| High-spec LLM primary | `Qwen3-30B-A3B-Q4_0` | Sync only for short bounded answers; otherwise async |
| High-throughput LLM control | `LFM2-24B-A2B-Q4_0` | Perf control; not default answer model until quality is qualified |
| Compact VLM | `Qwen3.5-2B.tar.gz` | Sync candidate if deadline allows; otherwise async |
| Balanced VLM | `Qwen3.5-4B.tar.gz` | Async by default |
| High-spec VLM | `qwen30ba3b-mm-q4_1.tar.gz` | Async only |
| VLM fallback/control | `Qwen3VL-4B + mmproj` | Async fallback/control |
| Embedding | BGE-Zh / Jina / Nomic / Qwen3-Embedding | `realtime_retrieval` |
| Reranker | `Bge-Reranker-V2-M3-Q4_0` | `realtime_retrieval` |
| ASR | `qwen3-asr-0.6B.tar.gz` | Sync or async depending clip length |

Do not promote fast runtime-only VLM models to production document extraction unless
their document-case quality passes. For example, FastVLM was fast in the last
round but did not pass document extraction quality.

## Load Levels

The test should run all levels in order. Later levels are invalid if earlier
levels fail health, correctness, or cleanup.

### L0: Preflight And Clean Baseline

Duration: 5-10 minutes.

Required checks:

- `GET /ready` succeeds.
- `GET /capacity` succeeds and reports no unexpected busy holder.
- `spacemit-tcm-smi` snapshot is recorded before any A100 work.
- If stale TCM blocks are present, record them and stop unless the run is
  explicitly a recovery test.
- One request per selected model role succeeds or produces an expected
  unsupported/incompatible error.
- No model starts by accident outside the planned hot set.

Pass criteria:

- Scheduler/inference gateway remains reachable.
- A100 is either clean or explicitly marked degraded before load starts.
- No crash loops, repeated init storms, or unexplained worker restarts.

### L1: Single-User Product Flow

Duration: 15 minutes.

Run a realistic sequential workflow:

1. Upload one document image.
2. Run compact VLM extraction.
3. Chunk and embed extracted text.
4. Run one foreground query.
5. Rerank candidate snippets.
6. Generate a short bounded answer.
7. Submit one heavier VLM extraction as async.
8. Poll the async job to completion.

Pass criteria:

- User-visible sync operations complete within their configured deadlines.
- Async acceptance returns quickly with a job id.
- The system is still healthy after job completion.
- Document extraction quality remains within the calibrated expectation for the
  selected compact VLM.

### L2: Normal Business Mix

Duration: 60 minutes after a 10-minute ramp.

Suggested arrival mix:

| Flow | Share | Initial rate target |
|---|---:|---|
| Foreground query | 60% | 2 requests/min total |
| Compact VLM document upload | 15% | 0.3 requests/min |
| Heavy VLM async | 5% | 0.05 requests/min |
| ASR note | 5% | 0.05 requests/min |
| Background ingestion | 15% | fills idle windows only |

The exact rate may be adjusted after L1, but the mix should remain realistic:
short retrieval should be frequent, heavy VLM should be rare, and background
work should be opportunistic.

Pass criteria:

- `control` endpoints p99 stay responsive.
- Foreground retrieval is not blocked behind long VLM/context work.
- Heavy VLM is accepted as async or rejected with a clear backpressure reason.
- Background ingestion makes progress only when foreground queues are healthy.
- No A100 quarantine, stale TCM leak, node hang, or SSH/network starvation.

### L3: Busy-Hour Burst

Duration: 30 minutes.

Pattern:

- 5-minute ramp to 3x L2 foreground query arrival.
- Inject a burst of document uploads: 3 compact VLM jobs within 2 minutes.
- Inject 1 high-spec VLM async job.
- Keep background ingestion enabled but low priority.

Expected behavior:

- The system may shed load or return `queue_full`, `async_required`,
  `busy_retry`, or `deadline_exceeded`.
- The system must not run long VLM ahead of short retrieval.
- The system must not convert backpressure into worker crashes.

Pass criteria:

- No process restart storm.
- No scheduler crash.
- No node-level loss of SSH or `/ready`.
- Error responses are classified and actionable.

### L4: Long-Context Isolation

Duration: until all planned jobs finish or TTL expires.

Run long-context 20B+ jobs only as async. Keep a low-rate foreground retrieval
stream running at the same time.

Pass criteria:

- Long-context jobs never enter normal sync `/infer`.
- Retrieval traffic remains admitted ahead of long-context work.
- Job TTL/cancel paths work.
- `A100-SVC` slot hold time is recorded for every long job.

### L5: Soak

Duration: 8 hours minimum, 24 hours preferred.

Use L2 rates with periodic L3 bursts every 2 hours. Include maintenance-window
behavior if the scheduler supports it:

- Background ingestion is allowed in low-load windows.
- Preload/reindex/cleanup never blocks foreground traffic.
- TCM state is sampled before, during, and after the soak.

Pass criteria:

- No increasing queue age without recovery.
- No memory growth trend that predicts OOM.
- No stale TCM accumulation after completed A100 jobs.
- No unexplained model degradation or quality drift.

### L6: Recovery And Fault Injection

Run only after the normal path passes.

Faults:

- Start with stale TCM blocks, or simulate them if safe.
- Kill one A100 worker.
- Kill one X100 worker.
- Configure one incompatible model package.
- Fill the async backlog.

Expected behavior:

- Stale TCM is detected before normal load starts.
- A worker crash stays scoped to the model/resource.
- Incompatible model package disables that model only unless there is evidence
  of A100 device poisoning.
- A100 quarantine is used for device-level failure, not for ordinary model
  incompatibility.

## SLO And Verdicts

### Hard Fail

Any item below is a hard FAIL:

- Scheduler or node becomes unreachable.
- SSH/network is starved by model load.
- A100/TCM state becomes stale without detection.
- A long VLM/context job blocks realtime retrieval in normal or burst phases.
- Sync `/infer` accepts a known long-context job without explicit override.
- Worker restart storm occurs.
- Kernel OOM or device reboot is required.

### Normal Mix Targets

| Metric | Target |
|---|---|
| `/ready` and `/capacity` p99 | < 500 ms |
| Foreground retrieval queue wait p95 | < 2 s in L2, < 10 s in L3 |
| Compact VLM sync p95 | within configured deadline, initially 30 s |
| Heavy VLM async accept p95 | < 1 s |
| Async job poll correctness | 100% terminal state eventually visible |
| Classified rejection rate | allowed under burst, must have actionable reason |
| Unclassified 5xx | 0 in L2/L3 |
| A100 stale TCM after run | 0 blocks, or explicit degraded verdict |

### Quality Guardrails

Performance-only success is not enough.

- Compact VLM document extraction must keep document-case pass and field
  accuracy within the previous calibrated band.
- Fast-but-weak VLM models must remain marked runtime-only unless quality passes.
- LLM answer generation must use no-think/chat-template controls where required;
  empty `content` with only reasoning output is a FAIL for production usage.

## Required Telemetry

Every run must collect:

- Request trace with `request_id`, flow, service class, model, sync/async,
  arrival time, start time, finish time, status, error reason.
- Queue wait by service class and model.
- A100 service holder and hold time.
- X100 core usage / load average.
- DRAM used/reserved and process RSS for key workers.
- TCM snapshot before/after each phase.
- `/capacity`, `/metrics`, and optional `/events` stream.
- Job lifecycle events for async work.
- Quality sample outputs for document extraction and LLM answers.

Raw artifacts stay under ignored `output/` or `reports/runs/`. Curated
conclusions belong in fixed-name reports such as `reports/k3-riscv-32g.en.md`.

## Harness Requirements

The next implementation should add a product-like scenario runner rather than
overloading single-model benchmark dimensions.

Recommended future command shape:

```bash
ATTUNE_BASE_URL=http://${K3_HOST}:8090 \
TARGET=k3-riscv-32g \
PROFILE=normal,busy,long_context,soak \
python3 scripts/run_k3_32g_realistic_stress.py
```

The runner should support:

- trace-driven arrivals;
- service-class-aware assertions;
- sync/async workflow steps;
- background ingestion with idle-window gating;
- TCM preflight/postflight hooks;
- resume-safe output directories;
- summary JSON and Markdown output.

Until that runner exists, this document is the test contract. Any ad hoc script
used for the next run must preserve the phases, pass/fail gates, telemetry, and
artifact placement rules above.

## Relationship To Existing Tests

| Existing run | Keep using it for | Not enough for |
|---|---|---|
| ModelZoo full matrix | model availability and baseline speed | mixed product traffic |
| VLM full document suite | per-model document quality and latency | queueing/backpressure |
| long-context suite | model capability limits | product admission behavior |
| single-model concurrency | endpoint throughput | multi-service resource fairness |
| k3-scheduler pair/stress tests | scheduler hard gate proof | product workflow realism |

The realistic stress round should combine the calibrated model knowledge
from those tests with product-shaped arrivals and scheduler-facing behavior.
