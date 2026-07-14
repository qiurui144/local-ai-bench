# Intel Linux iGPU

**Last updated:** 2026-07-14
**Chinese version:** [igpu.zh.md](igpu.zh.md)
**Contract source runs:** `intel-linux-20260712-contract-full`, `intel-linux-20260713-q25-7b-quality`, `intel-linux-20260713-q25-7b-quality-hfdatasets`

## Scope

Rows whose contract runtime resource class is `igpu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `embedding` | `bge-base-en-v1.5-igpu-intel-linux` | startup_state=warm_process | 15.1ms | 0.9489 | `sync_default` | quality_and_latency_within_tool_budget |
| `llm_chat` | `qwen2.5-1.5b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `llm_chat` | `qwen2.5-7b-intel-linux` | startup_state=warm_process | 12228.3ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen3-0.6b-intel-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `qwen3-0.6b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `llm_summary` | `qwen2.5-1.5b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `llm_summary` | `qwen2.5-7b-intel-linux` | startup_state=warm_process | 12228.3ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen3-0.6b-intel-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen3-0.6b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `rag_answer` | `qwen2.5-1.5b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `rag_answer` | `qwen2.5-7b-intel-linux` | startup_state=warm_process | 12228.3ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen3-0.6b-intel-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen3-0.6b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `rag_search_only` | `bge-base-en-v1.5-igpu-intel-linux` | startup_state=warm_process | 15.1ms | 0.9489 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-base-igpu-intel-linux` | startup_state=warm_process | 3273.9ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `vlm_doc_extract` | `llava-7b-intel-linux` | startup_state=warm_process | - | 0.0000 | `not_recommended` | vlm_document_field_accuracy_failed |
| `vlm_qa` | `llava-7b-intel-linux` | startup_state=warm_process | 24321.5ms | 0.8889 | `not_recommended` | vlm_entity_recall_failed |

## Decision

This hardware condition has 17 contract rows and 3 product-usable rows. Verdict mix: sync_default=2, sync_bounded=1, not_recommended=11, blocked=3.

## Evidence

| Run ID | Artifacts |
|---|---|
| `intel-linux-20260712-contract-full` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260712-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260712-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260712-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260712-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260712-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260712-contract-full/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/nas-contract-report.md) |
