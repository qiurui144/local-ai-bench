# AMD Windows iGPU

**Last updated:** 2026-07-14
**Chinese version:** [igpu.zh.md](igpu.zh.md)
**Contract source runs:** `amd-win-x86-20260712-contract-full`

## Scope

Rows whose contract runtime resource class is `igpu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `embedding` | `bge-base-en-v1.5-igpu-amd-win` | startup_state=warm_process | 2453.0ms | 0.9866 | `sync_default` | quality_and_latency_within_tool_budget |
| `llm_chat` | `qwen2.5-7b-amd-win` | startup_state=warm_process | 6312.8ms | - | `not_recommended` | translation_quality_failed |
| `llm_summary` | `qwen2.5-7b-amd-win` | startup_state=warm_process | 6312.8ms | - | `not_recommended` | translation_quality_failed |
| `ocr` | `rapidocr-amd-directml` | startup_state=warm_process | 518.1ms | 0.9296 | `sync_default` | quality_and_latency_within_tool_budget |
| `rag_answer` | `qwen2.5-7b-amd-win` | startup_state=warm_process | 6312.8ms | - | `not_recommended` | translation_quality_failed |
| `rag_search_only` | `bge-base-en-v1.5-igpu-amd-win` | startup_state=warm_process | 2453.0ms | 0.9866 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-base-igpu-amd-win` | startup_state=warm_process | 4527.7ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `vlm_doc_extract` | `llava-7b-amd-win` | startup_state=warm_process | 13666.2ms | 0.0667 | `not_recommended` | vlm_document_field_accuracy_failed |
| `vlm_qa` | `llava-7b-amd-win` | startup_state=warm_process | 13666.2ms | 0.8889 | `not_recommended` | vlm_entity_recall_failed |

## Decision

This hardware condition has 9 contract rows and 4 product-usable rows. Verdict mix: sync_default=3, sync_bounded=1, not_recommended=5.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-win-x86-20260712-contract-full` | [Parameter matrix](../../../output/reports/contract/amd-win-x86-20260712-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-win-x86-20260712-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/amd-win-x86-20260712-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-win-x86-20260712-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-win-x86-20260712-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-win-x86-20260712-contract-full/nas-contract-report.md) |
