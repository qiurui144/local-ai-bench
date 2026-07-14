# AMD Linux iGPU

**Last updated:** 2026-07-14
**Chinese version:** [igpu.zh.md](igpu.zh.md)
**Contract source runs:** `amd-linux-20260713-contract-full`

## Scope

Rows whose contract runtime resource class is `igpu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `embedding` | `bge-m3-amd-linux` | startup_state=warm_process | 104.5ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `embedding` | `qwen3-embedding-0.6b-amd-linux` | startup_state=warm_process | 89.8ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `llm_chat` | `llama3.2-1b-amd-linux` | startup_state=warm_process | 1076.2ms | - | `blocked` | not_measured |
| `llm_chat` | `llama3.2-3b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `phi-3.5-mini-amd-linux` | startup_state=warm_process | 6172.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen2.5-0.5b-amd-linux` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_chat` | `qwen2.5-1.5b-amd-linux` | startup_state=warm_process | 1117.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen2.5-3b-amd-linux` | startup_state=warm_process | 3445.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen2.5-7b-amd-linux` | startup_state=warm_process | 6712.3ms | - | `not_recommended` | general_ability_blocked |
| `llm_chat` | `qwen3-0.6b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `qwen3-1.7b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `qwen3-4b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `llama3.2-1b-amd-linux` | startup_state=warm_process | 1076.2ms | - | `blocked` | not_measured |
| `llm_summary` | `llama3.2-3b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `phi-3.5-mini-amd-linux` | startup_state=warm_process | 6172.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen2.5-0.5b-amd-linux` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_summary` | `qwen2.5-1.5b-amd-linux` | startup_state=warm_process | 1117.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen2.5-3b-amd-linux` | startup_state=warm_process | 3445.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen2.5-7b-amd-linux` | startup_state=warm_process | 6712.3ms | - | `not_recommended` | general_ability_blocked |
| `llm_summary` | `qwen3-0.6b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen3-1.7b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen3-4b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `ocr` | `rapidocr-amd-linux-directml` | startup_state=warm_process | - | - | `not_recommended` | SKIP |
| `rag_answer` | `llama3.2-1b-amd-linux` | startup_state=warm_process | 1076.2ms | - | `blocked` | not_measured |
| `rag_answer` | `llama3.2-3b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `phi-3.5-mini-amd-linux` | startup_state=warm_process | 6172.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen2.5-0.5b-amd-linux` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `rag_answer` | `qwen2.5-1.5b-amd-linux` | startup_state=warm_process | 1117.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen2.5-3b-amd-linux` | startup_state=warm_process | 3445.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen2.5-7b-amd-linux` | startup_state=warm_process | 6712.3ms | - | `not_recommended` | general_ability_blocked |
| `rag_answer` | `qwen3-0.6b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen3-1.7b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen3-4b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_search_only` | `bge-m3-amd-linux` | startup_state=warm_process | 104.5ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `rag_search_only` | `qwen3-embedding-0.6b-amd-linux` | startup_state=warm_process | 89.8ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `qwen2.5-3b-reranker-amd-linux` | startup_state=warm_process | 190.5ms | 0.9866 | `sync_default` | quality_and_latency_within_tool_budget |
| `reranker` | `qwen2.5-7b-reranker-amd-linux` | startup_state=warm_process | 300.6ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `vlm_doc_extract` | `llava-7b-amd-linux` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `vlm_doc_extract` | `minicpm-v-8b-amd-linux` | startup_state=warm_process | 10883.0ms | - | `blocked` | not_measured |
| `vlm_qa` | `llava-7b-amd-linux` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `vlm_qa` | `minicpm-v-8b-amd-linux` | startup_state=warm_process | 10883.0ms | 0.7778 | `not_recommended` | vlm_entity_recall_failed |

## Decision

This hardware condition has 41 contract rows and 6 product-usable rows. Verdict mix: sync_default=4, sync_bounded=2, not_recommended=14, blocked=21.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-linux-20260713-contract-full` | [Parameter matrix](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/nas-contract-report.md) |
