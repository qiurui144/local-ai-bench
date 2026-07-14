# AMD Linux

**Last updated:** 2026-07-14
**Chinese version:** [index.zh.md](index.zh.md)

## Scope

AMD Linux contract reporting is split by CPU fallback/tool rows, Radeon 780M iGPU/Vulkan rows, Linux NPU probe rows, mixed rows, and blocked-runtime rows.

Rows are grouped by `runtime.resource_class` from the contract matrix. `failed` or `not_recommended` rows are measured evidence, not missing reports. Rows with no concrete runtime are kept under Blocked Runtime.

## Contract Baseline

| Item | Value |
|---|---|
| target | amd-linux-x86 |
| source_runs | `amd-linux-20260713-contract-full`, `amd-linux-x86-20260714-applicable-gapfill` |
| status | partial |
| row_count | 50 |
| sync_default | 7 |
| sync_bounded | 2 |
| not_recommended | 18 |
| blocked | 23 |

## Hardware Path Summary

| Path | Rows | Usable rows | Workloads | Verdict mix | Report |
|---|---:|---:|---|---|---|
| [CPU](cpu.en.md) | 4 | 2 | asr, ocr | sync_default=2, not_recommended=2 | [cpu.en.md](cpu.en.md) |
| [iGPU](igpu.en.md) | 41 | 6 | embedding, llm_chat, llm_summary, ocr, rag_answer, rag_search_only, reranker, vlm_doc_extract, vlm_qa | sync_default=4, sync_bounded=2, not_recommended=15, blocked=20 | [igpu.en.md](igpu.en.md) |
| [NPU](npu.en.md) | 2 | 1 | asr, ocr | sync_default=1, not_recommended=1 | [npu.en.md](npu.en.md) |
| [Mixed Runtime](mixed.en.md) | 0 | 0 | - | - | [mixed.en.md](mixed.en.md) |
| [Blocked Runtime](blocked-runtime.en.md) | 3 | 0 | llm_chat, llm_summary, rag_answer | blocked=3 | [blocked-runtime.en.md](blocked-runtime.en.md) |

## Decision

9 contract rows are product-usable under the current verdict policy. 18 rows are measured but not recommended, and 23 rows remain blocked. Use the hardware subreports for the concrete path decision instead of mixing CPU, iGPU, NPU, and mixed-runtime evidence.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-linux-20260713-contract-full` | [Parameter matrix](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/nas-contract-report.md) |
| `amd-linux-x86-20260714-applicable-gapfill` | [Parameter matrix](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/run-summary.json), [Verdict table](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/nas-contract-report.md) |
