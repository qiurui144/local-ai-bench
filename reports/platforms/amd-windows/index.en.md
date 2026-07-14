# AMD Windows

**Last updated:** 2026-07-14
**Chinese version:** [index.zh.md](index.zh.md)

## Scope

AMD Windows contract reporting is split by CPU, Radeon 780M iGPU, XDNA NPU, mixed runtime, and blocked-runtime rows.

Rows are grouped by `runtime.resource_class` from the contract matrix. `failed` or `not_recommended` rows are measured evidence, not missing reports. Rows with no concrete runtime are kept under Blocked Runtime.

## Contract Baseline

| Item | Value |
|---|---|
| target | amd-win-x86 |
| source_runs | `amd-win-x86-20260712-contract-full` |
| status | complete |
| row_count | 14 |
| sync_default | 6 |
| sync_bounded | 1 |
| not_recommended | 7 |

## Hardware Path Summary

| Path | Rows | Usable rows | Workloads | Verdict mix | Report |
|---|---:|---:|---|---|---|
| [CPU](cpu.en.md) | 0 | 0 | - | - | [cpu.en.md](cpu.en.md) |
| [iGPU](igpu.en.md) | 9 | 4 | embedding, llm_chat, llm_summary, ocr, rag_answer, rag_search_only, reranker, vlm_doc_extract, vlm_qa | sync_default=3, sync_bounded=1, not_recommended=5 | [igpu.en.md](igpu.en.md) |
| [NPU](npu.en.md) | 4 | 2 | asr, ocr | sync_default=2, not_recommended=2 | [npu.en.md](npu.en.md) |
| [Mixed Runtime](mixed.en.md) | 1 | 1 | asr | sync_default=1 | [mixed.en.md](mixed.en.md) |
| [Blocked Runtime](blocked-runtime.en.md) | 0 | 0 | - | - | [blocked-runtime.en.md](blocked-runtime.en.md) |

## Decision

7 contract rows are product-usable under the current verdict policy. 7 rows are measured but not recommended, and 0 rows remain blocked. Use the hardware subreports for the concrete path decision instead of mixing CPU, iGPU, NPU, and mixed-runtime evidence.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-win-x86-20260712-contract-full` | local artifact dir `output/reports/contract/amd-win-x86-20260712-contract-full`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
