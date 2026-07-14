# Intel Windows

**Last updated:** 2026-07-14
**Chinese version:** [index.zh.md](index.zh.md)

## Scope

Intel Windows contract reporting is split by CPU, Intel Arc iGPU, AI Boost NPU, mixed OpenVINO/runtime rows, and blocked-runtime rows.

Rows are grouped by `runtime.resource_class` from the contract matrix. `failed` or `not_recommended` rows are measured evidence, not missing reports. Rows with no concrete runtime are kept under Blocked Runtime.

## Contract Baseline

| Item | Value |
|---|---|
| target | intel-win-x86 |
| source_runs | `intel-win-x86-20260713-contract-full`, `intel-win-x86-20260714-applicable-gapfill` |
| status | partial |
| row_count | 70 |
| sync_default | 5 |
| sync_bounded | 5 |
| not_recommended | 12 |
| blocked | 48 |

## Hardware Path Summary

| Path | Rows | Usable rows | Workloads | Verdict mix | Report |
|---|---:|---:|---|---|---|
| [CPU](cpu.en.md) | 8 | 0 | llm_chat, llm_summary, rag_answer, vlm_doc_extract, vlm_qa | not_recommended=4, blocked=4 | [cpu.en.md](cpu.en.md) |
| [iGPU](igpu.en.md) | 56 | 5 | asr, embedding, llm_chat, llm_summary, ocr, rag_answer, rag_search_only, reranker, vlm_doc_extract, vlm_qa | sync_default=3, sync_bounded=2, not_recommended=7, blocked=44 | [igpu.en.md](igpu.en.md) |
| [NPU](npu.en.md) | 1 | 0 | asr | not_recommended=1 | [npu.en.md](npu.en.md) |
| [Mixed Runtime](mixed.en.md) | 5 | 5 | asr, ocr, reranker | sync_default=2, sync_bounded=3 | [mixed.en.md](mixed.en.md) |
| [Blocked Runtime](blocked-runtime.en.md) | 0 | 0 | - | - | [blocked-runtime.en.md](blocked-runtime.en.md) |

## Decision

10 contract rows are product-usable under the current verdict policy. 12 rows are measured but not recommended, and 48 rows remain blocked. Use the hardware subreports for the concrete path decision instead of mixing CPU, iGPU, NPU, and mixed-runtime evidence.

## Evidence

| Run ID | Artifacts |
|---|---|
| `intel-win-x86-20260713-contract-full` | local artifact dir `output/reports/contract/intel-win-x86-20260713-contract-full`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
| `intel-win-x86-20260714-applicable-gapfill` | local artifact dir `output/reports/contract/intel-win-x86-20260714-applicable-gapfill`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
