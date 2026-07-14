# Intel Linux

**Last updated:** 2026-07-14
**Chinese version:** [index.zh.md](index.zh.md)

## Scope

Intel Linux contract reporting is split by CPU fallback rows, OpenVINO/iGPU rows, NPU rows where present, mixed OpenVINO rows, and blocked-runtime rows.

Rows are grouped by `runtime.resource_class` from the contract matrix. `failed` or `not_recommended` rows are measured evidence, not missing reports. Rows with no concrete runtime are kept under Blocked Runtime.

## Contract Baseline

| Item | Value |
|---|---|
| target | intel-linux |
| source_runs | `intel-linux-20260712-contract-full`, `intel-linux-20260713-q25-7b-quality`, `intel-linux-20260713-q25-7b-quality-hfdatasets` |
| status | partial |
| row_count | 19 |
| sync_default | 3 |
| sync_bounded | 1 |
| not_recommended | 12 |
| blocked | 3 |

## Hardware Path Summary

| Path | Rows | Usable rows | Workloads | Verdict mix | Report |
|---|---:|---:|---|---|---|
| [CPU](cpu.en.md) | 0 | 0 | - | - | [cpu.en.md](cpu.en.md) |
| [iGPU](igpu.en.md) | 17 | 3 | embedding, llm_chat, llm_summary, rag_answer, rag_search_only, reranker, vlm_doc_extract, vlm_qa | sync_default=2, sync_bounded=1, not_recommended=11, blocked=3 | [igpu.en.md](igpu.en.md) |
| [NPU](npu.en.md) | 0 | 0 | - | - | [npu.en.md](npu.en.md) |
| [Mixed Runtime](mixed.en.md) | 2 | 1 | asr, ocr | sync_default=1, not_recommended=1 | [mixed.en.md](mixed.en.md) |
| [Blocked Runtime](blocked-runtime.en.md) | 0 | 0 | - | - | [blocked-runtime.en.md](blocked-runtime.en.md) |

## Decision

4 contract rows are product-usable under the current verdict policy. 12 rows are measured but not recommended, and 3 rows remain blocked. Use the hardware subreports for the concrete path decision instead of mixing CPU, iGPU, NPU, and mixed-runtime evidence.

## Evidence

| Run ID | Artifacts |
|---|---|
| `intel-linux-20260712-contract-full` | local artifact dir `output/reports/contract/intel-linux-20260712-contract-full`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
| `intel-linux-20260713-q25-7b-quality` | local artifact dir `output/reports/contract/intel-linux-20260713-q25-7b-quality`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | local artifact dir `output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
