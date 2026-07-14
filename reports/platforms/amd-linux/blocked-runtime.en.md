# AMD Linux Blocked Runtime

**Last updated:** 2026-07-14
**Chinese version:** [blocked-runtime.zh.md](blocked-runtime.zh.md)
**Contract source runs:** `amd-linux-20260713-contract-full`, `amd-linux-x86-20260714-applicable-gapfill`

## Scope

Rows that did not reach a concrete hardware runtime and are grouped as blocked runtime evidence.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `llm_chat` | `qwen2.5-0.5b-amd-linux-onnx` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen2.5-0.5b-amd-linux-onnx` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen2.5-0.5b-amd-linux-onnx` | startup_state=warm_process | - | - | `blocked` | model_process_failed |

## Decision

This hardware condition has 3 contract rows and 0 product-usable rows. Verdict mix: blocked=3.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-linux-20260713-contract-full` | [Parameter matrix](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/nas-contract-report.md) |
| `amd-linux-x86-20260714-applicable-gapfill` | [Parameter matrix](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/run-summary.json), [Verdict table](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/nas-contract-report.md) |
