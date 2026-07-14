# AMD Linux NPU

**Last updated:** 2026-07-14
**Chinese version:** [npu.zh.md](npu.zh.md)
**Contract source runs:** `amd-linux-20260713-contract-full`

## Scope

Rows whose contract runtime resource class is `npu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `asr` | `whisper-tiny-amd-linux-npu` | startup_state=warm_process | 415.8ms | 0.9231 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `rapidocr-amd-linux-npu` | startup_state=warm_process | - | - | `not_recommended` | SKIP |

## Decision

This hardware condition has 2 contract rows and 1 product-usable rows. Verdict mix: sync_default=1, not_recommended=1.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-linux-20260713-contract-full` | [Parameter matrix](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/nas-contract-report.md) |
