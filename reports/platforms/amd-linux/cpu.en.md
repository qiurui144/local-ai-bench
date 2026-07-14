# AMD Linux CPU

**Last updated:** 2026-07-14
**Chinese version:** [cpu.zh.md](cpu.zh.md)
**Contract source runs:** `amd-linux-20260713-contract-full`

## Scope

Rows whose contract runtime resource class is `cpu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `asr` | `sensevoice-small-amd-linux` | startup_state=warm_process | 422.0ms | 0.9231 | `sync_default` | quality_and_latency_within_tool_budget |
| `asr` | `sensevoice-small-int8-amd-linux` | startup_state=warm_process | 160.9ms | 0.9231 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `paddleocr-amd-linux` | startup_state=warm_process | - | - | `not_recommended` | SKIP |
| `ocr` | `rapidocr-amd-linux` | startup_state=warm_process | 869.3ms | 0.8263 | `not_recommended` | FAIL |

## Decision

This hardware condition has 4 contract rows and 2 product-usable rows. Verdict mix: sync_default=2, not_recommended=2.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-linux-20260713-contract-full` | [Parameter matrix](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/nas-contract-report.md) |
