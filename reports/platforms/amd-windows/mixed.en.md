# AMD Windows Mixed Runtime

**Last updated:** 2026-07-14
**Chinese version:** [mixed.zh.md](mixed.zh.md)
**Contract source runs:** `amd-win-x86-20260712-contract-full`

## Scope

Rows whose contract runtime resource class is `mixed`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `asr` | `sensevoice-small-amd-win` | startup_state=warm_process | 437.0ms | 0.9231 | `sync_default` | quality_and_latency_within_tool_budget |

## Decision

This hardware condition has 1 contract rows and 1 product-usable rows. Verdict mix: sync_default=1.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-win-x86-20260712-contract-full` | [Parameter matrix](../../../output/reports/contract/amd-win-x86-20260712-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/amd-win-x86-20260712-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/amd-win-x86-20260712-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/amd-win-x86-20260712-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/amd-win-x86-20260712-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/amd-win-x86-20260712-contract-full/nas-contract-report.md) |
