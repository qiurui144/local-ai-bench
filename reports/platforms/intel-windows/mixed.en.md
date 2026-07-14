# Intel Windows Mixed Runtime

**Last updated:** 2026-07-14
**Chinese version:** [mixed.zh.md](mixed.zh.md)
**Contract source runs:** `intel-win-x86-20260713-contract-full`

## Scope

Rows whose contract runtime resource class is `mixed`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `asr` | `sensevoice-small-intel-win` | startup_state=warm_process | 1859.0ms | 0.9231 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `paddleocr-openvino-intel-win` | startup_state=warm_process | 1640.5ms | 0.9202 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `rapidocr-intel-openvino` | startup_state=warm_process | 5305.5ms | 0.9202 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-base-intel-win` | startup_state=warm_process | 16656.2ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-v2-m3-intel-win` | startup_state=warm_process | 17031.5ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |

## Decision

This hardware condition has 5 contract rows and 5 product-usable rows. Verdict mix: sync_default=2, sync_bounded=3.

## Evidence

| Run ID | Artifacts |
|---|---|
| `intel-win-x86-20260713-contract-full` | [Parameter matrix](../../../output/reports/contract/intel-win-x86-20260713-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-win-x86-20260713-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/intel-win-x86-20260713-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-win-x86-20260713-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-win-x86-20260713-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-win-x86-20260713-contract-full/nas-contract-report.md) |
