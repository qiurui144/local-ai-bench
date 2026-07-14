# Intel Linux Mixed Runtime

**Last updated:** 2026-07-14
**Chinese version:** [mixed.zh.md](mixed.zh.md)
**Contract source runs:** `intel-linux-20260712-contract-full`, `intel-linux-20260713-q25-7b-quality`, `intel-linux-20260713-q25-7b-quality-hfdatasets`

## Scope

Rows whose contract runtime resource class is `mixed`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `asr` | `whisper-tiny-openvino-intel-linux` | startup_state=warm_process | 5008.9ms | 0.3846 | `not_recommended` | FAIL |
| `ocr` | `paddleocr-openvino-intel-linux` | startup_state=warm_process | 1217.0ms | 0.9202 | `sync_default` | quality_and_latency_within_tool_budget |

## Decision

This hardware condition has 2 contract rows and 1 product-usable rows. Verdict mix: sync_default=1, not_recommended=1.

## Evidence

| Run ID | Artifacts |
|---|---|
| `intel-linux-20260712-contract-full` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260712-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260712-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260712-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260712-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260712-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260712-contract-full/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/nas-contract-report.md) |
