# AMD Windows NPU

**Last updated:** 2026-07-14
**Chinese version:** [npu.zh.md](npu.zh.md)
**Contract source runs:** `amd-win-x86-20260712-contract-full`

## Scope

Rows whose contract runtime resource class is `npu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `asr` | `whisper-base-amd-npu` | startup_state=warm_process | 9968.0ms | 0.4615 | `not_recommended` | FAIL |
| `asr` | `whisper-tiny-amd-npu` | startup_state=warm_process | 7734.0ms | 0.4615 | `not_recommended` | FAIL |
| `ocr` | `paddleocr-v4-amd-npu` | startup_state=warm_process | 2295.4ms | 0.9296 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `rapidocr-amd-npu` | startup_state=warm_process | 2304.2ms | 0.9296 | `sync_default` | quality_and_latency_within_tool_budget |

## Decision

This hardware condition has 4 contract rows and 2 product-usable rows. Verdict mix: sync_default=2, not_recommended=2.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-win-x86-20260712-contract-full` | local artifact dir `output/reports/contract/amd-win-x86-20260712-contract-full`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
