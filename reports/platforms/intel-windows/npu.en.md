# Intel Windows NPU

**Last updated:** 2026-07-14
**Chinese version:** [npu.zh.md](npu.zh.md)
**Contract source runs:** `intel-win-x86-20260713-contract-full`, `intel-win-x86-20260714-applicable-gapfill`

## Scope

Rows whose contract runtime resource class is `npu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `asr` | `whisper-base-npu-intel-win` | startup_state=warm_process | 11390.0ms | 0.0000 | `not_recommended` | FAIL |

## Decision

This hardware condition has 1 contract rows and 0 product-usable rows. Verdict mix: not_recommended=1.

## Evidence

| Run ID | Artifacts |
|---|---|
| `intel-win-x86-20260713-contract-full` | local artifact dir `output/reports/contract/intel-win-x86-20260713-contract-full`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
| `intel-win-x86-20260714-applicable-gapfill` | local artifact dir `output/reports/contract/intel-win-x86-20260714-applicable-gapfill`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
