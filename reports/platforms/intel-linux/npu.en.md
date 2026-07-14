# Intel Linux NPU

**Last updated:** 2026-07-14
**Chinese version:** [npu.zh.md](npu.zh.md)
**Contract source runs:** `intel-linux-20260712-contract-full`, `intel-linux-20260713-q25-7b-quality`, `intel-linux-20260713-q25-7b-quality-hfdatasets`

## Scope

Rows whose contract runtime resource class is `npu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| - | - | - | - | - | - | _No current contract rows for this hardware condition._ |

## Decision

This hardware condition has no current contract evidence. It must not be reported as covered.

## Evidence

| Run ID | Artifacts |
|---|---|
| `intel-linux-20260712-contract-full` | local artifact dir `output/reports/contract/intel-linux-20260712-contract-full`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
| `intel-linux-20260713-q25-7b-quality` | local artifact dir `output/reports/contract/intel-linux-20260713-q25-7b-quality`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | local artifact dir `output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
