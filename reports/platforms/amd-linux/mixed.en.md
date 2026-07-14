# AMD Linux Mixed Runtime

**Last updated:** 2026-07-14
**Chinese version:** [mixed.zh.md](mixed.zh.md)
**Contract source runs:** `amd-linux-20260713-contract-full`, `amd-linux-x86-20260714-applicable-gapfill`

## Scope

Rows whose contract runtime resource class is `mixed`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| - | - | - | - | - | - | _No current contract rows for this hardware condition._ |

## Decision

This hardware condition has no current contract evidence. It must not be reported as covered.

## Evidence

| Run ID | Artifacts |
|---|---|
| `amd-linux-20260713-contract-full` | local artifact dir `output/reports/contract/amd-linux-x86-20260713-contract-full`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
| `amd-linux-x86-20260714-applicable-gapfill` | local artifact dir `output/reports/contract/amd-linux-x86-20260714-applicable-gapfill`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
