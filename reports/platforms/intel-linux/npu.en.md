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
| `intel-linux-20260712-contract-full` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260712-contract-full/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260712-contract-full/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260712-contract-full/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260712-contract-full/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260712-contract-full/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260712-contract-full/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | [Parameter matrix](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/parameter-matrix.json), [Run summary](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/run-summary.json), [Verdict table](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/verdict-table.tsv), [Model profile](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/model-profile.json), [Scheduler contract](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/scheduler-contract.json), [Contract report](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/nas-contract-report.md) |
