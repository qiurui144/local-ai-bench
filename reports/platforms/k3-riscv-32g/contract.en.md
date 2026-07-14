# K3 RISC-V 32G Contract Supplement

**Last updated:** 2026-07-14
**Chinese version:** [contract.zh.md](contract.zh.md)

## Scope

K3 32G contract reporting is added as a contract supplement to the existing K3 platform reports.

## Contract Rows

| Hardware | Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---|---:|---:|---|---|
| X100 CPU | `llm_chat` | `Qwen3-30B-A3B-Q4_0` | max_output_tokens=32, startup_state=warm_process | 1388.0ms | 1.0000 | `sync_default` | quality_and_latency_within_default_budget |
| X100 CPU | `llm_chat` | `Qwen3-30B-A3B-Q4_0` | context_tokens=1024, target_context_tokens=1024, max_output_tokens=32, startup_state=warm_process, finish_reason=stop | 219305.0ms | 1.0000 | `async_default` | quality_passed_latency_requires_async |
| X100 CPU | `llm_chat` | `Qwen3-30B-A3B-Q4_0` | context_tokens=2048, target_context_tokens=2048, max_output_tokens=32, startup_state=warm_process, finish_reason=stop | 593929.0ms | 1.0000 | `async_only` | quality_passed_latency_exceeds_interactive_async_default |
| X100 CPU | `llm_summary` | `Qwen3-30B-A3B-Q4_0` | context_tokens=37, max_output_tokens=128, startup_state=warm_process, finish_reason=length | 16466.0ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |

## Evidence

| Run ID | Artifacts |
|---|---|
| `k3-32g-qwen30b-contract-20260708` | local artifact dir `output/reports/k3-riscv-32g/contract-qwen30b-20260708_221818/contract`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
