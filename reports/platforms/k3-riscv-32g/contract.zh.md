# K3 RISC-V 32G 合同补充

**最后更新：** 2026-07-14
**英文版本:** [contract.en.md](contract.en.md)

## 范围

K3 32G 合同报告作为现有 K3 平台报告的合同补充页。

## 合同行

| 硬件 | 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---|---:|---:|---|---|
| X100 CPU | `llm_chat` | `Qwen3-30B-A3B-Q4_0` | max_output_tokens=32, startup_state=warm_process | 1388.0ms | 1.0000 | `sync_default` | quality_and_latency_within_default_budget |
| X100 CPU | `llm_chat` | `Qwen3-30B-A3B-Q4_0` | context_tokens=1024, target_context_tokens=1024, max_output_tokens=32, startup_state=warm_process, finish_reason=stop | 219305.0ms | 1.0000 | `async_default` | quality_passed_latency_requires_async |
| X100 CPU | `llm_chat` | `Qwen3-30B-A3B-Q4_0` | context_tokens=2048, target_context_tokens=2048, max_output_tokens=32, startup_state=warm_process, finish_reason=stop | 593929.0ms | 1.0000 | `async_only` | quality_passed_latency_exceeds_interactive_async_default |
| X100 CPU | `llm_summary` | `Qwen3-30B-A3B-Q4_0` | context_tokens=37, max_output_tokens=128, startup_state=warm_process, finish_reason=length | 16466.0ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |

## 证据

| Run ID | 产物 |
|---|---|
| `k3-32g-qwen30b-contract-20260708` | 本地证据目录 `output/reports/k3-riscv-32g/contract-qwen30b-20260708_221818/contract`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
