# AMD Linux 阻塞运行时

**最后更新：** 2026-07-14
**英文版本:** [blocked-runtime.en.md](blocked-runtime.en.md)
**合同来源运行:** `amd-linux-20260713-contract-full`

## 范围

未进入明确硬件运行时的行，统一归为阻塞运行时证据。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `llm_chat` | `qwen2.5-0.5b-amd-linux-onnx` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen2.5-0.5b-amd-linux-onnx` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen2.5-0.5b-amd-linux-onnx` | startup_state=warm_process | - | - | `blocked` | model_process_failed |

## 结论

该硬件条件有 3 条合同实测行，其中 0 条为产品可用行。Verdict 分布：blocked=3。

## 证据

| Run ID | 产物 |
|---|---|
| `amd-linux-20260713-contract-full` | [参数矩阵](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/parameter-matrix.json), [运行摘要](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/run-summary.json), [verdict 表](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/verdict-table.tsv), [模型画像](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/model-profile.json), [scheduler 合同](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/scheduler-contract.json), [合同报告](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/nas-contract-report.md) |
