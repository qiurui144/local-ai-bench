# AMD Linux CPU

**最后更新：** 2026-07-14
**英文版本:** [cpu.en.md](cpu.en.md)
**合同来源运行:** `amd-linux-20260713-contract-full`, `amd-linux-x86-20260714-applicable-gapfill`

## 范围

合同 runtime resource class 为 `cpu` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `asr` | `sensevoice-small-amd-linux` | startup_state=warm_process | 422.0ms | 0.9231 | `sync_default` | quality_and_latency_within_tool_budget |
| `asr` | `sensevoice-small-int8-amd-linux` | startup_state=warm_process | 160.9ms | 0.9231 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `paddleocr-amd-linux` | startup_state=warm_process | - | - | `not_recommended` | SKIP |
| `ocr` | `rapidocr-amd-linux` | startup_state=warm_process | 869.3ms | 0.8263 | `not_recommended` | FAIL |

## 结论

该硬件条件有 4 条合同实测行，其中 2 条为产品可用行。Verdict 分布：sync_default=2, not_recommended=2。

## 证据

| Run ID | 产物 |
|---|---|
| `amd-linux-20260713-contract-full` | 本地证据目录 `output/reports/contract/amd-linux-x86-20260713-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `amd-linux-x86-20260714-applicable-gapfill` | 本地证据目录 `output/reports/contract/amd-linux-x86-20260714-applicable-gapfill`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
