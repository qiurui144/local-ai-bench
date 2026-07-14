# Intel Windows 混合运行时

**最后更新：** 2026-07-14
**英文版本:** [mixed.en.md](mixed.en.md)
**合同来源运行:** `intel-win-x86-20260713-contract-full`, `intel-win-x86-20260714-applicable-gapfill`

## 范围

合同 runtime resource class 为 `mixed` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `asr` | `sensevoice-small-intel-win` | startup_state=warm_process | 1859.0ms | 0.9231 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `paddleocr-openvino-intel-win` | startup_state=warm_process | 1640.5ms | 0.9202 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `rapidocr-intel-openvino` | startup_state=warm_process | 5305.5ms | 0.9202 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-base-intel-win` | startup_state=warm_process | 16656.2ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-v2-m3-intel-win` | startup_state=warm_process | 17031.5ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |

## 结论

该硬件条件有 5 条合同实测行，其中 5 条为产品可用行。Verdict 分布：sync_default=2, sync_bounded=3。

## 证据

| Run ID | 产物 |
|---|---|
| `intel-win-x86-20260713-contract-full` | 本地证据目录 `output/reports/contract/intel-win-x86-20260713-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `intel-win-x86-20260714-applicable-gapfill` | 本地证据目录 `output/reports/contract/intel-win-x86-20260714-applicable-gapfill`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
