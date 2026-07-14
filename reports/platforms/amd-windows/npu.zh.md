# AMD Windows NPU

**最后更新：** 2026-07-14
**英文版本:** [npu.en.md](npu.en.md)
**合同来源运行:** `amd-win-x86-20260712-contract-full`

## 范围

合同 runtime resource class 为 `npu` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `asr` | `whisper-base-amd-npu` | startup_state=warm_process | 9968.0ms | 0.4615 | `not_recommended` | FAIL |
| `asr` | `whisper-tiny-amd-npu` | startup_state=warm_process | 7734.0ms | 0.4615 | `not_recommended` | FAIL |
| `ocr` | `paddleocr-v4-amd-npu` | startup_state=warm_process | 2295.4ms | 0.9296 | `sync_default` | quality_and_latency_within_tool_budget |
| `ocr` | `rapidocr-amd-npu` | startup_state=warm_process | 2304.2ms | 0.9296 | `sync_default` | quality_and_latency_within_tool_budget |

## 结论

该硬件条件有 4 条合同实测行，其中 2 条为产品可用行。Verdict 分布：sync_default=2, not_recommended=2。

## 证据

| Run ID | 产物 |
|---|---|
| `amd-win-x86-20260712-contract-full` | 本地证据目录 `output/reports/contract/amd-win-x86-20260712-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
