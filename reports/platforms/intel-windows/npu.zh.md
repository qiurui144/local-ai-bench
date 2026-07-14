# Intel Windows NPU

**最后更新：** 2026-07-14
**英文版本:** [npu.en.md](npu.en.md)
**合同来源运行:** `intel-win-x86-20260713-contract-full`, `intel-win-x86-20260714-applicable-gapfill`

## 范围

合同 runtime resource class 为 `npu` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `asr` | `whisper-base-npu-intel-win` | startup_state=warm_process | 11390.0ms | 0.0000 | `not_recommended` | FAIL |

## 结论

该硬件条件有 1 条合同实测行，其中 0 条为产品可用行。Verdict 分布：not_recommended=1。

## 证据

| Run ID | 产物 |
|---|---|
| `intel-win-x86-20260713-contract-full` | 本地证据目录 `output/reports/contract/intel-win-x86-20260713-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `intel-win-x86-20260714-applicable-gapfill` | 本地证据目录 `output/reports/contract/intel-win-x86-20260714-applicable-gapfill`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
