# Intel Linux 阻塞运行时

**最后更新：** 2026-07-14
**英文版本:** [blocked-runtime.en.md](blocked-runtime.en.md)
**合同来源运行:** `intel-linux-20260712-contract-full`, `intel-linux-20260713-q25-7b-quality`, `intel-linux-20260713-q25-7b-quality-hfdatasets`

## 范围

未进入明确硬件运行时的行，统一归为阻塞运行时证据。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| - | - | - | - | - | - | _当前没有该硬件条件下的合同实测行。_ |

## 结论

该硬件条件当前没有合同实测证据，不能写成已覆盖。

## 证据

| Run ID | 产物 |
|---|---|
| `intel-linux-20260712-contract-full` | 本地证据目录 `output/reports/contract/intel-linux-20260712-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `intel-linux-20260713-q25-7b-quality` | 本地证据目录 `output/reports/contract/intel-linux-20260713-q25-7b-quality`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | 本地证据目录 `output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
