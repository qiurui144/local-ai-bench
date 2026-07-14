# Intel Linux 混合运行时

**最后更新：** 2026-07-14
**英文版本:** [mixed.en.md](mixed.en.md)
**合同来源运行:** `intel-linux-20260712-contract-full`, `intel-linux-20260713-q25-7b-quality`, `intel-linux-20260713-q25-7b-quality-hfdatasets`

## 范围

合同 runtime resource class 为 `mixed` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `asr` | `whisper-tiny-openvino-intel-linux` | startup_state=warm_process | 5008.9ms | 0.3846 | `not_recommended` | FAIL |
| `ocr` | `paddleocr-openvino-intel-linux` | startup_state=warm_process | 1217.0ms | 0.9202 | `sync_default` | quality_and_latency_within_tool_budget |

## 结论

该硬件条件有 2 条合同实测行，其中 1 条为产品可用行。Verdict 分布：sync_default=1, not_recommended=1。

## 证据

| Run ID | 产物 |
|---|---|
| `intel-linux-20260712-contract-full` | [参数矩阵](../../../output/reports/contract/intel-linux-20260712-contract-full/parameter-matrix.json), [运行摘要](../../../output/reports/contract/intel-linux-20260712-contract-full/run-summary.json), [verdict 表](../../../output/reports/contract/intel-linux-20260712-contract-full/verdict-table.tsv), [模型画像](../../../output/reports/contract/intel-linux-20260712-contract-full/model-profile.json), [scheduler 合同](../../../output/reports/contract/intel-linux-20260712-contract-full/scheduler-contract.json), [合同报告](../../../output/reports/contract/intel-linux-20260712-contract-full/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality` | [参数矩阵](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/parameter-matrix.json), [运行摘要](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/run-summary.json), [verdict 表](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/verdict-table.tsv), [模型画像](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/model-profile.json), [scheduler 合同](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/scheduler-contract.json), [合同报告](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | [参数矩阵](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/parameter-matrix.json), [运行摘要](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/run-summary.json), [verdict 表](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/verdict-table.tsv), [模型画像](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/model-profile.json), [scheduler 合同](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/scheduler-contract.json), [合同报告](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/nas-contract-report.md) |
