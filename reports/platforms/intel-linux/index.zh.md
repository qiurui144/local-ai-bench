# Intel Linux

**最后更新：** 2026-07-14
**英文版本:** [index.en.md](index.en.md)

## 范围

Intel Linux 合同报告按 CPU fallback、OpenVINO/iGPU、可用 NPU、OpenVINO 混合运行时和阻塞运行时行拆分。

行按合同矩阵里的 `runtime.resource_class` 分组。`failed` 或 `not_recommended` 是已有实测证据，不是报告缺失。没有进入明确运行时的行保留在阻塞运行时页。

## 合同基线

| 项目 | 值 |
|---|---|
| target | intel-linux |
| source_runs | `intel-linux-20260712-contract-full`, `intel-linux-20260713-q25-7b-quality`, `intel-linux-20260713-q25-7b-quality-hfdatasets` |
| status | partial |
| row_count | 19 |
| sync_default | 3 |
| sync_bounded | 1 |
| not_recommended | 12 |
| blocked | 3 |

## 硬件路径摘要

| 路径 | 行数 | 可用行 | 工作负载 | Verdict 分布 | 报告 |
|---|---:|---:|---|---|---|
| [CPU](cpu.zh.md) | 0 | 0 | - | - | [cpu.zh.md](cpu.zh.md) |
| [iGPU](igpu.zh.md) | 17 | 3 | embedding, llm_chat, llm_summary, rag_answer, rag_search_only, reranker, vlm_doc_extract, vlm_qa | sync_default=2, sync_bounded=1, not_recommended=11, blocked=3 | [igpu.zh.md](igpu.zh.md) |
| [NPU](npu.zh.md) | 0 | 0 | - | - | [npu.zh.md](npu.zh.md) |
| [混合运行时](mixed.zh.md) | 2 | 1 | asr, ocr | sync_default=1, not_recommended=1 | [mixed.zh.md](mixed.zh.md) |
| [阻塞运行时](blocked-runtime.zh.md) | 0 | 0 | - | - | [blocked-runtime.zh.md](blocked-runtime.zh.md) |

## 结论

当前 verdict 口径下有 4 行可作为产品可用证据。12 行已有实测但不推荐，3 行仍为 blocked。具体选型必须看对应硬件子报告，不要混用 CPU、iGPU、NPU 和混合运行时证据。

## 证据

| Run ID | 产物 |
|---|---|
| `intel-linux-20260712-contract-full` | [参数矩阵](../../../output/reports/contract/intel-linux-20260712-contract-full/parameter-matrix.json), [运行摘要](../../../output/reports/contract/intel-linux-20260712-contract-full/run-summary.json), [verdict 表](../../../output/reports/contract/intel-linux-20260712-contract-full/verdict-table.tsv), [模型画像](../../../output/reports/contract/intel-linux-20260712-contract-full/model-profile.json), [scheduler 合同](../../../output/reports/contract/intel-linux-20260712-contract-full/scheduler-contract.json), [合同报告](../../../output/reports/contract/intel-linux-20260712-contract-full/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality` | [参数矩阵](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/parameter-matrix.json), [运行摘要](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/run-summary.json), [verdict 表](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/verdict-table.tsv), [模型画像](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/model-profile.json), [scheduler 合同](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/scheduler-contract.json), [合同报告](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality/nas-contract-report.md) |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | [参数矩阵](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/parameter-matrix.json), [运行摘要](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/run-summary.json), [verdict 表](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/verdict-table.tsv), [模型画像](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/model-profile.json), [scheduler 合同](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/scheduler-contract.json), [合同报告](../../../output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets/nas-contract-report.md) |
