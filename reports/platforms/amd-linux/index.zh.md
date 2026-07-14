# AMD Linux

**最后更新：** 2026-07-14
**英文版本:** [index.en.md](index.en.md)

## 范围

AMD Linux 合同报告按 CPU fallback/tool、Radeon 780M iGPU/Vulkan、Linux NPU probe、混合运行时和阻塞运行时行拆分。

行按合同矩阵里的 `runtime.resource_class` 分组。`failed` 或 `not_recommended` 是已有实测证据，不是报告缺失。没有进入明确运行时的行保留在阻塞运行时页。

## 合同基线

| 项目 | 值 |
|---|---|
| target | amd-linux-x86 |
| source_runs | `amd-linux-20260713-contract-full`, `amd-linux-x86-20260714-applicable-gapfill` |
| status | partial |
| row_count | 50 |
| sync_default | 7 |
| sync_bounded | 2 |
| not_recommended | 18 |
| blocked | 23 |

## 硬件路径摘要

| 路径 | 行数 | 可用行 | 工作负载 | Verdict 分布 | 报告 |
|---|---:|---:|---|---|---|
| [CPU](cpu.zh.md) | 4 | 2 | asr, ocr | sync_default=2, not_recommended=2 | [cpu.zh.md](cpu.zh.md) |
| [iGPU](igpu.zh.md) | 41 | 6 | embedding, llm_chat, llm_summary, ocr, rag_answer, rag_search_only, reranker, vlm_doc_extract, vlm_qa | sync_default=4, sync_bounded=2, not_recommended=15, blocked=20 | [igpu.zh.md](igpu.zh.md) |
| [NPU](npu.zh.md) | 2 | 1 | asr, ocr | sync_default=1, not_recommended=1 | [npu.zh.md](npu.zh.md) |
| [混合运行时](mixed.zh.md) | 0 | 0 | - | - | [mixed.zh.md](mixed.zh.md) |
| [阻塞运行时](blocked-runtime.zh.md) | 3 | 0 | llm_chat, llm_summary, rag_answer | blocked=3 | [blocked-runtime.zh.md](blocked-runtime.zh.md) |

## 结论

当前 verdict 口径下有 9 行可作为产品可用证据。18 行已有实测但不推荐，23 行仍为 blocked。具体选型必须看对应硬件子报告，不要混用 CPU、iGPU、NPU 和混合运行时证据。

## 证据

| Run ID | 产物 |
|---|---|
| `amd-linux-20260713-contract-full` | [参数矩阵](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/parameter-matrix.json), [运行摘要](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/run-summary.json), [verdict 表](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/verdict-table.tsv), [模型画像](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/model-profile.json), [scheduler 合同](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/scheduler-contract.json), [合同报告](../../../output/reports/contract/amd-linux-x86-20260713-contract-full/nas-contract-report.md) |
| `amd-linux-x86-20260714-applicable-gapfill` | [参数矩阵](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/parameter-matrix.json), [运行摘要](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/run-summary.json), [verdict 表](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/verdict-table.tsv), [模型画像](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/model-profile.json), [scheduler 合同](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/scheduler-contract.json), [合同报告](../../../output/reports/contract/amd-linux-x86-20260714-applicable-gapfill/nas-contract-report.md) |
