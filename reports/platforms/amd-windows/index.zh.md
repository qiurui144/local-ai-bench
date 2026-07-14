# AMD Windows

**最后更新：** 2026-07-14
**英文版本:** [index.en.md](index.en.md)

## 范围

AMD Windows 合同报告按 CPU、Radeon 780M iGPU、XDNA NPU、混合运行时和阻塞运行时行拆分。

行按合同矩阵里的 `runtime.resource_class` 分组。`failed` 或 `not_recommended` 是已有实测证据，不是报告缺失。没有进入明确运行时的行保留在阻塞运行时页。

## 合同基线

| 项目 | 值 |
|---|---|
| target | amd-win-x86 |
| source_runs | `amd-win-x86-20260712-contract-full` |
| status | complete |
| row_count | 14 |
| sync_default | 6 |
| sync_bounded | 1 |
| not_recommended | 7 |

## 硬件路径摘要

| 路径 | 行数 | 可用行 | 工作负载 | Verdict 分布 | 报告 |
|---|---:|---:|---|---|---|
| [CPU](cpu.zh.md) | 0 | 0 | - | - | [cpu.zh.md](cpu.zh.md) |
| [iGPU](igpu.zh.md) | 9 | 4 | embedding, llm_chat, llm_summary, ocr, rag_answer, rag_search_only, reranker, vlm_doc_extract, vlm_qa | sync_default=3, sync_bounded=1, not_recommended=5 | [igpu.zh.md](igpu.zh.md) |
| [NPU](npu.zh.md) | 4 | 2 | asr, ocr | sync_default=2, not_recommended=2 | [npu.zh.md](npu.zh.md) |
| [混合运行时](mixed.zh.md) | 1 | 1 | asr | sync_default=1 | [mixed.zh.md](mixed.zh.md) |
| [阻塞运行时](blocked-runtime.zh.md) | 0 | 0 | - | - | [blocked-runtime.zh.md](blocked-runtime.zh.md) |

## 结论

当前 verdict 口径下有 7 行可作为产品可用证据。7 行已有实测但不推荐，0 行仍为 blocked。具体选型必须看对应硬件子报告，不要混用 CPU、iGPU、NPU 和混合运行时证据。

## 证据

| Run ID | 产物 |
|---|---|
| `amd-win-x86-20260712-contract-full` | [参数矩阵](../../../output/reports/contract/amd-win-x86-20260712-contract-full/parameter-matrix.json), [运行摘要](../../../output/reports/contract/amd-win-x86-20260712-contract-full/run-summary.json), [verdict 表](../../../output/reports/contract/amd-win-x86-20260712-contract-full/verdict-table.tsv), [模型画像](../../../output/reports/contract/amd-win-x86-20260712-contract-full/model-profile.json), [scheduler 合同](../../../output/reports/contract/amd-win-x86-20260712-contract-full/scheduler-contract.json), [合同报告](../../../output/reports/contract/amd-win-x86-20260712-contract-full/nas-contract-report.md) |
