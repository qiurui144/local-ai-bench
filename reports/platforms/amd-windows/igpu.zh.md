# AMD Windows iGPU

**最后更新：** 2026-07-14
**英文版本:** [igpu.en.md](igpu.en.md)
**合同来源运行:** `amd-win-x86-20260712-contract-full`

## 范围

合同 runtime resource class 为 `igpu` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `embedding` | `bge-base-en-v1.5-igpu-amd-win` | startup_state=warm_process | 2453.0ms | 0.9866 | `sync_default` | quality_and_latency_within_tool_budget |
| `llm_chat` | `qwen2.5-7b-amd-win` | startup_state=warm_process | 6312.8ms | - | `not_recommended` | translation_quality_failed |
| `llm_summary` | `qwen2.5-7b-amd-win` | startup_state=warm_process | 6312.8ms | - | `not_recommended` | translation_quality_failed |
| `ocr` | `rapidocr-amd-directml` | startup_state=warm_process | 518.1ms | 0.9296 | `sync_default` | quality_and_latency_within_tool_budget |
| `rag_answer` | `qwen2.5-7b-amd-win` | startup_state=warm_process | 6312.8ms | - | `not_recommended` | translation_quality_failed |
| `rag_search_only` | `bge-base-en-v1.5-igpu-amd-win` | startup_state=warm_process | 2453.0ms | 0.9866 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-base-igpu-amd-win` | startup_state=warm_process | 4527.7ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `vlm_doc_extract` | `llava-7b-amd-win` | startup_state=warm_process | 13666.2ms | 0.0667 | `not_recommended` | vlm_document_field_accuracy_failed |
| `vlm_qa` | `llava-7b-amd-win` | startup_state=warm_process | 13666.2ms | 0.8889 | `not_recommended` | vlm_entity_recall_failed |

## 结论

该硬件条件有 9 条合同实测行，其中 4 条为产品可用行。Verdict 分布：sync_default=3, sync_bounded=1, not_recommended=5。

## 证据

| Run ID | 产物 |
|---|---|
| `amd-win-x86-20260712-contract-full` | 本地证据目录 `output/reports/contract/amd-win-x86-20260712-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
