# AMD Linux iGPU

**最后更新：** 2026-07-14
**英文版本:** [igpu.en.md](igpu.en.md)
**合同来源运行:** `amd-linux-20260713-contract-full`, `amd-linux-x86-20260714-applicable-gapfill`

## 范围

合同 runtime resource class 为 `igpu` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `embedding` | `bge-m3-amd-linux` | startup_state=warm_process | 104.5ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `embedding` | `qwen3-embedding-0.6b-amd-linux` | startup_state=warm_process | 89.8ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `llm_chat` | `llama3.2-1b-amd-linux` | startup_state=warm_process | 0.0ms | - | `blocked` | not_measured |
| `llm_chat` | `llama3.2-3b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `phi-3.5-mini-amd-linux` | startup_state=warm_process | 6172.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen2.5-0.5b-amd-linux` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_chat` | `qwen2.5-1.5b-amd-linux` | startup_state=warm_process | 1117.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen2.5-3b-amd-linux` | startup_state=warm_process | 3445.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen2.5-7b-amd-linux` | startup_state=warm_process | 6712.3ms | - | `not_recommended` | general_ability_blocked |
| `llm_chat` | `qwen3-0.6b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `qwen3-1.7b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `qwen3-4b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `llama3.2-1b-amd-linux` | startup_state=warm_process | 0.0ms | - | `blocked` | not_measured |
| `llm_summary` | `llama3.2-3b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `phi-3.5-mini-amd-linux` | startup_state=warm_process | 6172.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen2.5-0.5b-amd-linux` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_summary` | `qwen2.5-1.5b-amd-linux` | startup_state=warm_process | 1117.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen2.5-3b-amd-linux` | startup_state=warm_process | 3445.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen2.5-7b-amd-linux` | startup_state=warm_process | 6712.3ms | - | `not_recommended` | general_ability_blocked |
| `llm_summary` | `qwen3-0.6b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen3-1.7b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen3-4b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `ocr` | `rapidocr-amd-linux-directml` | startup_state=warm_process | - | - | `not_recommended` | SKIP |
| `rag_answer` | `llama3.2-1b-amd-linux` | startup_state=warm_process | 0.0ms | - | `blocked` | not_measured |
| `rag_answer` | `llama3.2-3b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `phi-3.5-mini-amd-linux` | startup_state=warm_process | 6172.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen2.5-0.5b-amd-linux` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `rag_answer` | `qwen2.5-1.5b-amd-linux` | startup_state=warm_process | 1117.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen2.5-3b-amd-linux` | startup_state=warm_process | 3445.9ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen2.5-7b-amd-linux` | startup_state=warm_process | 6712.3ms | - | `not_recommended` | general_ability_blocked |
| `rag_answer` | `qwen3-0.6b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen3-1.7b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen3-4b-amd-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_search_only` | `bge-m3-amd-linux` | startup_state=warm_process | 104.5ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `rag_search_only` | `qwen3-embedding-0.6b-amd-linux` | startup_state=warm_process | 89.8ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `qwen2.5-3b-reranker-amd-linux` | startup_state=warm_process | 190.5ms | 0.9866 | `sync_default` | quality_and_latency_within_tool_budget |
| `reranker` | `qwen2.5-7b-reranker-amd-linux` | startup_state=warm_process | 300.6ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `vlm_doc_extract` | `llava-7b-amd-linux` | startup_state=warm_process | 12658.8ms | - | `blocked` | not_measured |
| `vlm_doc_extract` | `minicpm-v-8b-amd-linux` | startup_state=warm_process | 9983.5ms | - | `blocked` | not_measured |
| `vlm_qa` | `llava-7b-amd-linux` | startup_state=warm_process | 12658.8ms | 1.0000 | `not_recommended` | vlm_entity_recall_failed |
| `vlm_qa` | `minicpm-v-8b-amd-linux` | startup_state=warm_process | 9983.5ms | 0.6667 | `not_recommended` | vlm_entity_recall_failed |

## 结论

该硬件条件有 41 条合同实测行，其中 6 条为产品可用行。Verdict 分布：sync_default=4, sync_bounded=2, not_recommended=15, blocked=20。

## 证据

| Run ID | 产物 |
|---|---|
| `amd-linux-20260713-contract-full` | 本地证据目录 `output/reports/contract/amd-linux-x86-20260713-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `amd-linux-x86-20260714-applicable-gapfill` | 本地证据目录 `output/reports/contract/amd-linux-x86-20260714-applicable-gapfill`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
