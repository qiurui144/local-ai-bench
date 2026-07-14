# Intel Windows iGPU

**最后更新：** 2026-07-14
**英文版本:** [igpu.en.md](igpu.en.md)
**合同来源运行:** `intel-win-x86-20260713-contract-full`

## 范围

合同 runtime resource class 为 `igpu` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `asr` | `whisper-tiny-openvino-intel-win` | startup_state=warm_process | 69250.0ms | 0.0000 | `not_recommended` | FAIL |
| `embedding` | `bge-base-en-v1.5-igpu-intel-win` | startup_state=warm_process | 0.0ms | 0.0000 | `not_recommended` | FAIL |
| `embedding` | `bge-m3-intel-win` | startup_state=warm_process | 846.0ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `embedding` | `qwen3-embedding-0.6b-intel-win` | startup_state=warm_process | 662.7ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `llm_chat` | `llama3.2-1b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_chat` | `phi-3.5-mini-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_chat` | `qwen2.5-0.5b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_chat` | `qwen2.5-1.5b-igpu-intel-win` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_chat` | `qwen2.5-3b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_chat` | `qwen2.5-7b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `qwen2.5-7b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_chat` | `qwen2.5-coder-0.5b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_chat` | `qwen3-0.6b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_chat` | `qwen3-0.6b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_chat` | `qwen3-1.7b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_chat` | `qwen3-1.7b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_chat` | `qwen3-4b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_chat` | `qwen3-4b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `llama3.2-1b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `phi-3.5-mini-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `qwen2.5-0.5b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `qwen2.5-1.5b-igpu-intel-win` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `qwen2.5-3b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `qwen2.5-7b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen2.5-7b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `qwen2.5-coder-0.5b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_summary` | `qwen3-0.6b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_summary` | `qwen3-0.6b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `qwen3-1.7b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_summary` | `qwen3-1.7b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `llm_summary` | `qwen3-4b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `llm_summary` | `qwen3-4b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `ocr` | `rapidocr-intel-directml` | startup_state=warm_process | - | - | `not_recommended` | SKIP |
| `rag_answer` | `llama3.2-1b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_answer` | `phi-3.5-mini-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_answer` | `qwen2.5-0.5b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_answer` | `qwen2.5-1.5b-igpu-intel-win` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_answer` | `qwen2.5-3b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_answer` | `qwen2.5-7b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen2.5-7b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_answer` | `qwen2.5-coder-0.5b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `rag_answer` | `qwen3-0.6b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `rag_answer` | `qwen3-0.6b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_answer` | `qwen3-1.7b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `rag_answer` | `qwen3-1.7b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_answer` | `qwen3-4b-igpu-intel-win` | startup_state=warm_process | - | - | `blocked` | runtime_repair_failed |
| `rag_answer` | `qwen3-4b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `rag_search_only` | `bge-base-en-v1.5-igpu-intel-win` | startup_state=warm_process | 0.0ms | 0.0000 | `not_recommended` | FAIL |
| `rag_search_only` | `bge-m3-intel-win` | startup_state=warm_process | 846.0ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `rag_search_only` | `qwen3-embedding-0.6b-intel-win` | startup_state=warm_process | 662.7ms | 1.0000 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-base-igpu-intel-win` | startup_state=warm_process | 0.0ms | 0.9056 | `sync_default` | quality_and_latency_within_tool_budget |
| `vlm_doc_extract` | `llava-7b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |
| `vlm_qa` | `llava-7b-intel-win` | startup_state=warm_process | - | - | `blocked` | cpu_only_llm_vlm_blocked |

## 结论

该硬件条件有 56 条合同实测行，其中 5 条为产品可用行。Verdict 分布：sync_default=3, sync_bounded=2, not_recommended=7, blocked=44。

## 证据

| Run ID | 产物 |
|---|---|
| `intel-win-x86-20260713-contract-full` | [参数矩阵](../../../output/reports/contract/intel-win-x86-20260713-contract-full/parameter-matrix.json), [运行摘要](../../../output/reports/contract/intel-win-x86-20260713-contract-full/run-summary.json), [verdict 表](../../../output/reports/contract/intel-win-x86-20260713-contract-full/verdict-table.tsv), [模型画像](../../../output/reports/contract/intel-win-x86-20260713-contract-full/model-profile.json), [scheduler 合同](../../../output/reports/contract/intel-win-x86-20260713-contract-full/scheduler-contract.json), [合同报告](../../../output/reports/contract/intel-win-x86-20260713-contract-full/nas-contract-report.md) |
