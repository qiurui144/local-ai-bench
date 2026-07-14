# Intel Linux iGPU

**最后更新：** 2026-07-14
**英文版本:** [igpu.en.md](igpu.en.md)
**合同来源运行:** `intel-linux-20260712-contract-full`, `intel-linux-20260713-q25-7b-quality`, `intel-linux-20260713-q25-7b-quality-hfdatasets`

## 范围

合同 runtime resource class 为 `igpu` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `embedding` | `bge-base-en-v1.5-igpu-intel-linux` | startup_state=warm_process | 15.1ms | 0.9489 | `sync_default` | quality_and_latency_within_tool_budget |
| `llm_chat` | `qwen2.5-1.5b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `llm_chat` | `qwen2.5-7b-intel-linux` | startup_state=warm_process | 12228.3ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_chat` | `qwen3-0.6b-intel-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_chat` | `qwen3-0.6b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `llm_summary` | `qwen2.5-1.5b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `llm_summary` | `qwen2.5-7b-intel-linux` | startup_state=warm_process | 12228.3ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `qwen3-0.6b-intel-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `llm_summary` | `qwen3-0.6b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `rag_answer` | `qwen2.5-1.5b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `rag_answer` | `qwen2.5-7b-intel-linux` | startup_state=warm_process | 12228.3ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `qwen3-0.6b-intel-linux` | startup_state=warm_process | - | - | `blocked` | model_process_failed |
| `rag_answer` | `qwen3-0.6b-openvino-intel-linux` | startup_state=warm_process | 0.0ms | - | `not_recommended` | translation_l3_terminology_failed |
| `rag_search_only` | `bge-base-en-v1.5-igpu-intel-linux` | startup_state=warm_process | 15.1ms | 0.9489 | `sync_bounded` | quality_passed_sync_bounded_by_latency |
| `reranker` | `bge-reranker-base-igpu-intel-linux` | startup_state=warm_process | 3273.9ms | 1.0000 | `sync_default` | quality_and_latency_within_tool_budget |
| `vlm_doc_extract` | `llava-7b-intel-linux` | startup_state=warm_process | - | 0.0000 | `not_recommended` | vlm_document_field_accuracy_failed |
| `vlm_qa` | `llava-7b-intel-linux` | startup_state=warm_process | 24321.5ms | 0.8889 | `not_recommended` | vlm_entity_recall_failed |

## 结论

该硬件条件有 17 条合同实测行，其中 3 条为产品可用行。Verdict 分布：sync_default=2, sync_bounded=1, not_recommended=11, blocked=3。

## 证据

| Run ID | 产物 |
|---|---|
| `intel-linux-20260712-contract-full` | 本地证据目录 `output/reports/contract/intel-linux-20260712-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `intel-linux-20260713-q25-7b-quality` | 本地证据目录 `output/reports/contract/intel-linux-20260713-q25-7b-quality`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `intel-linux-20260713-q25-7b-quality-hfdatasets` | 本地证据目录 `output/reports/contract/intel-linux-20260713-q25-7b-quality-hfdatasets`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
