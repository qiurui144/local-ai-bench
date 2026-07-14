# Intel Windows CPU

**最后更新：** 2026-07-14
**英文版本:** [cpu.en.md](cpu.en.md)
**合同来源运行:** `intel-win-x86-20260713-contract-full`, `intel-win-x86-20260714-applicable-gapfill`

## 范围

合同 runtime resource class 为 `cpu` 的行。

## 工作负载结果

| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |
|---|---|---|---:|---:|---|---|
| `llm_chat` | `llama3.2-1b-intel-win` | startup_state=warm_process | 3696.8ms | - | `blocked` | not_measured |
| `llm_chat` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | 4884.7ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `llama3.2-1b-intel-win` | startup_state=warm_process | 3696.8ms | - | `blocked` | not_measured |
| `llm_summary` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | 4884.7ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `llama3.2-1b-intel-win` | startup_state=warm_process | 3696.8ms | - | `blocked` | not_measured |
| `rag_answer` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | 4884.7ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `vlm_doc_extract` | `llava-7b-intel-win` | startup_state=warm_process | 32484.8ms | - | `blocked` | not_measured |
| `vlm_qa` | `llava-7b-intel-win` | startup_state=warm_process | 32484.8ms | 1.0000 | `not_recommended` | vlm_entity_recall_failed |

## 结论

该硬件条件有 8 条合同实测行，其中 0 条为产品可用行。Verdict 分布：not_recommended=4, blocked=4。

## 证据

| Run ID | 产物 |
|---|---|
| `intel-win-x86-20260713-contract-full` | 本地证据目录 `output/reports/contract/intel-win-x86-20260713-contract-full`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
| `intel-win-x86-20260714-applicable-gapfill` | 本地证据目录 `output/reports/contract/intel-win-x86-20260714-applicable-gapfill`；参数矩阵 `parameter-matrix.json`，运行摘要 `run-summary.json`，verdict 表 `verdict-table.tsv`，模型画像 `model-profile.json`，scheduler 合同 `scheduler-contract.json`，合同报告 `nas-contract-report.md` |
