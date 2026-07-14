# Intel Windows CPU

**Last updated:** 2026-07-14
**Chinese version:** [cpu.zh.md](cpu.zh.md)
**Contract source runs:** `intel-win-x86-20260713-contract-full`, `intel-win-x86-20260714-applicable-gapfill`

## Scope

Rows whose contract runtime resource class is `cpu`.

## Workload Results

| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |
|---|---|---|---:|---:|---|---|
| `llm_chat` | `llama3.2-1b-intel-win` | startup_state=warm_process | 3696.8ms | - | `blocked` | not_measured |
| `llm_chat` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | 4884.7ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `llm_summary` | `llama3.2-1b-intel-win` | startup_state=warm_process | 3696.8ms | - | `blocked` | not_measured |
| `llm_summary` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | 4884.7ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `rag_answer` | `llama3.2-1b-intel-win` | startup_state=warm_process | 3696.8ms | - | `blocked` | not_measured |
| `rag_answer` | `qwen2.5-1.5b-intel-win` | startup_state=warm_process | 4884.7ms | - | `not_recommended` | translation_quality_and_terminology_failed |
| `vlm_doc_extract` | `llava-7b-intel-win` | startup_state=warm_process | 32484.8ms | - | `blocked` | not_measured |
| `vlm_qa` | `llava-7b-intel-win` | startup_state=warm_process | 32484.8ms | 1.0000 | `not_recommended` | vlm_entity_recall_failed |

## Decision

This hardware condition has 8 contract rows and 0 product-usable rows. Verdict mix: not_recommended=4, blocked=4.

## Evidence

| Run ID | Artifacts |
|---|---|
| `intel-win-x86-20260713-contract-full` | local artifact dir `output/reports/contract/intel-win-x86-20260713-contract-full`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
| `intel-win-x86-20260714-applicable-gapfill` | local artifact dir `output/reports/contract/intel-win-x86-20260714-applicable-gapfill`; Parameter matrix `parameter-matrix.json`, Run summary `run-summary.json`, Verdict table `verdict-table.tsv`, Model profile `model-profile.json`, Scheduler contract `scheduler-contract.json`, Contract report `nas-contract-report.md` |
