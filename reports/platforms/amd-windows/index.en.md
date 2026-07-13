# AMD Windows

**Last updated:** 2026-07-09
**Chinese version:** [index.zh.md](index.zh.md)

## Scope

This page records the AMD Windows contract baseline. This run aligns the NAS contract items on their selected same-stack paths only; it does not compare llama.cpp, Ollama, CPU-only, or other cross-stack dimensions.

Target host: Windows 11 / Ryzen 7 8845H / Radeon 780M / 27.75GB RAM. Coverage includes 10 contract items across LLM, RAG, embedding, reranker, VLM, OCR, and ASR.

## Contract Baseline

| Item | Value |
|---|---|
| target | `amd-win-x86` |
| run_id | `amd-20260709-baseline-contract-s1-final` |
| status | `complete` |
| row_count | 10 |
| blocked_test_items | 0 |
| sync_default | 4 |
| sync_bounded | 1 |
| not_recommended | 5 |

## Execution Path Summary

| Workload | Baseline model/path | Runtime | Verdict |
|---|---|---|---|
| LLM / RAG answer | `qwen2.5-7b-amd-win` | Ollama AMD/Vulkan service path | `not_recommended` |
| VLM | `llava-7b-amd-win` | Ollama AMD/VLM service path | `not_recommended` |
| Embedding / RAG search-only | `bge-base-en-v1.5-igpu-amd-win` | ONNX Runtime DirectML service path | `sync_default` / `sync_bounded` |
| Reranker | `bge-reranker-base-igpu-amd-win` | ONNX Runtime DirectML service path | `sync_default` |
| OCR | `rapidocr-amd-directml` | DirectML OCR path | `sync_default` |
| ASR | `sensevoice-small-amd-win` | AMD Windows ASR path | `sync_default` |

## Decision

The AMD Windows paths that are currently usable as NAS product sync-default baselines are OCR, ASR, embedding, and reranker. RAG search-only is usable as a bounded sync fallback.

The `qwen2.5-7b-amd-win` LLM/RAG answer rows and the `llava-7b-amd-win` VLM rows have measured performance and quality outputs, but their contract verdict is `not_recommended` because quality did not meet the current productization gate.

## Evidence

| Artifact | Path |
|---|---|
| Contract report | [nas-contract-report.md](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/nas-contract-report.md) |
| Parameter matrix | [parameter-matrix.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/parameter-matrix.json) |
| Run summary | [run-summary.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/run-summary.json) |
| Verdict table | [verdict-table.tsv](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/verdict-table.tsv) |
| Model profile | [model-profile.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/model-profile.json) |
| Scheduler contract | [scheduler-contract.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/scheduler-contract.json) |
| Main summary | [amd-20260709-baseline-contract-s1-capped_summary.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-capped_summary.json) |
| VLM supplemental summary | [amd-20260709-baseline-contract-s1-capped-vlm-scenarios_summary.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-capped-vlm-scenarios_summary.json) |
