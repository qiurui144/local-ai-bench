# AMD Windows iGPU Path

**Last updated:** 2026-07-09
**Chinese version:** [igpu.zh.md](igpu.zh.md)

## Scope

This page records only the iGPU/DirectML-related paths from the AMD Windows contract baseline. It does not compare different software stacks. LLM/VLM uses the Ollama AMD service path; embedding, reranker, and OCR use DirectML-related paths.

## Workload Results

| Workload | Model/path | p95 latency | Quality score | Verdict |
|---|---|---:|---:|---|
| LLM chat | `qwen2.5-7b-amd-win` | 6312.8ms | - | `not_recommended` |
| LLM summary | `qwen2.5-7b-amd-win` | 6312.8ms | - | `not_recommended` |
| RAG answer | `qwen2.5-7b-amd-win` | 6312.8ms | - | `not_recommended` |
| VLM image QA | `llava-7b-amd-win` | 13666.2ms | 0.8889 | `not_recommended` |
| VLM document extract | `llava-7b-amd-win` | 13666.2ms | 0.0667 | `not_recommended` |
| Embedding retrieval | `bge-base-en-v1.5-igpu-amd-win` | 2453.0ms | 0.9866 | `sync_default` |
| RAG search-only fallback | `bge-base-en-v1.5-igpu-amd-win` | 2453.0ms | 0.9866 | `sync_bounded` |
| Reranker candidates | `bge-reranker-base-igpu-amd-win` | 4527.7ms | 1.0000 | `sync_default` |
| OCR pages | `rapidocr-amd-directml` | 518.1ms | 0.9296 | `sync_default` |

ASR `sensevoice-small-amd-win` is included in the AMD Windows contract baseline: p95 latency 437.0ms, quality score 0.9231, verdict `sync_default`.

## Decision

Embedding, reranker, and OCR on the iGPU/DirectML path meet sync-default or bounded-sync product behavior. LLM and VLM rows completed the contract output, but their quality gate did not pass; they should remain retest targets after prompt, model, or dataset adjustments.

Final contract report: [nas-contract-report.md](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/nas-contract-report.md). Full machine-readable matrix: [parameter-matrix.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/parameter-matrix.json).
