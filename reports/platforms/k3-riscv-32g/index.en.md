# K3 RISC-V 32G

**Last updated:** 2026-07-14
**Chinese version:** [index.zh.md](index.zh.md)
**Legacy source:** [../../k3-riscv-32g.en.md](../../k3-riscv-32g.en.md)

## Scope

This platform report covers SpacemiT K3 X100 32GB on Bianbu Linux. It separates GGUF/llama.cpp model serving from ORT/SMT media paths and product workflow risk. Gaps in earlier summaries were filled from `reports/runs/k3-riscv-32g/20260704` and later `output/reports/k3-riscv-32g` evidence referenced in the legacy full report.

## Execution Path Summary

| Path | Runtime | Best-fit workloads | Status |
|---|---|---|---|
| [llama.cpp / GGUF](llama.en.md) | SpacemiT private llama.cpp, source-built llama.cpp, mtmd | LLM, GGUF+mmproj VLM, embedding, reranker | PASS with model-specific limits |
| [ORT / SMT](ort.en.md) | SpacemiT ORT EP, SMT media backend | Official ONNX vision, VLM tar, ASR tar, PP-OCRv5 | PASS after TCM-state control |
| [Workflow Risk](workflow-risk.en.md) | Product workflow layer | RAG, document OCR/VLM, ASR, long-context aviation manuals, stress controls | Requires admission/queue controls |
| [Contract Supplement](contract.en.md) | NAS contract matrix | Qwen3-30B-A3B sync, bounded sync, and async boundary rows | Complete for current Qwen3-30B contract slice |

## Selection Notes

| Role | Current choice | Decision |
|---|---|---|
| LLM primary | `Qwen3-30B-A3B-Q4_0` | Best qualified K3 LLM, but long-context sync use is high risk. |
| Large LLM candidate | `Qwen3.6-35B-A3B-UD-Q4_K_XL` | LLM path passes current probes; image input unsupported. |
| Single-service LLM+VLM candidate | external `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | Quality pass but async-only due latency. |
| Sync VLM | `Qwen3.5-2B.tar.gz` | Best practical document VLM: 30/30 cases, p95 12.239s. |
| Quality VLM | `Qwen3VL-4B + mmproj` | 30/30 quality control; too slow for default sync. |
| OCR | `PP-OCRv5_mobile_det+rec.onnx` | Line OCR pass; use OCR before VLM. |
| ASR | `qwen3-asr-0.6B.tar.gz` | Default over 1.7B due lower RTF/RSS at same normalized CER. |
| Embedding | `Bge-Small-Zh-V1.5-Q4_K_M` | Default: p95 5.85ms; `Bge-Small-En` fails invalid vectors. |
| Reranker | `Bge-Reranker-V2-M3-Q4_0` | Default; keep interactive top-k <=20. |

## Official Alignment

| Area | Alignment | Evidence |
|---|---|---|
| Official ONNX vision | 132/132 aligned | [ORT / SMT](ort.en.md) |
| Official LLM ModelZoo rows | 8/8 aligned with TCM enabled | [llama.cpp / GGUF](llama.en.md) |
| VLM VisionEncoder rows | 10/10 aligned probe | [ORT / SMT](ort.en.md) |
| ASR | partial / retest required for official-style aggregation | [ORT / SMT](ort.en.md) |
| OCR / embedding / reranker | local-only; no official ModelZoo rows in cited page | [Evidence Map](../../evidence/k3-riscv-32g.evidence.en.md) |

## Evidence

| Detail | Report |
|---|---|
| llama.cpp / GGUF path | [llama.en.md](llama.en.md) |
| ORT / SMT path | [ort.en.md](ort.en.md) |
| Workflow risk | [workflow-risk.en.md](workflow-risk.en.md) |
| NAS contract supplement | [contract.en.md](contract.en.md) |
| Raw evidence map | [../../evidence/k3-riscv-32g.evidence.en.md](../../evidence/k3-riscv-32g.evidence.en.md) |
| Legacy full report | [../../k3-riscv-32g.en.md](../../k3-riscv-32g.en.md) |
