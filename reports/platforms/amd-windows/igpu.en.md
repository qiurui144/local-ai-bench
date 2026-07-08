# AMD Windows iGPU Path

**Last updated:** 2026-07-08
**Chinese version:** [igpu.zh.md](igpu.zh.md)
**Legacy source:** [../../amd-windows-igpu.en.md](../../amd-windows-igpu.en.md)

## Scope

The Radeon 780M iGPU path covers Ollama Vulkan for LLM/embedding and ONNX DirectML for OCR. It is the practical acceleration path for AMD Windows in the current benchmark set.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| LLM | `qwen2.5-7b-amd-win` | 13.33 TPS, TTFT p50 953ms | Measured, quality caveats | Use only with task-specific validation |
| LLM | `llama3.2-3b-amd-win` | 28.99 TPS, TTFT p50 890ms | Measured, quality caveats | Lightweight/concurrency control |
| LLM | `qwen2.5-14b-amd-win` | 8.6 TPS | Measured | Only when larger parameter count is required |
| Embedding | `qwen3-embedding-0.6b-amd` | p50 875ms, hit@1 1.0 | PASS | Default AMD embedding route |
| Embedding | `bge-m3-amd` | p50 914ms, hit@1 1.0 | PASS | Multilingual alternate |
| OCR | `rapidocr-amd-directml` | p50 468.5ms, CER 7.04% | PASS | Fastest AMD OCR route |

## Decision

Use the iGPU path for AMD LLM performance coverage and for the fastest OCR route. Current LLM rows are performance-valid but not clean quality-pass rows, so production selection must be tied to the actual task and prompt class.
