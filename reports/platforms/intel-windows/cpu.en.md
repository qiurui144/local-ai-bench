# Intel Windows CPU Path

**Last updated:** 2026-07-08
**Chinese version:** [cpu.zh.md](cpu.zh.md)
**Legacy source:** [../../intel-windows-cpu.en.md](../../intel-windows-cpu.en.md)

## Scope

The Intel CPU path covers Ollama CPU LLM/embedding and ONNX CPU reranker. It is the current default LLM serving route for this platform.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| LLM | `qwen2.5-7b-intel-win` | 8.25 TPS, TTFT p50 4820ms, GA PASS | Translation FAIL | Stronger CPU quality, high latency |
| LLM | `qwen2.5-3b-intel-win` | 19.47 TPS, TTFT p50 781ms, GA PASS | Translation FAIL | Interactive CPU default |
| LLM | `llama3.2-1b-intel-win` | 25.26 TPS, TTFT p50 875ms | Quality incomplete | Lightweight control only |
| Embedding | `qwen3-embedding-0.6b-intel-win` | p50 617.5ms, hit@1 1.0 | PASS | CPU embedding default if iGPU wrapper is unavailable |
| Reranker | `bge-reranker-base-intel-win` | p50 148.5ms, nDCG/MRR 1.0 | PASS | CPU reranker default |
| Reranker | `bge-reranker-v2-m3-intel-win` | p50 546.5ms, nDCG/MRR 1.0 | PASS | Slow reranker alternate |

## Decision

Use `qwen2.5-3b-intel-win` for interactive CPU serving and `qwen2.5-7b-intel-win` only when stronger GA outweighs latency. Translation failures remain model-selection risks.
