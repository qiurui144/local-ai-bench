# AMD Windows CPU Path

**Last updated:** 2026-07-08
**Chinese version:** [cpu.zh.md](cpu.zh.md)
**Legacy source:** [../../amd-windows-cpu.en.md](../../amd-windows-cpu.en.md)

## Scope

The AMD CPU path is an ONNX Runtime CPU baseline for OCR and the production path for ONNX rerankers. LLM and embedding measurements on this platform use the Radeon 780M iGPU path instead.

## Workload Results

| Workload | Model/path | Metric | Status | Decision |
|---|---|---:|---|---|
| OCR baseline | `rapidocr-cpu` | p50 1592.5ms, CER 7.04% | PASS | Reference baseline only |
| OCR baseline | `paddleocr-cpu` | p50 1829.5ms, CER 7.04% | PASS | Slower baseline |
| Reranker | `bge-reranker-base-amd-win` | p50 78ms, nDCG/MRR 1.0 | PASS | Default reranker |
| Reranker | `bge-reranker-v2-m3-amd-win` | p50 289ms, nDCG/MRR 1.0 | PASS | Use only when rerank quality justifies latency |

## Decision

Use CPU for AMD reranking. Use DirectML iGPU for OCR unless thermal isolation or batch scheduling requires the NPU/CPU path. CPU-only LLM is not recommended; use the iGPU report for LLM data.
