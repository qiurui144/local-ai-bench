# Intel Windows iGPU Path

**Last updated:** 2026-07-08
**Chinese version:** [igpu.zh.md](igpu.zh.md)
**Legacy source:** [../../intel-windows-igpu.en.md](../../intel-windows-igpu.en.md)

## Scope

The Intel Arc iGPU path uses OpenVINO and optimum-intel. It is validated for OCR, embedding, and reranker. LLM inference is confirmed through optimum-intel but still needs a stable serving layer for full benchmark parity.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| LLM | `qwen2.5-7b-int4-ov` | 8.1 TPS, cold load 115s | CONFIRMED | Experimental until HTTP serving is stable |
| LLM | `qwen2.5-1.5b-int4-ov` | 10.6 TPS, cold load 54s | CONFIRMED | Experimental control |
| Embedding | `bge-base-en-v1.5-int8-ov` | warm latency about 25ms | PASS | Fast iGPU embedding where service wrapper exists |
| Reranker | `bge-reranker-base-int8-ov` | avg 36.4ms | PASS | Fast iGPU reranker where wrapper exists |
| OCR | `rapidocr-openvino` | p50 OCR 797ms | PASS | Recommended Intel OCR route |

## Decision

Use OpenVINO iGPU for OCR and accelerated embedding/reranker when the application can host the wrapper. For LLM, keep CPU as the calibrated report default until OpenVINO serving is wired into the benchmark harness.
