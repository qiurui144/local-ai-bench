# Intel Windows NPU Path

**Last updated:** 2026-07-08
**Chinese version:** [npu.zh.md](npu.zh.md)
**Source:** split from [../../intel-windows.en.md](../../intel-windows.en.md) and [../../intel-windows-igpu.en.md](../../intel-windows-igpu.en.md)

## Scope

The Intel AI Boost NPU path uses OpenVINO NPU/VPUX. It is validated for static-shape OCR models and the Whisper encoder. Dynamic-shape embedding, reranker, and SenseVoice ASR models fail on the current export/runtime shape constraints.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| OCR detection | PP-OCRv4 det static `[1,3,640,640]` | 33ms | PASS | NPU pipeline component |
| OCR recognition | PP-OCRv4 rec static `[1,3,48,320]` | 11ms | PASS | Requires H=48 static reshape |
| OCR classifier | PP-OCRv4 cls static `[1,3,48,192]` | 3ms | PASS | NPU pipeline component |
| ASR encoder | Whisper-base INT8 encoder static `[1,80,3000]` | 115ms | PASS | Encoder only; decoder remains CPU |
| Embedding | BGE INT8 OpenVINO | dynamic shape failure | FAIL | Use iGPU or CPU |
| Reranker | BGE reranker INT8 OpenVINO | dynamic shape failure | FAIL | Use iGPU or CPU |
| SenseVoice ASR | SenseVoice ONNX | dynamic self-attention mask issue | FAIL | Use DirectML path |

## Decision

Keep Intel NPU as a separate report path because its pass/fail profile is shape-specific. Use it for static OCR components and selected encoders, not for general LLM, embedding, or reranker serving.
