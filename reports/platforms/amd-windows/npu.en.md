# AMD Windows NPU Path

**Last updated:** 2026-07-08
**Chinese version:** [npu.zh.md](npu.zh.md)
**Legacy source:** [../../amd-windows-npu.en.md](../../amd-windows-npu.en.md)

## Scope

The AMD XDNA NPU path covers VitisAI OCR and candidate Lemonade/FastFlowLM LLM routes. OCR is measured. NPU LLM remains pending on this Ryzen 8845H/XDNA generation.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| OCR | `rapidocr-amd-npu` via VitisAI | p50 2031ms, CER 7.04% | PASS | Batch or isolation path, not fastest |
| ASR | `sensevoice-small-amd-win` via DirectML | RTF 0.073, CER 7.69% | PASS | Best AMD ASR route in current evidence |
| LLM pure NPU | Lemonade / FastFlowLM candidates | not calibrated | PENDING-VERIFY | Do not use for model selection yet |
| LLM hybrid iGPU+NPU | Lemonade hybrid candidates | not calibrated | PENDING-VERIFY | Needs Ryzen AI software and end-to-end harness run |

## Decision

Keep AMD NPU in the report architecture because it is a distinct hardware path. For current model selection, use iGPU for LLM/OCR and CPU for reranker. Treat NPU LLM as a future validation workstream, not a current recommendation.
