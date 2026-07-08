# AMD Windows

**Last updated:** 2026-07-08
**Chinese version:** [index.zh.md](index.zh.md)
**Legacy source:** [../../amd-windows.en.md](../../amd-windows.en.md)

## Scope

AMD Windows is split into CPU, Radeon 780M iGPU, and AMD XDNA NPU paths. The iGPU is the practical LLM/OCR acceleration path in the current evidence set. CPU remains the reranker path and OCR baseline. NPU is qualified for VitisAI OCR but not yet for LLM serving.

## Execution Path Summary

| Path | Runtime | Best-fit workloads | Status |
|---|---|---|---|
| [CPU](cpu.en.md) | ONNX Runtime CPU EP | Reranker, OCR baseline | PASS |
| [iGPU](igpu.en.md) | Ollama Vulkan, ONNX DirectML | LLM, embedding, fastest OCR | Measured; LLM quality caveats |
| [NPU](npu.en.md) | VitisAI, DirectML, Lemonade/FastFlowLM candidates | OCR batch, ASR, future LLM NPU | OCR/ASR PASS; LLM pending |

## Selection Notes

| Role | Current choice | Decision |
|---|---|---|
| LLM route | Radeon 780M iGPU via Ollama Vulkan | `qwen2.5-7b` and `llama3.2-3b` have valid performance data, but current quality gates are not clean passes. |
| OCR route | Radeon 780M DirectML | `rapidocr-amd-directml` is fastest: p50 468.5ms. |
| Reranker route | CPU ONNX | `bge-reranker-base-amd-win` p50 78ms; default for latency-sensitive paths. |
| NPU route | VitisAI OCR / pending LLM NPU | OCR works but is slower than DirectML; use when isolating iGPU or for batch/power experiments. |

## Evidence

| Detail | Report |
|---|---|
| CPU path | [cpu.en.md](cpu.en.md) |
| iGPU path | [igpu.en.md](igpu.en.md) |
| NPU path | [npu.en.md](npu.en.md) |
| Legacy full report | [../../amd-windows.en.md](../../amd-windows.en.md) |
| Legacy CPU detail | [../../amd-windows-cpu.en.md](../../amd-windows-cpu.en.md) |
| Legacy iGPU detail | [../../amd-windows-igpu.en.md](../../amd-windows-igpu.en.md) |
| Legacy NPU detail | [../../amd-windows-npu.en.md](../../amd-windows-npu.en.md) |
