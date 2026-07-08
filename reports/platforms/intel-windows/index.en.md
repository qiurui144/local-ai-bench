# Intel Windows

**Last updated:** 2026-07-08
**Chinese version:** [index.zh.md](index.zh.md)
**Legacy source:** [../../intel-windows.en.md](../../intel-windows.en.md)

## Scope

Intel Windows is split into CPU, Intel Arc iGPU, and Intel AI Boost NPU paths. CPU Ollama is the current calibrated LLM route. OpenVINO iGPU is validated for OCR/embedding/reranker and experimental LLM. NPU is validated for static-shape OCR and Whisper encoder, but not for dynamic embedding/reranker.

## Execution Path Summary

| Path | Runtime | Best-fit workloads | Status |
|---|---|---|---|
| [CPU](cpu.en.md) | Ollama CPU, ONNX Runtime CPU | LLM, embedding, reranker | PASS with translation caveats for LLM |
| [iGPU](igpu.en.md) | OpenVINO / optimum-intel | OCR, embedding, reranker, experimental LLM | PASS for non-LLM; LLM serving pending |
| [NPU](npu.en.md) | OpenVINO NPU / VPUX | Static OCR, Whisper encoder | PASS for static models; dynamic models fail |

## Selection Notes

| Role | Current choice | Decision |
|---|---|---|
| LLM route | CPU Ollama | `qwen2.5-3b` for interactive use, `qwen2.5-7b` for stronger GA with higher latency. |
| OCR route | OpenVINO iGPU, or NPU static OCR when pipeline supports it | DirectML OCR is not usable on this platform. |
| Embedding/reranker route | CPU or OpenVINO iGPU | iGPU warm path is faster where a serving wrapper exists. |
| NPU route | Static OCR and Whisper encoder | Keep separate because dynamic-shape failures are path-specific. |

## Evidence

| Detail | Report |
|---|---|
| CPU path | [cpu.en.md](cpu.en.md) |
| iGPU path | [igpu.en.md](igpu.en.md) |
| NPU path | [npu.en.md](npu.en.md) |
| Legacy full report | [../../intel-windows.en.md](../../intel-windows.en.md) |
| Legacy CPU detail | [../../intel-windows-cpu.en.md](../../intel-windows-cpu.en.md) |
| Legacy iGPU/NPU detail | [../../intel-windows-igpu.en.md](../../intel-windows-igpu.en.md) |
