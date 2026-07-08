# Model Selection

**Last updated:** 2026-07-08
**Chinese version:** [model-selection.zh.md](model-selection.zh.md)
**Purpose:** provide the first usable decision layer for model and platform selection.

## Scope

This report consolidates legacy platform reports and run-log evidence. Rows marked as local-only are not official vendor baseline claims. Missing legacy coverage was filled from `reports/runs` and `output/reports` summaries where available.

## Decision Summary

| Role | Recommended model/path | Platform | Status | Why | Main risk | Evidence |
|---|---|---|---|---|---|---|
| K3 LLM primary | `Qwen3-30B-A3B-Q4_0` | K3 32G llama.cpp | PASS with limits | Best K3 full-GA coverage; short bounded requests pass | Long-context sync is high risk: 1K realistic run reached 223.834s and 30.488GiB RSS | [K3 llama](../platforms/k3-riscv-32g/llama.en.md) |
| K3 large LLM candidate | `Qwen3.6-35B-A3B-UD-Q4_K_XL` | K3 32G llama.cpp | PASS LLM, not VLM | Current system runtime serves text; 1K/3K needles pass | `vision_backend=none`; not a single-model LLM+VLM solution | [K3 llama](../platforms/k3-riscv-32g/llama.en.md) |
| K3 single-service LLM+VLM candidate | `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | K3 32G mtmd | PASS, async-only | 29/30 document cases, field acc 0.9942, JSON 1.0 | Avg/p95 68.822/78.774s; long text is async-only | [K3 workflow risk](../platforms/k3-riscv-32g/workflow-risk.en.md) |
| K3 sync VLM default | `Qwen3.5-2B.tar.gz` | K3 SMT media backend | PASS | 30/30 document cases; p95 12.239s | Requires clean TCM state; not a replacement for dedicated OCR | [K3 ORT/SMT](../platforms/k3-riscv-32g/ort.en.md) |
| K3 VLM quality control | `Qwen3VL-4B + mmproj` | K3 mtmd | PASS, slow | 30/30 document cases; JSON 1.0 | Realistic p95 can reach 78.064s | [K3 llama](../platforms/k3-riscv-32g/llama.en.md) |
| K3 OCR default | `PP-OCRv5_mobile_det+rec.onnx` | K3 ORT path | PASS line OCR | 72-line broad run CER 0.0039 | Full document layout assembly remains product work | [K3 ORT/SMT](../platforms/k3-riscv-32g/ort.en.md) |
| K3 ASR default | `qwen3-asr-0.6B.tar.gz` | K3 SMT audio | PASS | Same normalized CER as 1.7B with lower RTF/RSS | Needs Chinese normalization for fair scoring | [K3 ORT/SMT](../platforms/k3-riscv-32g/ort.en.md) |
| K3 embedding default | `Bge-Small-Zh-V1.5-Q4_K_M` | K3 GGUF embedding | PASS | p95 5.85ms; overall Hit@1 0.9722 | `Bge-Small-En` returns invalid vectors on this runtime | [K3 llama](../platforms/k3-riscv-32g/llama.en.md) |
| K3 reranker default | `Bge-Reranker-V2-M3-Q4_0` | K3 GGUF reranker | PASS | Hit@1 1.0 through top50 | Interactive top-k should stay <=20 | [K3 llama](../platforms/k3-riscv-32g/llama.en.md) |
| AMD Windows LLM | `qwen2.5-7b-amd-win` or `llama3.2-3b-amd-win` | Radeon 780M iGPU | Measured, quality caveats | 7B has 13.33 TPS; 3B has 28.99 TPS | Current harness quality gates fail; use for performance/route coverage | [AMD iGPU](../platforms/amd-windows/igpu.en.md) |
| AMD Windows OCR | `rapidocr-amd-directml` | Radeon 780M DirectML | PASS | p50 468.5ms, faster than CPU and NPU paths | NPU OCR is slower and better suited to batch/thermal separation | [AMD iGPU](../platforms/amd-windows/igpu.en.md) |
| AMD Windows reranker | `bge-reranker-base-amd-win` | CPU ONNX | PASS | p50 78ms, nDCG/MRR 1.0 | v2-m3 is 3.7x slower | [AMD CPU](../platforms/amd-windows/cpu.en.md) |
| Intel Windows CPU LLM | `qwen2.5-3b-intel-win` for interactive, `qwen2.5-7b-intel-win` for quality | Intel CPU | GA PASS, translation caveats | 3B TTFT p50 781ms; 7B GA scores stronger | Translation thresholds fail | [Intel CPU](../platforms/intel-windows/cpu.en.md) |
| Intel Windows iGPU/NPU OCR | OpenVINO OCR, NPU static PP-OCRv4 where pipeline supports it | Intel Arc / AI Boost NPU | PASS | OpenVINO p50 797ms; NPU det+rec+cls about 47ms | NPU needs static shapes and pipeline integration | [Intel NPU](../platforms/intel-windows/npu.en.md) |
| RK1828 LLM/VLM | `qwen3-vl-2b-rk1820` | RK1828 PCIe NPU | PASS primary dims | TTFT p50/p95 143/244ms; TPS 108.5 | 768-token runtime context; conversation drift fails | [RK1828 NPU](../platforms/rk3588/rk1828-npu.en.md) |
| RK3588 embedding | `minicpm-embed-rk3588` | RK3588 RKNPU3 | PASS | hit@1/MRR/nDCG 1.0; p50 143ms | Embedding-only path in this deployment | [RK3588 RKNPU3](../platforms/rk3588/rk3588-rknpu.en.md) |

## K3 Recommended Stack

| Workflow | Recommended stack | Decision |
|---|---|---|
| Realtime RAG | BGE-Zh embedding -> BGE reranker top-k <=20 -> bounded Qwen3-30B answer | Usable with strict context and token caps |
| Document OCR | PP-OCRv5 first; route to VLM only for visual reasoning | Do not use VLM as OCR replacement |
| Sync VLM | Qwen3.5-2B SMT | Best latency/quality balance |
| Async/high-quality VLM | Qwen3VL-4B, qwen30ba3b-mm, or external Qwen3.5-35B+mmproj | Requires queue, TTL, cancellation, and visible job status |
| ASR | qwen3-ASR 0.6B SMT | Default over 1.7B |
| Aviation manuals / long documents | offline text/OCR -> embedding -> reranker -> cited LLM evidence window | Whole-manual direct ingestion is not a K3 sync path |

## Windows Recommended Stack

| Platform | CPU | GPU/iGPU | NPU |
|---|---|---|---|
| AMD Windows | Reranker default; OCR baseline | LLM/embedding via Ollama Vulkan; OCR via DirectML | OCR batch/thermal separation; LLM NPU remains pending |
| Intel Windows | LLM CPU default; reranker CPU | OpenVINO iGPU for OCR/embedding/reranker and experimental LLM | Static-shape OCR and Whisper encoder; embedding/reranker dynamic shapes fail |

## RK Recommended Stack

| Chip path | Recommended role | Decision |
|---|---|---|
| RK1828 PCIe NPU | LLM/VLM and ASR | `qwen3-vl-2b-rk1820` and `rk-asr-rk1820` are calibrated |
| RK3588 RKNPU3 | Embedding | `minicpm-embed-rk3588` is calibrated |
| RKNN3 cached models | Future LLM/VLM/OCR coverage | 46/46 artifacts cached, but service loading and calibration are pending |

## Evidence Map

| Evidence class | Path |
|---|---|
| K3 raw evidence map | [../evidence/k3-riscv-32g.evidence.en.md](../evidence/k3-riscv-32g.evidence.en.md) |
| K3 legacy full report | [../k3-riscv-32g.en.md](../k3-riscv-32g.en.md) |
| AMD legacy full report | [../amd-windows.en.md](../amd-windows.en.md) |
| Intel legacy full report | [../intel-windows.en.md](../intel-windows.en.md) |
| RK legacy full report | [../rk3588.en.md](../rk3588.en.md) |

## Status Vocabulary

| Status | Meaning |
|---|---|
| PASS | End-to-end or route-specific gates passed for the stated workload. |
| PASS with limits | Core route works, but product use needs explicit bounds. |
| Measured, quality caveats | Performance is valid, but one or more quality gates failed or were incomplete. |
| Local-only | No official vendor baseline exists for this row; use only as project data. |
| Pending verify | Artifact is cached or path is known, but device/service calibration is not complete. |
| Fail | The row is not usable for the stated workload. |
