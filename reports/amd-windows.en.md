# AMD Windows Platform — Model Selection & Benchmark Report

**Platform:** amd-win-x86 | Ryzen 8845H + Radeon 780M iGPU + AMD XDNA NPU, Windows 11  
**Last calibrated:** 2026-06-20. This file is updated in place.

---

## Hardware Overview

| Compute Unit | Specs | Role |
|---|---|---|
| **CPU** | Ryzen 8845H (4× Zen4 P-core + 4× Zen4c E-core) | ONNX Runtime CPU — OCR baseline, Reranker |
| **iGPU** | Radeon 780M (RDNA3, 12 CU, 17.9 GiB shared VRAM) | Ollama Vulkan — LLM + Embedding; ONNX DirectML — OCR |
| **NPU** | AMD XDNA (AI 300 Series, 16 TOPS) | ONNX VitisAI — OCR (batch); ASR |

---

## Execution Mode Comparison

All measured values are p50 latency or TPS from E2E calibration runs.

| Workload | CPU path | iGPU path (Vulkan / DirectML) | NPU path (VitisAI) |
|---|---|---|---|
| **LLM 7B** | ~3–5 TPS (est.) | **13.33 TPS** ✓ | — |
| **LLM 3B** | ~8–12 TPS (est.) | **28.99 TPS** ✓ | — |
| **LLM 0.6B** | — | **91.09 TPS** ✓ | — |
| **LLM NPU (1B, Lemonade)** | — | — | **~80–100 TPS** PENDING-VERIFY |
| **LLM NPU (1.5B, Lemonade)** | — | — | **~60–80 TPS** PENDING-VERIFY |
| **LLM NPU (3.8B, Lemonade)** | — | — | **~30–50 TPS** PENDING-VERIFY |
| **Embedding 0.6B** | — | 875 ms p50 ✓ | — |
| **OCR text (p50)** | 1593 ms | **469 ms** ✓ fastest | 2031 ms |
| **OCR structured (p50)** | 859 ms | **477 ms** ✓ | 1868 ms |
| **ASR (RTF)** | — | — | **0.073** ✓ |
| **Reranker base (p50)** | **78 ms** ✓ | — | — |
| **Reranker v2-m3 (p50)** | 289 ms | — | — |

CPU-only LLM is not independently benchmarked; Ollama defaults to Vulkan iGPU.
OCR quality (CER 7.04%) is identical across all three paths.
**NPU LLM via Lemonade**: AMD XDNA NPU can run LLM inference using `lemonade-server` — all values PENDING-VERIFY.

**→ Mode details:**
- [iGPU (Vulkan + DirectML) — LLM, Embedding, OCR fastest path](./amd-windows-igpu.en.md)
- [NPU (VitisAI + DirectML) — OCR batch, ASR](./amd-windows-npu.en.md)
- [CPU ONNX — OCR baseline, Reranker](./amd-windows-cpu.en.md)

---

## Selection Summary

| Role | Selected Model | Execution mode | Rationale |
|---|---|---|---|
| LLM primary | `qwen2.5-7b-amd-win` | iGPU (Vulkan) | Best overall quality; translation/scenarios FAIL is model capability ceiling |
| LLM lightweight | `llama3.2-3b-amd-win` | iGPU (Vulkan) | 32k context validated, 32-concurrency verified |
| Embedding (primary) | `qwen3-embedding-0.6b-amd` | iGPU (Vulkan) | Best retrieval quality, lower latency |
| Embedding (multilingual) | `bge-m3-amd` | iGPU (Vulkan) | Drop-in multilingual alternative |
| Reranker (default) | `bge-reranker-base-amd-win` | CPU ONNX | p50 78 ms, sufficient quality for most use cases |
| Reranker (quality) | `bge-reranker-v2-m3-amd-win` | CPU ONNX | Equal nDCG/MRR but 3.7× latency — use only when ranking quality is critical |
| OCR (primary) | `rapidocr-amd-directml` | iGPU DirectML | Fastest path: p50 468 ms |
| OCR (batch / low-power) | `rapidocr-amd-npu` | NPU VitisAI | p50 2031 ms — saves iGPU for concurrent LLM |
| ASR | `sensevoice-small-amd-win` | NPU DirectML | PASS: CER 7.69%, RTF 0.073 |
| VLM | *(not recommended)* | — | `llava-7b-amd-win` runs but accuracy FAIL; no qualified VLM on this platform |

---

## Full Model Results

| Model | Execution | Role | Status | Key Metrics |
|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | iGPU Vulkan | llm_primary | **FAIL** | TPS 13.33; TTFT p50/p95 953/6241 ms; PP/TG 116/16 t/s; general_ability PASS (gsm8k=0.880/mmlu=0.600/hellaswag=0.790, 3-seed); translation FAIL (zh→en term=79%<80%; en→zh chrF=36.4<40.0) |
| `qwen2.5-14b-amd-win` | iGPU Vulkan | llm_parameter_uplift | **MEASURED** | TPS 7.67; TTFT p50/p95 8274/14792 ms; max-ctx 16k |
| `llama3.2-3b-amd-win` | iGPU Vulkan | llm_baseline | **FAIL** | TPS 28.99; TTFT p50/p95 890/5207 ms; PP/TG 124/39 t/s; max ctx 32k; general_ability FAIL (gsm8k=0.710/PASS, mmlu=0.390/FAIL, hellaswag=0.320/FAIL, 3-seed); translation FAIL (zh→en term=55%<80%; en→zh chrF=27.6<40.0) |
| `qwen3-0.6b-amd` | iGPU Vulkan | llm_nano | **FAIL** | TPS 91.09; TTFT p50 1781 ms; general_ability FAIL (gsm8k=0.390/PASS, mmlu=0.000/FAIL, hellaswag=0.000/FAIL — 0.6B MCQ capability gap, confirmed post parser-fix 2026-06-20) |
| `llava-7b-amd-win` | iGPU Vulkan | vlm_baseline | **FAIL** | TPS 16.84; TTFT p50 890 ms; accuracy FAIL |
| `qwen3-embedding-0.6b-amd` | iGPU Vulkan | embedding_primary | **PASS** | hit@1 1.000; nDCG 1.000; p50 875 ms |
| `bge-m3-amd` | iGPU Vulkan | embedding_bge | **PASS** | hit@1 1.000; nDCG 1.000; p50 914 ms |
| `rapidocr-amd-directml` | iGPU DirectML | ocr_gpu | **PASS** | CER 7.04%; p50 468.5 ms; structured field acc 92.86%; structured p50 476.5 ms |
| `rapidocr-amd-npu` | NPU VitisAI | ocr_npu | **PASS** | CER 7.04%; p50 2031 ms; structured field acc 92.86%; structured p50 1867.5 ms |
| `rapidocr-cpu` | CPU ONNX | ocr_cpu_baseline | **PASS** | CER 7.04%; p50 1592.5 ms; structured field acc 92.86%; structured p50 859.0 ms |
| `paddleocr-cpu` | CPU ONNX | ocr_cpu_paddle | **PASS** | CER 7.04%; p50 1829.5 ms |
| `bge-reranker-base-amd-win` | CPU ONNX | reranker_default | **PASS** | nDCG 1.000; MRR 1.000; p50 78 ms |
| `bge-reranker-v2-m3-amd-win` | CPU ONNX | reranker_stronger | **PASS** | nDCG 1.000; MRR 1.000; p50 289 ms |
| `sensevoice-small-amd-win` | NPU DirectML | asr | **PASS** | CER 7.69%; RTF 0.073 |

**Status legend:** PASS = all thresholds met. FAIL = one or more quality/perf thresholds missed.
MEASURED = latency/throughput collected; quality dims not fully qualified.

---

## Known Limitations

- **LLM translation FAIL** — All LLM models (3B/7B/0.6B tested) fail translation quality gate (chrF or term_match_rate below threshold). Model capability ceiling, not a deployment blocker.
- **qwen3-0.6b MCQ FAIL** — mmlu=0.000, hellaswag=0.000. Root cause is 0.6B model incapacity to reliably output MCQ letter answers (A/B/C/D), confirmed after `<think>` parser fix applied. gsm8k=0.390 (open-ended math) passes.
- **LLM conditioned FAIL** — Long-context conditioning fails across all tested models.
- **LLM conversation_drift FAIL** — Multi-turn drift detection fails.
- **LLM scenarios FAIL** — Domain scenario tests fail.
- **No qualified VLM** — `llava-7b-amd-win` accuracy FAIL; no VLM workloads recommended until a better model is validated.
- **NPU LLM PENDING-VERIFY** — AMD XDNA NPU LLM via Lemonade Server (`lemonade-server serve --device npu`) supports Llama-3.2-1B, Phi-3.5-mini, Qwen2.5-1.5B (W4A8). Thresholds not yet calibrated; all TPS estimates are pre-measurement.

---

## Calibration History

| Date | Event |
|---|---|
| 2026-06-19 | Initial full calibration: all 14 models measured across CPU/iGPU/NPU paths; thresholds set from E2E runs |
| 2026-06-20 | Added quality dims: qwen2.5-7b general_ability PASS (3-seed, gsm8k=0.880/mmlu=0.600/hellaswag=0.790); all 3 LLM models translation FAIL formally documented; llama3.2-3b general_ability FAIL (3-seed); qwen3-0.6b general_ability FAIL — re-run with parser fix confirms mmlu/hellaswag=0.000 is real 0.6B MCQ capability gap |

---

## 中文摘要

**平台：** amd-win-x86 | Ryzen 8845H + Radeon 780M iGPU + AMD XDNA NPU，Windows 11  
**最后校准：** 2026-06-20。本文件原地更新。

### 硬件概述

| 计算单元 | 规格 | 角色 |
|---|---|---|
| CPU | Ryzen 8845H | ONNX CPU — OCR 基线、Reranker |
| iGPU | Radeon 780M（RDNA3，17.9 GiB 共享显存） | Ollama Vulkan — LLM/Embedding；ONNX DirectML — OCR |
| NPU | AMD XDNA（16 TOPS） | ONNX VitisAI — OCR 批处理；ASR |

### 执行模式对比

| 任务 | CPU 路径 | iGPU 路径（Vulkan/DirectML） | NPU 路径（VitisAI） |
|---|---|---|---|
| LLM 7B | ~3–5 TPS（估算） | **13.33 TPS** ✓ | — |
| LLM 3B | ~8–12 TPS（估算） | **28.99 TPS** ✓ | — |
| **LLM NPU（Lemonade）** | — | — | **~80–100 TPS**（PENDING-VERIFY） |
| OCR 文字 p50 | 1593 ms | **469 ms** ✓ 最快 | 2031 ms |
| ASR RTF | — | — | **0.073** ✓ |
| Reranker base p50 | **78 ms** ✓ | — | — |

AMD XDNA NPU 现通过 Lemonade Server 支持 LLM 推理（W4A8 量化，≤3.8B 模型）。PENDING-VERIFY。

**→ 详细模式文档：**
- [iGPU（Vulkan + DirectML）— LLM、Embedding、OCR 最快路径](./amd-windows-igpu.en.md)
- [NPU（VitisAI + DirectML）— OCR 批处理、ASR](./amd-windows-npu.en.md)
- [CPU ONNX — OCR 基线、Reranker](./amd-windows-cpu.en.md)

### 选型摘要

| 角色 | 推荐模型 | 执行模式 |
|---|---|---|
| LLM 主力 | `qwen2.5-7b-amd-win` | iGPU Vulkan |
| LLM 轻量 | `llama3.2-3b-amd-win` | iGPU Vulkan |
| Embedding（首选） | `qwen3-embedding-0.6b-amd` | iGPU Vulkan |
| Embedding（多语言） | `bge-m3-amd` | iGPU Vulkan |
| Reranker（默认） | `bge-reranker-base-amd-win` | CPU ONNX |
| OCR（首选） | `rapidocr-amd-directml` | iGPU DirectML |
| OCR（批处理） | `rapidocr-amd-npu` | NPU VitisAI |
| ASR | `sensevoice-small-amd-win` | NPU DirectML |
