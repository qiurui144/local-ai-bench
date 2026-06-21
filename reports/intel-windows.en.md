# Intel Windows Platform — Model Selection & Benchmark Report

**Platform:** intel-win-x86 | Intel Core Ultra laptop, Windows 11  
**Last calibrated:** 2026-06-21. This file is updated in place.

---

## Hardware Overview

| Compute Unit | Specs | Role |
|---|---|---|
| **CPU** | Intel Core Ultra (P-core + E-core) | Ollama CPU — LLM + Embedding; ONNX CPU — Reranker |
| **iGPU** | Intel Arc integrated graphics | ONNX OpenVINO — OCR (PASS); ONNX DirectML — OCR (FAIL) |
| **NPU** | Intel NPU (AI Boost) | Not yet tested |

---

## Execution Mode Comparison

| Workload | CPU path | iGPU / OpenVINO | NPU |
|---|---|---|---|
| **LLM 7B** | 8.25 TPS; TTFT 4820 ms | not configured | not tested |
| **LLM 3B** | 19.47 TPS; TTFT 781 ms | not configured | not tested |
| **LLM 1B** | 25.26 TPS; TTFT 875 ms | not configured | not tested |
| **Embedding 0.6B** | 617.5 ms p50 | not configured | — |
| **OCR text (p50)** | 1593 ms (reference) | 797 ms OpenVINO ✓; 946 ms DirectML ✗ | not tested |
| **OCR structured (p50)** | 859 ms (reference) | 868 ms OpenVINO ✓; 985 ms DirectML ✗ | not tested |
| **ASR (RTF)** | — | 0.341 (DirectML) ✓ | — |
| **Reranker base (p50)** | 148.5 ms ✓ | — | — |
| **Reranker v2-m3 (p50)** | 546.5 ms ✓ | — | — |

Intel DirectML OCR is **not usable** (CER 202%). Use OpenVINO instead.  
Intel iGPU LLM acceleration is not yet configured; all LLM runs use CPU-only Ollama.

**→ Mode details:**
- [CPU mode — LLM, Embedding, Reranker](./intel-windows-cpu.en.md)
- [iGPU / OpenVINO / DirectML — OCR, ASR](./intel-windows-igpu.en.md)

---

## Selection Summary

| Role | Selected Model | Execution mode | Rationale |
|---|---|---|---|
| LLM quality | `qwen2.5-7b-intel-win` | CPU | Best GA quality on platform; FAIL translation (term/chrF thresholds); high TTFT for interactive use |
| LLM daily use | `qwen2.5-3b-intel-win` | CPU | Lightweight, 8-concurrency verified; TTFT 781 ms suitable for interactive |
| LLM lightweight | `llama3.2-1b-intel-win` | CPU | 32-concurrency, 32k context verified |
| Embedding | `qwen3-embedding-0.6b-intel-win` | CPU | PASS: hit@1 1.000, p50 617.5 ms |
| Reranker (default) | `bge-reranker-base-intel-win` | CPU ONNX | p50 148.5 ms, sufficient for most use cases |
| Reranker (quality) | `bge-reranker-v2-m3-intel-win` | CPU ONNX | Equal nDCG/MRR but p50 546.5 ms — use when ranking quality is critical |
| OCR | `rapidocr-intel-openvino` | iGPU OpenVINO | PASS: p50 797 ms; DirectML unavailable |
| ASR | `sensevoice-small-intel-win` | DirectML | PASS: CER 7.69%, RTF 0.341 |
| VLM | *(not recommended)* | — | `llava-7b-intel-win` runs but accuracy FAIL |

---

## Full Model Results

| Model | Execution | Role | Status | Key Metrics |
|---|---|---|---|---|
| `qwen2.5-7b-intel-win` | CPU (Ollama) | llm_quality | **FAIL** (translation) | TPS 8.25; TTFT p50/p95 4820/8441 ms; PP/TG 112/9 t/s; GA PASS (GSM8K 0.833/MMLU 0.719/HellaSwag 0.767); translation FAIL (zh→en term 79%<80%; en→zh chrF 36.9<40) |
| `qwen2.5-3b-intel-win` | CPU (Ollama) | llm_baseline | **FAIL** (translation) | TPS 19.47; TTFT p50/p95 781/3495 ms; GA PASS (GSM8K 0.74/MMLU 0.53/HellaSwag 0.76); translation FAIL (en→zh chrF 33/34.8 < 40) |
| `llama3.2-1b-intel-win` | CPU (Ollama) | llm_nano | **FAIL** | TPS 25.26; TTFT p50/p95 875/3308 ms; PP/TG 130/35 t/s; max ctx 32k; GA/translation SKIPPED (1B model not GA-tested by design) |
| `llava-7b-intel-win` | CPU (Ollama) | vlm_baseline | **FAIL** | TPS 10.02; TTFT p50 703 ms; accuracy FAIL |
| `qwen3-embedding-0.6b-intel-win` | CPU (Ollama) | embedding | **PASS** | hit@1 1.000; nDCG 1.000; p50 617.5 ms |
| `bge-reranker-base-intel-win` | CPU ONNX | reranker_default | **PASS** | nDCG 1.000; MRR 1.000; p50 148.5 ms |
| `bge-reranker-v2-m3-intel-win` | CPU ONNX | reranker_stronger | **PASS** | nDCG 1.000; MRR 1.000; p50 546.5 ms |
| `rapidocr-intel-openvino` | iGPU OpenVINO | ocr_openvino | **PASS** | CER 7.04%; p50 797 ms; structured field acc 92.86%; structured p50 867.5 ms |
| `rapidocr-intel-directml` | iGPU DirectML | ocr_directml | **FAIL** | CER 202.35% — not usable |
| `sensevoice-small-intel-win` | DirectML | asr | **PASS** | CER 7.69%; RTF 0.341 |

**Status legend:** PASS = all thresholds met. FAIL = one or more thresholds missed.
MEASURED = latency/throughput collected; quality dims not fully qualified.

---

## Known Limitations

- **qwen2.5-3b translation FAIL** — en→zh chrF 33.0/34.8 < 40.0 threshold; term-match 64/74% < 80%. 3B CPU model insufficient for Chinese translation; use 7B or cloud backend for translation tasks.
- **conditioned BLOCKED** — Requires running from controller with HF cache; not yet measured.
- **Intel DirectML OCR not usable** — `rapidocr-intel-directml` CER 202.35%; FP16 precision issue on Intel iGPU with DirectML. Use OpenVINO path.
- **No qualified VLM** — `llava-7b-intel-win` accuracy FAIL.
- **LLM TTFT high (7B)** — `qwen2.5-7b-intel-win` p50 TTFT 4820 ms is driven by CPU-only prefill; prefer `qwen2.5-3b-intel-win` for interactive use.
- **iGPU LLM not tested** — Intel iGPU LLM acceleration (via OpenVINO or IPEX) is not yet configured.
- **general_ability unblocked 2026-06-21** — Resolved by running inference from controller over HTTP with local HF cache. qwen2.5-3b GA PASS (GSM8K 0.74/MMLU 0.53/HellaSwag 0.76); qwen2.5-7b GA PASS (GSM8K 0.833/MMLU 0.719/HellaSwag 0.767).

---

## Calibration History

| Date | Event |
|---|---|
| 2026-06-19 | Initial full calibration: all 10 models measured; CPU LLM, OpenVINO OCR, DirectML ASR calibrated; general_ability/conditioned BLOCKED pending datasets install |
| 2026-06-21 | general_ability unblocked (HTTP inference from controller + local HF cache); qwen2.5-3b: GSM8K 0.74/MMLU 0.53/HellaSwag 0.76 PASS; translation FAIL (en→zh chrF 33-34.8 < 40); qwen2.5-7b: GA PASS (GSM8K 0.833/MMLU 0.719/HellaSwag 0.767); translation FAIL (zh→en term 79%<80%; en→zh chrF 36.9<40) |

---

## 中文摘要

**平台：** intel-win-x86 | Intel Core Ultra 笔电，Windows 11  
**最后校准：** 2026-06-19。本文件原地更新。

### 硬件概述

| 计算单元 | 规格 | 角色 |
|---|---|---|
| CPU | Intel Core Ultra | Ollama CPU — LLM/Embedding；ONNX CPU — Reranker |
| iGPU | Intel Arc 集成显卡 | ONNX OpenVINO — OCR（PASS）；ONNX DirectML — OCR（FAIL） |
| NPU | Intel AI Boost | 未测试 |

### 执行模式对比

| 任务 | CPU 路径 | iGPU/OpenVINO | NPU |
|---|---|---|---|
| LLM 7B | 8.25 TPS；TTFT 4820 ms | 未配置 | 未测试 |
| LLM 3B | 19.47 TPS；TTFT 781 ms | 未配置 | — |
| OCR 文字 p50 | 1593 ms（参考） | 797 ms OpenVINO ✓；946 ms DirectML ✗ | — |
| ASR RTF | — | 0.341（DirectML）✓ | — |
| Reranker base p50 | 148.5 ms ✓ | — | — |

**→ 详细模式文档：**
- [CPU 模式 — LLM、Embedding、Reranker](./intel-windows-cpu.en.md)
- [iGPU/OpenVINO/DirectML — OCR、ASR](./intel-windows-igpu.en.md)

### 选型摘要

| 角色 | 推荐模型 | 执行模式 |
|---|---|---|
| LLM 质量首选 | `qwen2.5-7b-intel-win` | CPU（延迟偏高） |
| LLM 日常首选 | `qwen2.5-3b-intel-win` | CPU（TTFT 781 ms） |
| LLM 轻量 | `llama3.2-1b-intel-win` | CPU |
| Embedding | `qwen3-embedding-0.6b-intel-win` | CPU |
| Reranker（默认） | `bge-reranker-base-intel-win` | CPU ONNX |
| OCR | `rapidocr-intel-openvino` | iGPU OpenVINO |
| ASR | `sensevoice-small-intel-win` | DirectML |

### 已知局限

- **Intel DirectML OCR 不可用** — CER 202.35%，改用 OpenVINO 路径。
- **general_ability 已解锁（2026-06-21）** — qwen2.5-3b GA PASS（GSM8K 0.74/MMLU 0.53/HellaSwag 0.76）；qwen2.5-7b GA PASS（GSM8K 0.833/MMLU 0.719/HellaSwag 0.767）；两者翻译维度均 FAIL。
- **qwen2.5-3b 翻译 FAIL** — en→zh chrF 33.0/34.8 < 40.0，3B 模型中文翻译不足；翻译任务建议用 7B 或云端。
- **iGPU LLM 未测试** — Intel iGPU LLM 加速尚未配置。
