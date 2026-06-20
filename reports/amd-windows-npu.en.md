> [← AMD Windows overview](./amd-windows.en.md)

# AMD Windows — NPU (AMD XDNA / VitisAI / Lemonade) Performance

**Hardware:** AMD Ryzen 8845H with AMD AI 300 Series NPU (XDNA, 16 TOPS)  
**Backends:** VitisAI via ONNX Runtime (OCR) · ONNX DirectML (ASR) · Lemonade Server (LLM)  
**Last calibrated:** 2026-06-19 (OCR/ASR); Lemonade LLM PENDING-VERIFY

---

## NPU Scope on AMD Windows

| Workload | NPU path | Status |
|---|---|---|
| OCR (text + structured) | VitisAI EP (onnxruntime-vitisai) | **PASS** |
| ASR | DirectML (onnxruntime-directml on XDNA) | **PASS** |
| **LLM inference** | **Lemonade Server** (AMD RyzenAI + W4A8 quantization, port 8000) | **PENDING-VERIFY** |
| Embedding / Reranker | Not supported via NPU; uses Vulkan (Ollama) or CPU (ONNX) | N/A |

**LLM on XDNA NPU is now supported** via AMD's [Lemonade Server](https://github.com/amd/lemonade)
(`pip install lemonade-server`). It runs small models (≤3.8B) directly on the XDNA NPU using
the RyzenAI W4A8 runtime — freeing the 780M iGPU for concurrent tasks.
See [iGPU mode](./amd-windows-igpu.en.md) for calibrated LLM performance via Vulkan.

---

## NPU LLM via Lemonade Server (PENDING-VERIFY)

### Setup

```cmd
pip install lemonade-server

# Phi-3.5-mini (3.8B, default recommended)
lemonade-server serve --model Phi-3.5-mini-instruct --device npu --port 8000

# Llama-3.2-1B (fastest)
lemonade-server serve --model llama3.2-1b --device npu --port 8000

# Qwen2.5-1.5B
lemonade-server serve --model Qwen2.5-1.5B-Instruct --device npu --port 8000
```

The server exposes OpenAI-compatible `/v1/chat/completions` on port 8000.

### Run Benchmark

```bash
export LEMONADE_AMD_BASE_URL="http://<AMD_HOST>:8000/v1"
python run_benchmark.py --model phi-3.5-mini-amd-npu --target amd-win-x86
python run_benchmark.py --model llama3.2-1b-amd-npu --target amd-win-x86
```

### Expected Performance (PENDING-VERIFY)

| Model | Expected TPS | iGPU baseline | Notes |
|---|---|---|---|
| `llama3.2-1b-amd-npu` | ~80–100 t/s | 25 t/s (Vulkan) | Significant NPU speedup expected |
| `phi-3.5-mini-amd-npu` | ~30–50 t/s | N/A tested | 3.8B model on XDNA |
| `qwen2.5-1.5b-amd-npu` | ~60–80 t/s | N/A tested | W4A8, multilingual |

**These values are pre-calibration estimates. Run E2E harness to calibrate thresholds.**

**Key trade-off**: NPU LLM frees iGPU for OCR/embedding concurrent workloads; useful when
running multi-modal pipelines where OCR + LLM need to overlap.

---

## Configuration

### VitisAI OCR

Requires the AMD RyzenAI runtime and `onnxruntime-vitisai` package:

```cmd
pip install onnxruntime-vitisai
```

The VitisAI EP is selected automatically when the model target is `amd-npu`:
```bash
# Set the ONNX backend before running
export OCR_BACKEND=vitisai
python run_benchmark.py --model rapidocr-amd-npu --target amd-win-x86
```

### DirectML ASR

No separate installation needed beyond standard `onnxruntime-directml`.
ASR benchmark invokes DirectML automatically:
```bash
python run_benchmark.py --model sensevoice-small-amd-win --target amd-win-x86 --skip ttft,throughput,embedding,rerank,ocr
```

---

## OCR Results (VitisAI NPU)

| Model | CER | NED | p50 | Structured field acc | Structured p50 | Status |
|---|---|---|---|---|---|---|
| `rapidocr-amd-npu` | 7.04% | 6.18% | 2031 ms | 92.86% | 1867.5 ms | **PASS** |

**Quality identical to DirectML and CPU paths** (CER 7.04% is the dataset floor).
**Latency higher than DirectML**: VitisAI p50 is 2031 ms vs DirectML 469 ms.
Use VitisAI for thermal/power-constrained batch workloads where iGPU bandwidth is needed elsewhere.

---

## ASR Results (DirectML — AMD platform)

| Model | CER | RTF | Status |
|---|---|---|---|
| `sensevoice-small-amd-win` | 7.69% | 0.073 | **PASS** |

**RTF 0.073** means 1 second of audio processes in 73 ms — 13.7× faster than real-time.
This is the best ASR path on AMD Windows.

---

## Performance Comparison: OCR Paths

| Path | Backend | p50 OCR | p50 Structured | Notes |
|---|---|---|---|---|
| CPU ONNX | CPU | 1593 ms | 859 ms | [→ CPU doc](./amd-windows-cpu.en.md) |
| **iGPU DirectML** | 780M | **469 ms** | **477 ms** | **Fastest — recommended** |
| NPU VitisAI | XDNA | 2031 ms | 1868 ms | Power-efficient for batch |

---

## 中文摘要

**硬件：** AMD XDNA NPU（16 TOPS），VitisAI + DirectML + Lemonade Server  
**最后校准：** 2026-06-19（OCR/ASR）；Lemonade LLM PENDING-VERIFY

### NPU 覆盖范围

| 任务 | 支持 |
|---|---|
| OCR | **PASS**（VitisAI） |
| ASR | **PASS**（DirectML） |
| **LLM 推理** | **PENDING-VERIFY**（Lemonade Server，XDNA NPU，W4A8 量化） |
| Embedding | 不支持（走 iGPU Vulkan） |

AMD XDNA NPU 现支持 LLM 推理，通过 AMD [Lemonade Server](https://github.com/amd/lemonade)（`pip install lemonade-server`）实现。支持 ≤3.8B 的小模型直接在 XDNA NPU 上运行，释放 780M iGPU 用于并发 OCR/Embedding 任务。

### 关键数据

| 模型 | 指标 | 状态 |
|---|---|---|
| rapidocr-amd-npu（OCR） | p50 2031 ms，CER 7.04% | **PASS** |
| sensevoice-small-amd-win（ASR） | RTF 0.073，CER 7.69% | **PASS** |
| llama3.2-1b-amd-npu（LLM） | 预期 ~80–100 t/s | **PENDING-VERIFY** |
| phi-3.5-mini-amd-npu（LLM） | 预期 ~30–50 t/s | **PENDING-VERIFY** |
| qwen2.5-1.5b-amd-npu（LLM） | 预期 ~60–80 t/s | **PENDING-VERIFY** |

OCR 三条路径延迟对比：CPU 1593 ms → DirectML 469 ms（最快） → VitisAI 2031 ms。
