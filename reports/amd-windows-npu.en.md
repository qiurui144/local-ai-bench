> [← AMD Windows overview](./amd-windows.en.md)

# AMD Windows — NPU (AMD XDNA / VitisAI / Lemonade / FastFlowLM) Performance

**Hardware:** AMD Ryzen 8845H + Radeon 780M iGPU + AMD XDNA NPU (16 TOPS)  
**Backends:** VitisAI via ONNX Runtime (OCR) · ONNX DirectML (ASR) · Lemonade Server / FastFlowLM (LLM NPU)  
**Last calibrated:** 2026-06-19 (OCR/ASR); Lemonade LLM PENDING-VERIFY

---

## NPU Scope on AMD Windows

| Workload | NPU path | Status |
|---|---|---|
| OCR (text + structured) | VitisAI EP (onnxruntime-vitisai) | **PASS** |
| ASR | DirectML (onnxruntime-directml on XDNA) | **PASS** |
| **LLM — NPU pure** | **Lemonade Server** `--device npu` (W4A8 quantization) | **PENDING-VERIFY** |
| **LLM — Hybrid (iGPU+NPU)** | **Lemonade Server** `--device hybrid` (Prefill iGPU, Decode NPU) | **PENDING-VERIFY** ⚠️ Ryzen AI 300 preferred |
| **LLM — pure NPU alt** | **FastFlowLM** (no iGPU involvement) | **PENDING-VERIFY** |
| Embedding / Reranker | Not supported via NPU; uses Vulkan (Ollama) or CPU (ONNX) | N/A |

---

## Three-Mode LLM Inference Architecture

AMD Windows supports three distinct LLM inference paths on the same hardware:

| Mode | Prefill | Decode | Tool | Model size | When to use |
|---|---|---|---|---|---|
| **iGPU (Vulkan)** | Radeon 780M | Radeon 780M | Ollama / llama.cpp | ≤14B (calibrated) | Default — best latency, all sizes |
| **Hybrid (iGPU+NPU)** | Radeon 780M (fast prefill) | XDNA NPU (power-efficient decode) | Lemonade `--device hybrid` | ≤7B | Prolonged decode, thermal constraint |
| **Pure NPU** | XDNA NPU | XDNA NPU | Lemonade `--device npu` / FastFlowLM | ≤3.8B | iGPU fully occupied by OCR/ASR |

**Calibrated values**: iGPU Vulkan is the only mode with E2E-calibrated thresholds (qwen2.5-7b: 13.33 TPS; qwen3-0.6b: 91 TPS). Hybrid and Pure NPU modes are PENDING-VERIFY.

### Hybrid Mode Architecture Detail

```
User prompt → Lemonade Server (--device hybrid)
                 │
     ┌───────────┴──────────────┐
     │ Prefill (prompt encoding) │  ← Radeon 780M iGPU (high TFLOPS, fast)
     └───────────┬──────────────┘
                 │ KV-cache handed off
     ┌───────────┴──────────────┐
     │ Decode (token generation) │  ← AMD XDNA NPU (W4A8 quantized, power-efficient)
     └───────────┬──────────────┘
                 │
              Response
```

**Why hybrid?** Prefill is compute-intensive (benefits from iGPU TFLOPS); decode is memory-bandwidth-intensive and sustained (NPU W4A8 is efficient for long decode).

---

## ⚠️ Hardware Generation Caveat

> **Full hybrid mode is documented for Ryzen AI 300 series (STX/KRK, e.g. Ryzen AI 7 H350) with XDNA+ NPU and Radeon 860M iGPU.** Our test device (Ryzen 8845H) is the previous generation: XDNA NPU (not XDNA+) and Radeon 780M (RDNA3, not RDNA3.5).

| Feature | Ryzen 8845H (our device) | Ryzen AI 300 / H350 (STX) |
|---|---|---|
| CPU gen | Hawk Point | Strix Point |
| iGPU | Radeon 780M (RDNA3, 12 CU) | Radeon 860M (RDNA3.5, 16 CU) |
| NPU | XDNA (16 TOPS) | XDNA+ (50 TOPS) |
| Hybrid mode support | Expected but PENDING-VERIFY | Documented + tested (article) |
| Lemonade version | ≥1.0 | ≥1.0 |

Pure NPU mode (Lemonade `--device npu`) and OCR/ASR via VitisAI/DirectML are not gated on NPU generation and are expected to work on 8845H.

---

## Hardware Setup (Required Before NPU LLM)

### BIOS: UMA Frame Buffer Size

Increase UMA Frame Buffer allocation for hybrid mode:

```
BIOS → Advanced → AMD CBS → GFX Configuration → UMA Frame buffer Size → 4GB or 8GB
```

Default is often 512 MB — insufficient for hybrid mode. Set to 4–8 GB.

### NPU Turbo Mode

Enable NPU turbo for maximum NPU decode TPS:

```cmd
# Run as Administrator
xrt-smi configure --pmode turbo
```

Verify:
```cmd
xrt-smi examine --report platform
```

### Ryzen AI Software Version

Hybrid and FastFlowLM modes require **Ryzen AI Software 1.7+**:
- Installer: `ryzen-ai-lt-1.7.1.exe` (available in `drivers/amd-win/`)
- NPU driver: NPU_RAI1.6.1_314_WHQL.zip (recommended) or NPU_RAI1.5_280_WHQL.zip

---

## NPU LLM via Lemonade Server (PENDING-VERIFY)

### Install

```cmd
pip install lemonade-server
```

### Models and Commands

**Standard models (confirmed for XDNA NPU):**

```cmd
# Pure NPU mode
lemonade-server serve --model Phi-3.5-mini-instruct --device npu --port 8000
lemonade-server serve --model llama3.2-1b --device npu --port 8000
lemonade-server serve --model Qwen2.5-1.5B-Instruct --device npu --port 8000
```

**Hybrid mode (iGPU prefill + NPU decode):**

```cmd
# Hybrid mode — requires iGPU visible to Lemonade
lemonade-server serve --model DeepSeek-R1-Distill-Qwen-7B-Hybrid --device hybrid --port 8000
```

**Pure NPU models (FastFlowLM format):**

```cmd
# FastFlowLM pure NPU (no iGPU used)
lemonade-server serve --model DeepSeek-R1-Distill-Qwen-7B-NPU --device npu --port 8000
```

**Large MoE model (Gemma MoE, via Lemonade):**

```cmd
# Gemma-4-26B-A4B (Mixture-of-Experts, active params ~4B)
lemonade-server serve --model Gemma-4-26B-A4B-it-GGUF --device hybrid --port 8000
```

The server exposes OpenAI-compatible `/v1/chat/completions` on port 8000.

### Run Benchmark

```bash
export LEMONADE_AMD_BASE_URL="http://<AMD_HOST>:8000/v1"
python run_benchmark.py --model phi-3.5-mini-amd-npu --target amd-win-x86
python run_benchmark.py --model llama3.2-1b-amd-npu --target amd-win-x86
```

### Expected Performance (PENDING-VERIFY — no measured numbers)

| Model | Mode | Expected TPS | iGPU Vulkan baseline | Notes |
|---|---|---|---|---|
| `llama3.2-1b` | pure NPU | ~80–100 t/s | 25 t/s (Vulkan) | Reported significant NPU speedup |
| `Phi-3.5-mini` | pure NPU | ~30–50 t/s | N/A tested | 3.8B W4A8 |
| `Qwen2.5-1.5B` | pure NPU | ~60–80 t/s | N/A tested | W4A8, multilingual |
| `DeepSeek-R1-Distill-Qwen-7B` | hybrid | PENDING | N/A | Hybrid: iGPU prefill + NPU decode |
| `DeepSeek-R1-Distill-Qwen-7B` | pure NPU | PENDING | N/A | FastFlowLM format |
| `Gemma-4-26B-A4B-it-GGUF` | hybrid | PENDING | N/A | MoE: ~4B active params |

**No measured numbers from external sources.** All TPS values above are pre-calibration estimates from community reports. Run E2E harness to calibrate thresholds.

**Key trade-off**: NPU LLM frees iGPU for OCR/Embedding concurrent workloads; useful when running multi-modal pipelines where OCR + LLM need to overlap.

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

**硬件：** AMD Ryzen 8845H + Radeon 780M + AMD XDNA NPU（16 TOPS）  
**后端：** VitisAI (OCR) · DirectML (ASR) · Lemonade Server / FastFlowLM (LLM)  
**最后校准：** 2026-06-19（OCR/ASR）；Lemonade LLM PENDING-VERIFY

### 三模式 LLM 推理对比

| 模式 | Prefill | Decode | 工具 | 适用场景 |
|---|---|---|---|---|
| **iGPU Vulkan（已校准）** | Radeon 780M | Radeon 780M | Ollama | 默认推荐，延迟最低 |
| **Hybrid（PENDING-VERIFY）** | Radeon 780M（高 TFLOPS） | XDNA NPU（W4A8 节能） | Lemonade `--device hybrid` | 长时间 decode，热约束场景 |
| **纯 NPU（PENDING-VERIFY）** | XDNA NPU | XDNA NPU | Lemonade/FastFlowLM | iGPU 被 OCR/ASR 占用时 |

### ⚠️ 硬件代际限制

Hybrid 模式的完整支持在 **Ryzen AI 300 系列（STX/KRK，如 Ryzen AI 7 H350，XDNA+ 50 TOPS）** 上经过验证。本测试设备 Ryzen 8845H 为上一代（XDNA 16 TOPS，Radeon 780M RDNA3），hybrid 模式为 PENDING-VERIFY。纯 NPU 模式（Lemonade `--device npu`）和 OCR/ASR（VitisAI/DirectML）不受代际限制，预期在 8845H 上可用。

### 硬件前置配置

- **BIOS**：UMA Frame Buffer Size 设为 4–8 GB（默认 512 MB 不足以支持 Hybrid 模式）
- **NPU 超速**：以管理员运行 `xrt-smi configure --pmode turbo`
- **驱动版本**：Ryzen AI Software 1.7+（`ryzen-ai-lt-1.7.1.exe`），NPU 驱动 v1.6.1 WHQL

### NPU 支持的新模型（PENDING-VERIFY）

| 模型 | 模式 | 参数量 |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B-Hybrid | Hybrid | 7B |
| DeepSeek-R1-Distill-Qwen-7B-NPU | 纯 NPU（FastFlowLM） | 7B |
| Gemma-4-26B-A4B-it-GGUF | Hybrid | 26B（活跃 ~4B，MoE） |

### NPU 覆盖范围

| 任务 | 支持 |
|---|---|
| OCR | **PASS**（VitisAI，p50 2031 ms，CER 7.04%） |
| ASR | **PASS**（DirectML，RTF 0.073，CER 7.69%） |
| LLM 纯 NPU（≤3.8B） | **PENDING-VERIFY**（Lemonade `--device npu`） |
| LLM Hybrid（iGPU+NPU，≤7B） | **PENDING-VERIFY**（Lemonade `--device hybrid`） ⚠️ Ryzen AI 300 preferred |
| LLM FastFlowLM | **PENDING-VERIFY** |
| Embedding | 不支持 NPU，走 iGPU Vulkan |

### OCR 路径延迟对比

CPU 1593 ms → DirectML 469 ms（最快，推荐） → VitisAI 2031 ms（节能批处理）
