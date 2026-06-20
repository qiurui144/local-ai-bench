> [← AMD Windows overview](./amd-windows.en.md)

# AMD Windows — NPU (AMD XDNA / VitisAI) Performance

**Hardware:** AMD Ryzen 8845H with AMD AI 300 Series NPU (XDNA, 16 TOPS)  
**Backends:** VitisAI via ONNX Runtime (OCR) · ONNX DirectML (ASR)  
**Last calibrated:** 2026-06-19

---

## NPU Scope on AMD Windows

| Workload | NPU path | Status |
|---|---|---|
| OCR (text + structured) | VitisAI EP (onnxruntime-vitisai) | **PASS** |
| ASR | DirectML (onnxruntime-directml on XDNA) | **PASS** |
| LLM inference | Not supported — AMD XDNA requires proprietary AMD NPU SDK; Ollama uses Vulkan iGPU instead | N/A |
| Embedding / Reranker | Not supported via NPU; uses Vulkan (Ollama) or CPU (ONNX) | N/A |

The AMD XDNA NPU targets fixed-function ONNX model execution. General-purpose LLM
serving (like Ollama) does not use it — LLM inference routes to the 780M iGPU via Vulkan.
See [iGPU mode](./amd-windows-igpu.en.md) for LLM performance.

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

**硬件：** AMD XDNA NPU（16 TOPS），VitisAI + DirectML  
**最后校准：** 2026-06-19

### NPU 覆盖范围

| 任务 | 支持 |
|---|---|
| OCR | **PASS**（VitisAI） |
| ASR | **PASS**（DirectML） |
| LLM/Embedding | 不支持（走 iGPU Vulkan） |

### 关键数据

| 模型 | 指标 | 状态 |
|---|---|---|
| rapidocr-amd-npu（OCR） | p50 2031 ms，CER 7.04% | **PASS** |
| sensevoice-small-amd-win（ASR） | RTF 0.073，CER 7.69% | **PASS** |

OCR 三条路径延迟对比：CPU 1593 ms → DirectML 469 ms（最快） → VitisAI 2031 ms。
