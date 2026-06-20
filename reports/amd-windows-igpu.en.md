> [中文版](./amd-windows-igpu.zh.md)
> [← AMD Windows overview](./amd-windows.en.md)

# AMD Windows — iGPU (Radeon 780M / Vulkan + DirectML) Performance

**Hardware:** AMD Ryzen 8845H + Radeon 780M (RDNA3), 17.9 GiB shared VRAM  
**Backends:** Ollama Vulkan (LLM / Embedding) · ONNX DirectML (OCR)  
**Last calibrated:** 2026-06-19

---

## Configuration

### Ollama — enable Vulkan GPU offload

The Radeon 780M requires `HSA_OVERRIDE_GFX_VERSION=gfx1102` to be recognized by Ollama's
ROCm/Vulkan path. Set before starting `ollama serve`:

```cmd
setx /M OLLAMA_HOST 0.0.0.0
setx /M HSA_OVERRIDE_GFX_VERSION gfx1102
ollama.exe serve
```

Verify GPU is active — `ollama ps` should show `100% GPU` for a running model:

```cmd
ollama ps
# NAME         ID       SIZE    PROCESSOR  UNTIL
# qwen2.5:7b   ...      5.2GB   100% GPU   ...
```

### ONNX DirectML OCR — no extra config needed

`rapidocr-amd-directml` uses the DirectML backend from `onnxruntime-directml`.
No additional environment variables required; DirectML auto-selects the 780M.

---

## LLM Results (Ollama Vulkan)

| Model | TPS | TTFT p50 | TTFT p95 | PP t/s | TG t/s | Max ctx | Status |
|---|---|---|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | 13.33 | 953 ms | 6241 ms | 116 | 16 | 16k | FAIL (quality) |
| `qwen2.5-14b-amd-win` | 7.67 | 8274 ms | 14792 ms | 94 | 9 | 16k | MEASURED |
| `llama3.2-3b-amd-win` | 28.99 | 890 ms | 5207 ms | 124 | 39 | 32k | FAIL (quality) |
| `qwen3-0.6b-amd` | 91.09 | 1781 ms | 1781 ms | — | — | — | FAIL (quality) |
| `llava-7b-amd-win` | 16.84 | 890 ms | 891 ms | 835 | 19 | — | FAIL (accuracy) |

**PP/TG** = Prefill / Token Generation tokens-per-second (llama-bench style).  
**Status note:** FAIL on quality dimensions (translation/scenarios/conditioned) means the model
does not pass the harness quality gates on this hardware tier. Performance numbers are valid and
measured.

### Concurrency (Vulkan)

| Model | Peak concurrency | Sustained TPS at peak |
|---|---|---|
| `llama3.2-3b-amd-win` | c50 | 36.21 t/s |
| `llama3.2-3b-amd-win` | c16 limit | 37.88 t/s |
| `qwen2.5-7b-amd-win` | c8 | 16.70 t/s |
| `qwen2.5-14b-amd-win` | c8 limit | 8.95 t/s |

---

## Embedding Results (Ollama Vulkan)

| Model | hit@1 | nDCG@10 | MRR | p50 latency | Status |
|---|---|---|---|---|---|
| `qwen3-embedding-0.6b-amd` | 1.000 | 1.000 | 1.000 | 875 ms | **PASS** |
| `bge-m3-amd` | 1.000 | 1.000 | 1.000 | 914 ms | **PASS** |

---

## OCR Results (ONNX DirectML — 780M)

| Model | CER | NED | p50 | Structured field acc | Structured p50 | Status |
|---|---|---|---|---|---|---|
| `rapidocr-amd-directml` | 7.04% | 6.18% | 468.5 ms | 92.86% | 476.5 ms | **PASS** |

**DirectML is the fastest OCR path on AMD Windows** — 3.4× faster than CPU ONNX, 4.3× faster
than VitisAI NPU.

---

## 中文摘要

**硬件：** Radeon 780M（RDNA3），Vulkan + DirectML 后端  
**最后校准：** 2026-06-19

### 配置方式

Ollama 启动前设置：
```cmd
setx /M HSA_OVERRIDE_GFX_VERSION gfx1102
ollama.exe serve
```
ONNX DirectML OCR 无需额外配置。

### 关键性能数据

| 模型 | TPS | TTFT p50 | 状态 |
|---|---|---|---|
| qwen2.5-7b（LLM） | 13.33 | 953 ms | FAIL（质量维度） |
| llama3.2-3b（LLM） | 28.99 | 890 ms | FAIL（质量维度） |
| qwen3-embedding-0.6b | — | 875 ms | **PASS** |
| rapidocr-directml（OCR） | — | 469 ms | **PASS**（最快路径） |
