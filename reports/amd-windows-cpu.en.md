> [← AMD Windows overview](./amd-windows.en.md)

# AMD Windows — CPU-Only Paths (ONNX Runtime CPU)

**Hardware:** AMD Ryzen 8845H (4× Zen4 P-core + 4× Zen4c E-core)  
**Backend:** ONNX Runtime CPU execution provider  
**Last calibrated:** 2026-06-19

---

## CPU Scope on AMD Windows

| Workload | CPU path | Notes |
|---|---|---|
| OCR (text + structured) | ONNX Runtime CPU EP | Reference baseline |
| Reranker | ONNX Runtime CPU EP | All reranker models run CPU |
| LLM inference | Not benchmarked separately | Ollama defaults to Vulkan iGPU; see [iGPU doc](./amd-windows-igpu.en.md) |
| Embedding | Not benchmarked separately | Ollama Vulkan handles embedding |
| ASR | Not benchmarked separately | DirectML path used |

The CPU path serves as the **reference baseline** for OCR latency comparison and as the
**production path** for the reranker (rerankers are CPU-ONNX by design — small enough that
CPU overhead is acceptable and avoids GPU memory competition with LLMs).

---

## OCR Results (CPU ONNX)

| Model | Target | CER | NED | p50 OCR | Structured field acc | Structured p50 | Status |
|---|---|---|---|---|---|---|---|
| `rapidocr-cpu` | local/reference | 7.04% | 6.18% | 1592.5 ms | 92.86% | 859.0 ms | **PASS** |
| `paddleocr-cpu` | local/reference | 7.04% | 6.18% | 1829.5 ms | — | — | **PASS** |

These are the **reference models** — their numbers establish the CPU baseline against which
DirectML and VitisAI are compared. Quality (CER 7.04%) is identical across all three paths.

**CPU ONNX vs other paths (text OCR p50):**

| Path | p50 | Relative |
|---|---|---|
| CPU ONNX (rapidocr) | 1593 ms | 1.0× baseline |
| CPU ONNX (paddleocr) | 1830 ms | 1.15× |
| NPU VitisAI | 2031 ms | 1.27× slower than CPU |
| **iGPU DirectML** | **469 ms** | **3.4× faster than CPU** |

---

## Reranker Results (CPU ONNX)

Both rerankers run on CPU ONNX Runtime regardless of which GPU backend is active. They do not
compete with LLM/Embedding for VRAM.

| Model | nDCG@10 | MRR | p50 latency | p95 latency | Status |
|---|---|---|---|---|---|
| `bge-reranker-base-amd-win` | 1.000 | 1.000 | 78 ms | — | **PASS** |
| `bge-reranker-v2-m3-amd-win` | 1.000 | 1.000 | 289 ms | — | **PASS** |

**Recommendation:** `bge-reranker-base-amd-win` for latency-sensitive paths (p50 78 ms).
`bge-reranker-v2-m3-amd-win` offers the same retrieval quality at 3.7× the latency — use only
when re-ranking quality is the bottleneck rather than inference speed.

---

## LLM CPU-Only Estimation

Ollama's default on AMD Windows is Vulkan GPU offload (all LLM benchmarks in this repo used
Vulkan). To force CPU-only mode:

```cmd
setx /M OLLAMA_NUM_GPU 0
ollama.exe serve
```

**Expected CPU-only performance** (estimated from CPU PP rates):
- 7B model: ~2–4 TPS (vs 13.33 TPS on Vulkan iGPU)
- 3B model: ~6–10 TPS (vs 28.99 TPS on Vulkan iGPU)

CPU-only LLM mode is **not recommended** for interactive use on this hardware.

---

## 中文摘要

**硬件：** Ryzen 8845H CPU，ONNX Runtime CPU 执行提供器  
**最后校准：** 2026-06-19

### CPU 路径覆盖范围

- OCR：ONNX CPU（基线参考）
- Reranker：ONNX CPU（生产路径）
- LLM：未单独测试（建议用 iGPU Vulkan 路径）

### 关键数据

| 模型 | p50 | 状态 |
|---|---|---|
| rapidocr-cpu（OCR 文字） | 1593 ms | PASS（基线） |
| paddleocr-cpu（OCR 文字） | 1830 ms | PASS |
| bge-reranker-base（Reranker） | 78 ms | **PASS** |
| bge-reranker-v2-m3（Reranker） | 289 ms | PASS |
