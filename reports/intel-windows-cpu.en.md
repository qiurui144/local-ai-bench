> [← Intel Windows overview](./intel-windows.en.md)

# Intel Windows — CPU Mode (Ollama CPU + ONNX Runtime CPU)

**Hardware:** Intel laptop CPU (Core Ultra series)  
**Backend:** Ollama CPU-only (LLM / Embedding) · ONNX Runtime CPU EP (Reranker)  
**Last calibrated:** 2026-06-19

---

## Configuration

Ollama on this Intel platform runs in **CPU-only mode** — no GPU offload is configured.
This is the current default setup; iGPU acceleration paths are documented separately in
[intel-windows-igpu.en.md](./intel-windows-igpu.en.md).

No special environment variables required:

```cmd
# Default Ollama startup — CPU mode
ollama.exe serve
```

Verify CPU mode is active (no GPU layers):
```cmd
ollama ps
# NAME         ...  PROCESSOR  UNTIL
# qwen2.5:3b   ...  100% CPU   ...
```

---

## LLM Results (Ollama CPU)

| Model | TPS | TTFT p50 | TTFT p95 | PP t/s | TG t/s | Max ctx | Status |
|---|---|---|---|---|---|---|---|
| `qwen2.5-7b-intel-win` | 8.25 | 4820 ms | 8441 ms | 112 | 9 | 16k | MEASURED |
| `qwen2.5-3b-intel-win` | 19.47 | 781 ms | 3495 ms | 124 | 26 | 16k | FAIL (quality) |
| `llama3.2-1b-intel-win` | 25.26 | 875 ms | 3308 ms | 130 | 35 | 32k | FAIL (quality) |
| `llava-7b-intel-win` | 10.02 | 703 ms | 703 ms | 1074 | 11 | — | FAIL (accuracy) |

**MEASURED** = latency and throughput collected; quality dimensions not fully qualified.  
**FAIL (quality)** = perf metrics valid; quality gates (translation / general_ability) not passed.

### TTFT Context (Intel CPU vs AMD iGPU)

| Model | Intel CPU p50 | AMD iGPU p50 | Ratio |
|---|---|---|---|
| 7B | 4820 ms | 953 ms | 5.1× slower on Intel CPU |
| 3B | 781 ms | 890 ms | Comparable (AMD slightly slower) |
| 1B | 875 ms | — | — |

The 7B TTFT gap is explained by the absence of GPU offload — prefill is entirely CPU-bound
on Intel. For interactive use, prefer `qwen2.5-3b-intel-win`.

### Concurrency (CPU)

| Model | Peak concurrency | Sustained TPS at peak |
|---|---|---|
| `llama3.2-1b-intel-win` | c32 limit | 32.52 t/s |
| `qwen2.5-3b-intel-win` | c8 | 24.68 t/s |
| `qwen2.5-7b-intel-win` | c16 limit | 9.54 t/s |

---

## Embedding Results (Ollama CPU)

| Model | hit@1 | nDCG@10 | MRR | p50 | Status |
|---|---|---|---|---|---|
| `qwen3-embedding-0.6b-intel-win` | 1.000 | 1.000 | 1.000 | 617.5 ms | **PASS** |

Intel CPU embedding p50 (617.5 ms) is faster than AMD iGPU embedding (875 ms) because
smaller embedding models are CPU-memory-bandwidth bound, not compute bound.

---

## Reranker Results (CPU ONNX)

| Model | nDCG@10 | MRR | p50 | Status |
|---|---|---|---|---|
| `bge-reranker-base-intel-win` | 1.000 | 1.000 | 148.5 ms | **PASS** |
| `bge-reranker-v2-m3-intel-win` | 1.000 | 1.000 | 546.5 ms | **PASS** |

Reranker latency on Intel CPU is ~1.9× higher than AMD CPU (148.5 ms vs 78 ms for base model),
consistent with the Intel platform being at a lower CPU frequency tier.

---

## 中文摘要

**硬件：** Intel 笔电 CPU，纯 CPU 推理（无 GPU 卸载）  
**最后校准：** 2026-06-19

### 关键数据

| 模型 | TPS | TTFT p50 | 状态 |
|---|---|---|---|
| qwen2.5-7b（LLM） | 8.25 | 4820 ms | MEASURED（延迟偏高） |
| qwen2.5-3b（LLM） | 19.47 | 781 ms | FAIL（质量维度） |
| llama3.2-1b（LLM） | 25.26 | 875 ms | FAIL（质量维度） |
| qwen3-embedding-0.6b | — | 617.5 ms | **PASS** |
| bge-reranker-base | — | 148.5 ms | **PASS** |

**推荐：** 交互场景用 `qwen2.5-3b-intel-win`（781 ms TTFT），高质量场景用 `qwen2.5-7b-intel-win`（接受高延迟）。
