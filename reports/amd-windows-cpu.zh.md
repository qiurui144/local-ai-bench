> [English](./amd-windows-cpu.en.md)

# AMD Windows — 纯 CPU 路径（ONNX Runtime CPU）

**硬件：** AMD Ryzen 8845H（4× Zen4 P 核 + 4× Zen4c E 核）  
**后端：** ONNX Runtime CPU 执行提供器  
**最后校准：** 2026-06-19

---

## AMD Windows 上 CPU 路径的覆盖范围

| 任务 | CPU 路径 | 备注 |
|---|---|---|
| OCR（文字 + 结构化） | ONNX Runtime CPU EP | 参考基线 |
| Reranker | ONNX Runtime CPU EP | 所有 Reranker 模型走 CPU |
| LLM 推理 | 未单独测试 | Ollama 默认使用 Vulkan iGPU；见 [iGPU 文档](./amd-windows-igpu.zh.md) |
| Embedding | 未单独测试 | Ollama Vulkan 负责 Embedding |
| ASR | 未单独测试 | 使用 DirectML 路径 |

CPU 路径作为 OCR 延迟对比的**参考基线**，同时也是 Reranker 的**生产路径**（Reranker 设计为走 CPU-ONNX — 模型足够小，CPU 开销可接受，且避免与 LLM 竞争 GPU 显存）。

---

## OCR 测试结果（CPU ONNX）

| 模型 | 目标 | CER | NED | p50 OCR | 结构化字段准确率 | 结构化 p50 | 状态 |
|---|---|---|---|---|---|---|---|
| `rapidocr-cpu` | local/reference | 7.04% | 6.18% | 1592.5 ms | 92.86% | 859.0 ms | **PASS** |
| `paddleocr-cpu` | local/reference | 7.04% | 6.18% | 1829.5 ms | — | — | **PASS** |

这两个是**参考模型** — 其数据建立了 CPU 基线，用于与 DirectML 和 VitisAI 比较。所有三条路径的质量（CER 7.04%）完全一致。

**CPU ONNX 与其他路径的 OCR 文字 p50 对比：**

| 路径 | p50 | 相对值 |
|---|---|---|
| CPU ONNX（rapidocr） | 1593 ms | 1.0× 基线 |
| CPU ONNX（paddleocr） | 1830 ms | 1.15× |
| NPU VitisAI | 2031 ms | 1.27× 比 CPU 慢 |
| **iGPU DirectML** | **469 ms** | **比 CPU 快 3.4×** |

---

## Reranker 测试结果（CPU ONNX）

无论哪种 GPU 后端处于激活状态，两个 Reranker 均在 CPU ONNX Runtime 上运行，不与 LLM/Embedding 竞争显存。

| 模型 | nDCG@10 | MRR | p50 延迟 | p95 延迟 | 状态 |
|---|---|---|---|---|---|
| `bge-reranker-base-amd-win` | 1.000 | 1.000 | 78 ms | — | **PASS** |
| `bge-reranker-v2-m3-amd-win` | 1.000 | 1.000 | 289 ms | — | **PASS** |

**推荐：** 延迟敏感场景用 `bge-reranker-base-amd-win`（p50 78 ms）。`bge-reranker-v2-m3-amd-win` 检索质量相同，但延迟高 3.7× — 仅在排序质量是瓶颈而非推理速度时使用。

---

## LLM 纯 CPU 模式估算

Ollama 在 AMD Windows 上默认使用 Vulkan GPU 卸载（本仓库所有 LLM 基准均使用 Vulkan）。强制纯 CPU 模式：

```cmd
setx /M OLLAMA_NUM_GPU 0
ollama.exe serve
```

**纯 CPU 预期性能**（从 CPU PP 速率估算）：
- 7B 模型：~2–4 TPS（vs Vulkan iGPU 的 13.33 TPS）
- 3B 模型：~6–10 TPS（vs Vulkan iGPU 的 28.99 TPS）

本硬件上**不推荐**使用 CPU 纯推理模式进行交互式使用。
