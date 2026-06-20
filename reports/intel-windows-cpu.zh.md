> [English](./intel-windows-cpu.en.md)

# Intel Windows — CPU 模式（Ollama CPU + ONNX Runtime CPU）

**硬件：** Intel 笔电 CPU（Core Ultra 系列）  
**后端：** Ollama 纯 CPU（LLM / Embedding）· ONNX Runtime CPU EP（Reranker）  
**最后校准：** 2026-06-19

---

## 配置

本 Intel 平台上 Ollama 以**纯 CPU 模式**运行 — 未配置 GPU 卸载。这是当前默认设置；iGPU 加速路径单独记录于 [intel-windows-igpu.zh.md](./intel-windows-igpu.zh.md)。

无需特殊环境变量：

```cmd
# 默认 Ollama 启动 — CPU 模式
ollama.exe serve
```

验证 CPU 模式已激活（无 GPU 层）：
```cmd
ollama ps
# NAME         ...  PROCESSOR  UNTIL
# qwen2.5:3b   ...  100% CPU   ...
```

---

## LLM 测试结果（Ollama CPU）

| 模型 | TPS | TTFT p50 | TTFT p95 | PP t/s | TG t/s | 最大上下文 | 状态 |
|---|---|---|---|---|---|---|---|
| `qwen2.5-7b-intel-win` | 8.25 | 4820 ms | 8441 ms | 112 | 9 | 16k | MEASURED |
| `qwen2.5-3b-intel-win` | 19.47 | 781 ms | 3495 ms | 124 | 26 | 16k | FAIL（质量）|
| `llama3.2-1b-intel-win` | 25.26 | 875 ms | 3308 ms | 130 | 35 | 32k | FAIL（质量）|
| `llava-7b-intel-win` | 10.02 | 703 ms | 703 ms | 1074 | 11 | — | FAIL（准确率）|

**MEASURED** = 已采集延迟和吞吐；质量维度未完整认证。  
**FAIL（质量）** = 性能指标有效；质量门控（translation / general_ability）未通过。

### TTFT 上下文（Intel CPU vs AMD iGPU）

| 模型 | Intel CPU p50 | AMD iGPU p50 | 比率 |
|---|---|---|---|
| 7B | 4820 ms | 953 ms | Intel CPU 慢 5.1× |
| 3B | 781 ms | 890 ms | 相当（AMD 略慢） |
| 1B | 875 ms | — | — |

7B TTFT 差距由缺少 GPU 卸载解释 — Intel 上预填充完全由 CPU 完成。交互场景首选 `qwen2.5-3b-intel-win`。

### 并发（CPU）

| 模型 | 峰值并发 | 峰值下持续 TPS |
|---|---|---|
| `llama3.2-1b-intel-win` | c32 上限 | 32.52 t/s |
| `qwen2.5-3b-intel-win` | c8 | 24.68 t/s |
| `qwen2.5-7b-intel-win` | c16 上限 | 9.54 t/s |

---

## Embedding 测试结果（Ollama CPU）

| 模型 | hit@1 | nDCG@10 | MRR | p50 | 状态 |
|---|---|---|---|---|---|
| `qwen3-embedding-0.6b-intel-win` | 1.000 | 1.000 | 1.000 | 617.5 ms | **PASS** |

Intel CPU embedding p50（617.5 ms）比 AMD iGPU embedding（875 ms）更快，因为较小的 embedding 模型受内存带宽限制而非算力限制。

---

## Reranker 测试结果（CPU ONNX）

| 模型 | nDCG@10 | MRR | p50 | 状态 |
|---|---|---|---|---|
| `bge-reranker-base-intel-win` | 1.000 | 1.000 | 148.5 ms | **PASS** |
| `bge-reranker-v2-m3-intel-win` | 1.000 | 1.000 | 546.5 ms | **PASS** |

Intel CPU 上的 Reranker 延迟比 AMD CPU 高约 1.9×（base 模型：148.5 ms vs 78 ms），与 Intel 平台 CPU 频率层级较低一致。

---

**推荐：** 交互场景用 `qwen2.5-3b-intel-win`（TTFT 781 ms），高质量要求场景用 `qwen2.5-7b-intel-win`（接受 4820 ms 高延迟）。
