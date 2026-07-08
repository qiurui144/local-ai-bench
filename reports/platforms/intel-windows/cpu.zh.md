# Intel Windows CPU 路径

**最后更新：** 2026-07-08
**英文版本：** [cpu.en.md](cpu.en.md)
**旧报告来源：** [../../intel-windows-cpu.en.md](../../intel-windows-cpu.en.md)

## 范围

Intel CPU 路径覆盖 Ollama CPU 的 LLM/embedding，以及 ONNX CPU reranker。它是当前该平台默认的 LLM serving 路径。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| LLM | `qwen2.5-7b-intel-win` | 8.25 TPS，TTFT p50 4820ms，GA 通过 | 翻译失败 | CPU 质量更强，但延迟高 |
| LLM | `qwen2.5-3b-intel-win` | 19.47 TPS，TTFT p50 781ms，GA 通过 | 翻译失败 | 交互 CPU 默认 |
| LLM | `llama3.2-1b-intel-win` | 25.26 TPS，TTFT p50 875ms | 质量未完整覆盖 | 仅轻量对照 |
| Embedding | `qwen3-embedding-0.6b-intel-win` | p50 617.5ms，hit@1 1.0 | 通过 | iGPU wrapper 不可用时的 CPU embedding 默认 |
| Reranker | `bge-reranker-base-intel-win` | p50 148.5ms，nDCG/MRR 1.0 | 通过 | CPU reranker 默认 |
| Reranker | `bge-reranker-v2-m3-intel-win` | p50 546.5ms，nDCG/MRR 1.0 | 通过 | 慢速 reranker 备选 |

## 结论

交互 CPU serving 使用 `qwen2.5-3b-intel-win`；只有更强 GA 能抵消延迟时才使用 `qwen2.5-7b-intel-win`。翻译失败仍是模型选型风险。
