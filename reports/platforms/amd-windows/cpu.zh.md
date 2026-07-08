# AMD Windows CPU 路径

**最后更新：** 2026-07-08
**英文版本：** [cpu.en.md](cpu.en.md)
**旧报告来源：** [../../amd-windows-cpu.en.md](../../amd-windows-cpu.en.md)

## 范围

AMD CPU 路径是 ONNX Runtime CPU 的 OCR 基线，也是 ONNX reranker 的生产路径。该平台的 LLM 和 embedding 测量使用 Radeon 780M iGPU 路径。

## 工作负载结果

| 工作负载 | 模型/路径 | 指标 | 状态 | 结论 |
|---|---|---:|---|---|
| OCR 基线 | `rapidocr-cpu` | p50 1592.5ms，CER 7.04% | 通过 | 仅作为参考基线 |
| OCR 基线 | `paddleocr-cpu` | p50 1829.5ms，CER 7.04% | 通过 | 较慢基线 |
| Reranker | `bge-reranker-base-amd-win` | p50 78ms，nDCG/MRR 1.0 | 通过 | 默认 reranker |
| Reranker | `bge-reranker-v2-m3-amd-win` | p50 289ms，nDCG/MRR 1.0 | 通过 | 只有 rerank 质量值得牺牲延迟时使用 |

## 结论

AMD 上 reranking 使用 CPU。OCR 优先用 DirectML iGPU，除非需要热隔离或批处理调度才考虑 NPU/CPU 路径。CPU-only LLM 不推荐；LLM 数据见 iGPU 报告。
