# AMD Windows iGPU 路径

**最后更新：** 2026-07-08
**英文版本：** [igpu.en.md](igpu.en.md)
**旧报告来源：** [../../amd-windows-igpu.en.md](../../amd-windows-igpu.en.md)

## 范围

Radeon 780M iGPU 路径覆盖 Ollama Vulkan 的 LLM/embedding，以及 ONNX DirectML 的 OCR。它是当前 AMD Windows 基准集中最实用的加速路径。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| LLM | `qwen2.5-7b-amd-win` | 13.33 TPS，TTFT p50 953ms | 已测，有质量风险 | 需按具体任务验证后使用 |
| LLM | `llama3.2-3b-amd-win` | 28.99 TPS，TTFT p50 890ms | 已测，有质量风险 | 轻量/并发控制模型 |
| LLM | `qwen2.5-14b-amd-win` | 8.6 TPS | 已测 | 仅在必须提高参数规模时使用 |
| Embedding | `qwen3-embedding-0.6b-amd` | p50 875ms，hit@1 1.0 | 通过 | AMD embedding 默认路径 |
| Embedding | `bge-m3-amd` | p50 914ms，hit@1 1.0 | 通过 | 多语言备选 |
| OCR | `rapidocr-amd-directml` | p50 468.5ms，CER 7.04% | 通过 | AMD 最快 OCR 路径 |

## 结论

AMD 的 LLM 性能覆盖和最快 OCR 路径使用 iGPU。当前 LLM 行性能有效，但不是干净的质量通过，因此生产选型必须绑定具体任务和 prompt 类型。
