# AMD Windows NPU 路径

**最后更新：** 2026-07-08
**英文版本：** [npu.en.md](npu.en.md)
**旧报告来源：** [../../amd-windows-npu.en.md](../../amd-windows-npu.en.md)

## 范围

AMD XDNA NPU 路径覆盖 VitisAI OCR 和 Lemonade/FastFlowLM LLM 候选路线。OCR 已测。该 Ryzen 8845H/XDNA 代际上的 NPU LLM 仍待验证。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| OCR | `rapidocr-amd-npu` via VitisAI | p50 2031ms，CER 7.04% | 通过 | 批处理或隔离路径，不是最快 |
| ASR | `sensevoice-small-amd-win` via DirectML | RTF 0.073，CER 7.69% | 通过 | 当前证据中 AMD 最佳 ASR 路径 |
| LLM pure NPU | Lemonade / FastFlowLM 候选 | 未校准 | 待验证 | 暂不能用于模型选型 |
| LLM hybrid iGPU+NPU | Lemonade hybrid 候选 | 未校准 | 待验证 | 需要 Ryzen AI 软件和端到端 harness 实测 |

## 结论

AMD NPU 是独立硬件路径，因此在报告架构中保留。当前模型选型中，LLM/OCR 用 iGPU，reranker 用 CPU。NPU LLM 作为后续验证工作流，不作为当前推荐。
