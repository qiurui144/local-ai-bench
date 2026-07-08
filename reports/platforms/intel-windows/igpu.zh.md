# Intel Windows iGPU 路径

**最后更新：** 2026-07-08
**英文版本：** [igpu.en.md](igpu.en.md)
**旧报告来源：** [../../intel-windows-igpu.en.md](../../intel-windows-igpu.en.md)

## 范围

Intel Arc iGPU 路径使用 OpenVINO 和 optimum-intel。它已验证 OCR、embedding 和 reranker。LLM 推理已通过 optimum-intel 确认，但仍需要稳定 serving 层才能与完整 benchmark 对齐。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| LLM | `qwen2.5-7b-int4-ov` | 8.1 TPS，冷启动 115s | 已确认 | HTTP serving 稳定前保持实验性 |
| LLM | `qwen2.5-1.5b-int4-ov` | 10.6 TPS，冷启动 54s | 已确认 | 实验性对照 |
| Embedding | `bge-base-en-v1.5-int8-ov` | warm latency 约 25ms | 通过 | 有 service wrapper 时的快速 iGPU embedding |
| Reranker | `bge-reranker-base-int8-ov` | 平均 36.4ms | 通过 | 有 wrapper 时的快速 iGPU reranker |
| OCR | `rapidocr-openvino` | OCR p50 797ms | 通过 | 推荐 Intel OCR 路径 |

## 结论

应用可托管 wrapper 时，OpenVINO iGPU 用于 OCR 和加速 embedding/reranker。LLM 仍以 CPU 作为已校准报告默认，直到 OpenVINO serving 接入 benchmark harness。
