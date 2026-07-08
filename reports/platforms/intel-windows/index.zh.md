# Intel Windows

**最后更新：** 2026-07-08
**英文版本：** [index.en.md](index.en.md)
**旧报告来源：** [../../intel-windows.en.md](../../intel-windows.en.md)

## 范围

Intel Windows 拆分为 CPU、Intel Arc iGPU 和 Intel AI Boost NPU 三条路径。CPU Ollama 是当前已校准的 LLM 路径。OpenVINO iGPU 已验证 OCR/embedding/reranker 和实验性 LLM。NPU 已验证静态 shape OCR 和 Whisper encoder，但动态 embedding/reranker 失败。

## 执行路径摘要

| 路径 | Runtime | 最适合工作负载 | 状态 |
|---|---|---|---|
| [CPU](cpu.zh.md) | Ollama CPU、ONNX Runtime CPU | LLM、embedding、reranker | LLM 有翻译风险，其余通过 |
| [iGPU](igpu.zh.md) | OpenVINO / optimum-intel | OCR、embedding、reranker、实验性 LLM | 非 LLM 通过；LLM serving 待补 |
| [NPU](npu.zh.md) | OpenVINO NPU / VPUX | 静态 OCR、Whisper encoder | 静态模型通过；动态模型失败 |

## 选型说明

| 角色 | 当前选择 | 结论 |
|---|---|---|
| LLM 路径 | CPU Ollama | 交互用 `qwen2.5-3b`，更强 GA 用 `qwen2.5-7b` 但延迟更高。 |
| OCR 路径 | OpenVINO iGPU，或 pipeline 支持时用 NPU 静态 OCR | DirectML OCR 在该平台不可用。 |
| Embedding/reranker 路径 | CPU 或 OpenVINO iGPU | 有 serving wrapper 时 iGPU warm path 更快。 |
| NPU 路径 | 静态 OCR 和 Whisper encoder | 需要单独保留，因为动态 shape 失败是路径特定问题。 |

## 证据

| 详情 | 报告 |
|---|---|
| CPU 路径 | [cpu.zh.md](cpu.zh.md) |
| iGPU 路径 | [igpu.zh.md](igpu.zh.md) |
| NPU 路径 | [npu.zh.md](npu.zh.md) |
| 旧完整报告 | [../../intel-windows.en.md](../../intel-windows.en.md) |
| 旧 CPU 详情 | [../../intel-windows-cpu.en.md](../../intel-windows-cpu.en.md) |
| 旧 iGPU/NPU 详情 | [../../intel-windows-igpu.en.md](../../intel-windows-igpu.en.md) |
