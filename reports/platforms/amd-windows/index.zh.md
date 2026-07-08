# AMD Windows

**最后更新：** 2026-07-08
**英文版本：** [index.en.md](index.en.md)
**旧报告来源：** [../../amd-windows.en.md](../../amd-windows.en.md)

## 范围

AMD Windows 拆分为 CPU、Radeon 780M iGPU 和 AMD XDNA NPU 三条路径。当前证据中，iGPU 是实际可用的 LLM/OCR 加速路径。CPU 保留为 reranker 路径和 OCR 基线。NPU 已验证 VitisAI OCR，但 LLM serving 仍待验证。

## 执行路径摘要

| 路径 | Runtime | 最适合工作负载 | 状态 |
|---|---|---|---|
| [CPU](cpu.zh.md) | ONNX Runtime CPU EP | Reranker、OCR 基线 | 通过 |
| [iGPU](igpu.zh.md) | Ollama Vulkan、ONNX DirectML | LLM、embedding、最快 OCR | 已测；LLM 有质量风险 |
| [NPU](npu.zh.md) | VitisAI、DirectML、Lemonade/FastFlowLM 候选 | OCR 批处理、ASR、未来 LLM NPU | OCR/ASR 通过；LLM 待验证 |

## 选型说明

| 角色 | 当前选择 | 结论 |
|---|---|---|
| LLM 路径 | Radeon 780M iGPU via Ollama Vulkan | `qwen2.5-7b` 和 `llama3.2-3b` 有有效性能数据，但当前质量门禁不是干净通过。 |
| OCR 路径 | Radeon 780M DirectML | `rapidocr-amd-directml` 最快：p50 468.5ms。 |
| Reranker 路径 | CPU ONNX | `bge-reranker-base-amd-win` p50 78ms；延迟敏感场景默认。 |
| NPU 路径 | VitisAI OCR / 待验证 LLM NPU | OCR 可用但慢于 DirectML；适合隔离 iGPU 或做批处理/功耗实验。 |

## 证据

| 详情 | 报告 |
|---|---|
| CPU 路径 | [cpu.zh.md](cpu.zh.md) |
| iGPU 路径 | [igpu.zh.md](igpu.zh.md) |
| NPU 路径 | [npu.zh.md](npu.zh.md) |
| 旧完整报告 | [../../amd-windows.en.md](../../amd-windows.en.md) |
| 旧 CPU 详情 | [../../amd-windows-cpu.en.md](../../amd-windows-cpu.en.md) |
| 旧 iGPU 详情 | [../../amd-windows-igpu.en.md](../../amd-windows-igpu.en.md) |
| 旧 NPU 详情 | [../../amd-windows-npu.en.md](../../amd-windows-npu.en.md) |
