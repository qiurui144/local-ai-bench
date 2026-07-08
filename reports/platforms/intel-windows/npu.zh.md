# Intel Windows NPU 路径

**最后更新：** 2026-07-08
**英文版本：** [npu.en.md](npu.en.md)
**来源：** 从 [../../intel-windows.en.md](../../intel-windows.en.md) 和 [../../intel-windows-igpu.en.md](../../intel-windows-igpu.en.md) 拆分

## 范围

Intel AI Boost NPU 路径使用 OpenVINO NPU/VPUX。它已验证静态 shape OCR 模型和 Whisper encoder。动态 shape 的 embedding、reranker、SenseVoice ASR 在当前导出/runtime shape 约束下失败。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| OCR 检测 | PP-OCRv4 det static `[1,3,640,640]` | 33ms | 通过 | NPU pipeline 组件 |
| OCR 识别 | PP-OCRv4 rec static `[1,3,48,320]` | 11ms | 通过 | 需要 H=48 静态 reshape |
| OCR 分类 | PP-OCRv4 cls static `[1,3,48,192]` | 3ms | 通过 | NPU pipeline 组件 |
| ASR encoder | Whisper-base INT8 encoder static `[1,80,3000]` | 115ms | 通过 | 仅 encoder；decoder 仍在 CPU |
| Embedding | BGE INT8 OpenVINO | 动态 shape 失败 | 失败 | 使用 iGPU 或 CPU |
| Reranker | BGE reranker INT8 OpenVINO | 动态 shape 失败 | 失败 | 使用 iGPU 或 CPU |
| SenseVoice ASR | SenseVoice ONNX | 动态 self-attention mask 问题 | 失败 | 使用 DirectML 路径 |

## 结论

Intel NPU 必须作为单独报告路径保留，因为其通过/失败特征和 shape 强相关。它适合静态 OCR 组件和部分 encoder，不适合通用 LLM、embedding 或 reranker serving。
