# K3 RISC-V 16G CPU ORT / sherpa

**最后更新：** 2026-07-14
**英文版本：** [cpu-ort.en.md](cpu-ort.en.md)
**旧报告来源：** [../../k3-riscv.en.md](../../k3-riscv.en.md)

## 范围

本页覆盖 K3 16G CPU 侧 ONNX Runtime 和 sherpa-onnx 底层模型的旧证据。它不是当前 v1 NAS contract 产物。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| Embedding | `bge-m3` ONNX | 77ms cold search | PASS | 旧本地 retrieval 默认路径 |
| Reranker | `bge-reranker-base` ORT | ranking correct | PASS | 旧本地 reranker 默认路径 |
| OCR | PP-OCRv4 + layout | 315ms | PASS | 旧本地 OCR 默认路径 |
| ASR | sherpa SenseVoice + diarization | RTF 0.17 | PASS | 旧本地 ASR 默认路径 |

## 结论

这些行支撑 K3 16G 本地底层模型架构，但在 P3 合同复测生成 v1 必需产物前仍属于 legacy evidence。

## 证据

| 详情 | 报告 |
|---|---|
| 旧完整报告 | [../../k3-riscv.en.md](../../k3-riscv.en.md) |
