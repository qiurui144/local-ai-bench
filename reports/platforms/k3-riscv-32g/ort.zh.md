# K3 ORT / SMT 路径

**最后更新：** 2026-07-08
**英文版本：** [ort.en.md](ort.en.md)
**来源：** [SpacemiT ORT 运行报告](../../runs/k3-riscv-32g/20260704/k3-riscv-32g-spacemit-ort.en.md)、[旧完整报告](../../k3-riscv-32g.en.md)

## 范围

本路径覆盖 SpacemiT ONNX Runtime EP、TCM 状态管理、SMT VLM/ASR media backend、官方 ONNX vision 对齐、PP-OCRv5 OCR 和 ASR。

## 官方 ONNX Vision

| 领域 | 结果 | 结论 |
|---|---|---|
| Runtime 可用性 | `onnxruntime_perf_test`、`spacemit-ort 2.0.3`、`spacemit-tcm` 存在 | 通过 |
| 初始失败根因 | 死亡 PID 占用 stale TCM blocks | 不是模型/runtime 能力失败 |
| 修复 | `spacemit-tcm-smi -c` 释放 TCM blocks | 必须作为运行卫生步骤 |
| 官方矩阵 | 33 个 ONNX 模型 x 1/2/4/8 core = 132/132 通过 | 与 ModelZoo 对齐 |
| 最差偏差 | mean -1.27%，max absolute -2.60ms | runtime 不变时无需重测 |

## SMT VLM / ASR

| 模型 | TCM 释放后本地结果 | 官方/probe 关系 | 结论 |
|---|---|---|---|
| `Qwen3.5-2B.tar.gz` | SMT vision 通过；图片 OCR 请求 8.356s；完整文档 30/30，p95 12.239s | VisionEncoder 行对齐 | 同步 VLM 默认 |
| `Qwen3.5-4B.tar.gz` | Runtime 通过；完整文档 suite 8/30 | VisionEncoder 行对齐 | 文档质量不足，不默认 |
| `Qwen3.5-0.8B.tar.gz` | Runtime 通过；文档 suite 18/30 | VisionEncoder 行对齐 | 仅 partial |
| `qwen30ba3b-mm-q4_1.tar.gz` | 29/30 文档，p95 51.188s | VisionEncoder 行对齐 | 高规格异步 VLM |
| `fastvlm-mm-0.5b-q4_1.tar.gz` | runtime 快，质量失败 | VisionEncoder 行对齐 | 不作生产 VLM |
| `qwen3-asr-0.6B.tar.gz` | RTF p50/p95 0.168/0.512；normalized CER avg 0.0192 | 仍需官方口径复测 | ASR 默认 |
| `qwen3-asr-1.7B-dynq-q4km.tar.gz` | RTF p50/p95 0.358/1.486；normalized CER 相同 | 慢于 0.6B | 不默认 |

## OCR

| 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---:|---|---|
| `PP-OCRv5_mobile_det+rec.onnx` | 72 行 broad run p50 2372.3ms，p95 2985.5ms，CER 0.0039 | 行 OCR 通过 | 默认 OCR 路径 |
| MinerU package | package 已检查，无确认 serving wrapper | 包通过 / E2E 待补 | 保留为后续文档解析工作 |

## 已知限制

| 限制 | 影响 |
|---|---|
| TCM 占用 | stale TCM blocks 会让有效 ORT/SMT 运行失败；每次前后都要记录 TCM。 |
| ASR 官方对齐 | 当前 ASR 结果是有用的产品本地数据，但仍需精确官方口径聚合。 |
| OCR 范围 | PP-OCRv5 是行 OCR；版面重建仍是产品工作。 |
| SMT 质量拆分 | Runtime 通过不等于文档质量通过。 |
