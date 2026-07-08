# K3 RISC-V 32G

**最后更新：** 2026-07-08
**英文版本：** [index.en.md](index.en.md)
**旧报告来源：** [../../k3-riscv-32g.en.md](../../k3-riscv-32g.en.md)

## 范围

本平台报告覆盖 SpacemiT K3 X100 32GB / Bianbu Linux。它将 GGUF/llama.cpp 模型服务、ORT/SMT 媒体路径、产品工作流风险分开。早期摘要没有覆盖的缺口，来自 `reports/runs/k3-riscv-32g/20260704` 以及旧完整报告中引用的后续 `output/reports/k3-riscv-32g` 证据。

## 执行路径摘要

| 路径 | Runtime | 最适合工作负载 | 状态 |
|---|---|---|---|
| [llama.cpp / GGUF](llama.zh.md) | SpacemiT private llama.cpp、源码构建 llama.cpp、mtmd | LLM、GGUF+mmproj VLM、embedding、reranker | 通过但有模型级限制 |
| [ORT / SMT](ort.zh.md) | SpacemiT ORT EP、SMT media backend | 官方 ONNX vision、VLM tar、ASR tar、PP-OCRv5 | 控制 TCM 状态后通过 |
| [工作流风险](workflow-risk.zh.md) | 产品工作流层 | RAG、文档 OCR/VLM、ASR、飞行手册长文本、压力控制 | 需要准入/队列控制 |

## 选型说明

| 角色 | 当前选择 | 结论 |
|---|---|---|
| LLM 主模型 | `Qwen3-30B-A3B-Q4_0` | K3 上资格最完整的 LLM，但长上下文同步使用高风险。 |
| 大 LLM 候选 | `Qwen3.6-35B-A3B-UD-Q4_K_XL` | LLM 路径通过当前 probe；图片输入不支持。 |
| 单服务 LLM+VLM 候选 | 外部 `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | 质量通过，但因为延迟只能异步。 |
| 同步 VLM | `Qwen3.5-2B.tar.gz` | 最实用文档 VLM：30/30 case，p95 12.239s。 |
| 质量 VLM | `Qwen3VL-4B + mmproj` | 30/30 质量控制；太慢，不适合作默认同步。 |
| OCR | `PP-OCRv5_mobile_det+rec.onnx` | 行 OCR 通过；先 OCR 后 VLM。 |
| ASR | `qwen3-asr-0.6B.tar.gz` | 同等 normalized CER 下 RTF/RSS 更低，优于 1.7B 作为默认。 |
| Embedding | `Bge-Small-Zh-V1.5-Q4_K_M` | 默认：p95 5.85ms；`Bge-Small-En` 无效向量失败。 |
| Reranker | `Bge-Reranker-V2-M3-Q4_0` | 默认；在线 top-k 建议 <=20。 |

## 官方对齐

| 领域 | 对齐情况 | 证据 |
|---|---|---|
| 官方 ONNX vision | 132/132 对齐 | [ORT / SMT](ort.zh.md) |
| 官方 LLM ModelZoo 行 | TCM enabled 下 8/8 对齐 | [llama.cpp / GGUF](llama.zh.md) |
| VLM VisionEncoder 行 | 10/10 对齐 probe | [ORT / SMT](ort.zh.md) |
| ASR | partial / 需要官方口径聚合复测 | [ORT / SMT](ort.zh.md) |
| OCR / embedding / reranker | 本地数据；引用的 ModelZoo 页面没有官方行 | [证据映射](../../evidence/k3-riscv-32g.evidence.zh.md) |

## 证据

| 详情 | 报告 |
|---|---|
| llama.cpp / GGUF 路径 | [llama.zh.md](llama.zh.md) |
| ORT / SMT 路径 | [ort.zh.md](ort.zh.md) |
| 工作流风险 | [workflow-risk.zh.md](workflow-risk.zh.md) |
| 原始证据映射 | [../../evidence/k3-riscv-32g.evidence.zh.md](../../evidence/k3-riscv-32g.evidence.zh.md) |
| 旧完整报告 | [../../k3-riscv-32g.en.md](../../k3-riscv-32g.en.md) |
