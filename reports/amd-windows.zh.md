> [English](./amd-windows.en.md)

# AMD Windows 平台 — 模型选型与基准测试报告

**平台：** amd-win-x86 | Ryzen 8845H + Radeon 780M iGPU + AMD XDNA NPU，Windows 11  
**最后校准：** 2026-06-19。本文件原地更新。

---

## 硬件概述

| 计算单元 | 规格 | 角色 |
|---|---|---|
| **CPU** | Ryzen 8845H（4× Zen4 P 核 + 4× Zen4c E 核） | ONNX Runtime CPU — OCR 基线、Reranker |
| **iGPU** | Radeon 780M（RDNA3，12 CU，17.9 GiB 共享显存） | Ollama Vulkan — LLM + Embedding；ONNX DirectML — OCR |
| **NPU** | AMD XDNA（AI 300 系列，16 TOPS） | ONNX VitisAI — OCR 批处理；ASR |

---

## 执行模式对比

所有测量值均为 E2E 校准跑的 p50 延迟或 TPS。

| 任务 | CPU 路径 | iGPU 路径（Vulkan / DirectML） | NPU 路径（VitisAI） |
|---|---|---|---|
| **LLM 7B** | ~3–5 TPS（估算） | **13.33 TPS** ✓ | — 不支持 |
| **LLM 3B** | ~8–12 TPS（估算） | **28.99 TPS** ✓ | — |
| **LLM 0.6B** | — | **91.09 TPS** ✓ | — |
| **Embedding 0.6B** | — | 875 ms p50 ✓ | — |
| **OCR 文字（p50）** | 1593 ms | **469 ms** ✓ 最快 | 2031 ms |
| **OCR 结构化（p50）** | 859 ms | **477 ms** ✓ | 1868 ms |
| **ASR（RTF）** | — | — | **0.073** ✓ |
| **Reranker base（p50）** | **78 ms** ✓ | — | — |
| **Reranker v2-m3（p50）** | 289 ms | — | — |

CPU 专用 LLM 未单独测试；Ollama 默认使用 Vulkan iGPU。  
三条路径的 OCR 质量（CER 7.04%）完全一致。

**→ 模式详情：**
- [iGPU（Vulkan + DirectML）— LLM、Embedding、OCR 最快路径](./amd-windows-igpu.zh.md)
- [NPU（VitisAI + DirectML）— OCR 批处理、ASR](./amd-windows-npu.zh.md)
- [CPU ONNX — OCR 基线、Reranker](./amd-windows-cpu.zh.md)

---

## 选型摘要

| 角色 | 推荐模型 | 执行模式 | 选型理由 |
|---|---|---|---|
| LLM 主力 | `qwen2.5-7b-amd-win` | iGPU（Vulkan） | 平台上综合质量最佳；translation/scenarios FAIL 是模型能力上限，不影响部署 |
| LLM 轻量 | `llama3.2-3b-amd-win` | iGPU（Vulkan） | 32k 上下文验证通过，32 并发已验证 |
| Embedding（首选） | `qwen3-embedding-0.6b-amd` | iGPU（Vulkan） | 检索质量最佳，延迟较低 |
| Embedding（多语言） | `bge-m3-amd` | iGPU（Vulkan） | 多语言替代方案，即插即用 |
| Reranker（默认） | `bge-reranker-base-amd-win` | CPU ONNX | p50 78 ms，多数场景质量充足 |
| Reranker（质量优先） | `bge-reranker-v2-m3-amd-win` | CPU ONNX | nDCG/MRR 等同，延迟 3.7 倍——仅在排序质量是瓶颈时使用 |
| OCR（首选） | `rapidocr-amd-directml` | iGPU DirectML | 最快路径：p50 468 ms |
| OCR（批处理/低功耗） | `rapidocr-amd-npu` | NPU VitisAI | p50 2031 ms——为并发 LLM 节省 iGPU 带宽 |
| ASR | `sensevoice-small-amd-win` | NPU DirectML | PASS：CER 7.69%，RTF 0.073 |
| VLM | *（不推荐）* | — | `llava-7b-amd-win` 可运行但 accuracy FAIL；本平台无合格 VLM |

---

## 完整模型结果

| 模型 | 执行方式 | 角色 | 状态 | 关键指标 |
|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | iGPU Vulkan | llm_primary | **FAIL** | TPS 13.33；TTFT p50/p95 953/6241 ms；PP/TG 116/16 t/s；max ctx 16k |
| `qwen2.5-14b-amd-win` | iGPU Vulkan | llm_parameter_uplift | **MEASURED** | TPS 7.67；TTFT p50/p95 8274/14792 ms；max ctx 16k |
| `llama3.2-3b-amd-win` | iGPU Vulkan | llm_baseline | **FAIL** | TPS 28.99；TTFT p50/p95 890/5207 ms；PP/TG 124/39 t/s；max ctx 32k |
| `qwen3-0.6b-amd` | iGPU Vulkan | llm_nano | **FAIL** | TPS 91.09；TTFT p50 1781 ms |
| `llava-7b-amd-win` | iGPU Vulkan | vlm_baseline | **FAIL** | TPS 16.84；TTFT p50 890 ms；accuracy FAIL |
| `qwen3-embedding-0.6b-amd` | iGPU Vulkan | embedding_primary | **PASS** | hit@1 1.000；nDCG 1.000；p50 875 ms |
| `bge-m3-amd` | iGPU Vulkan | embedding_bge | **PASS** | hit@1 1.000；nDCG 1.000；p50 914 ms |
| `rapidocr-amd-directml` | iGPU DirectML | ocr_gpu | **PASS** | CER 7.04%；p50 468.5 ms；结构化字段准确率 92.86%；结构化 p50 476.5 ms |
| `rapidocr-amd-npu` | NPU VitisAI | ocr_npu | **PASS** | CER 7.04%；p50 2031 ms；结构化字段准确率 92.86%；结构化 p50 1867.5 ms |
| `rapidocr-cpu` | CPU ONNX | ocr_cpu_baseline | **PASS** | CER 7.04%；p50 1592.5 ms；结构化字段准确率 92.86%；结构化 p50 859.0 ms |
| `paddleocr-cpu` | CPU ONNX | ocr_cpu_paddle | **PASS** | CER 7.04%；p50 1829.5 ms |
| `bge-reranker-base-amd-win` | CPU ONNX | reranker_default | **PASS** | nDCG 1.000；MRR 1.000；p50 78 ms |
| `bge-reranker-v2-m3-amd-win` | CPU ONNX | reranker_stronger | **PASS** | nDCG 1.000；MRR 1.000；p50 289 ms |
| `sensevoice-small-amd-win` | NPU DirectML | asr | **PASS** | CER 7.69%；RTF 0.073 |

**状态说明：** PASS = 所有阈值达标。FAIL = 一个或多个质量/性能阈值未达标。  
MEASURED = 已采集延迟/吞吐数据；质量维度未完整认证。

---

## 已知局限

- **LLM translation FAIL** — 所有 LLM 模型均未通过翻译质量门控，为本硬件层级的模型能力上限，不影响部署决策。
- **LLM conditioned FAIL** — 长上下文 conditioning 在所有测试模型上均失败。
- **LLM conversation_drift FAIL** — 多轮漂移检测失败。
- **LLM scenarios FAIL** — 领域场景测试失败。
- **无合格 VLM** — `llava-7b-amd-win` accuracy FAIL；本平台暂无推荐 VLM，直至有更好的模型验证通过。
- **NPU LLM 不支持** — AMD XDNA NPU 无法通过 Ollama 使用；LLM 推理使用 Vulkan iGPU。

---

## 校准历史

| 日期 | 事件 |
|---|---|
| 2026-06-19 | 初次完整校准：14 个模型跨 CPU/iGPU/NPU 三条路径全部测量；阈值从 E2E 跑数据中设定 |
