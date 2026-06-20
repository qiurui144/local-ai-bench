> [English](./intel-windows.en.md)

# Intel Windows 平台 — 模型选型与基准测试报告

**平台：** intel-win-x86 | Intel Core Ultra 笔电，Windows 11  
**最后校准：** 2026-06-19。本文件原地更新。

---

## 硬件概述

| 计算单元 | 规格 | 角色 |
|---|---|---|
| **CPU** | Intel Core Ultra（P 核 + E 核） | Ollama CPU — LLM + Embedding；ONNX CPU — Reranker |
| **iGPU** | Intel Arc 集成显卡 | ONNX OpenVINO — OCR（PASS）；ONNX DirectML — OCR（FAIL） |
| **NPU** | Intel NPU（AI Boost） | 尚未测试 |

---

## 执行模式对比

| 任务 | CPU 路径 | iGPU / OpenVINO | NPU |
|---|---|---|---|
| **LLM 7B** | 8.25 TPS；TTFT 4820 ms | 未配置 | 未测试 |
| **LLM 3B** | 19.47 TPS；TTFT 781 ms | 未配置 | 未测试 |
| **LLM 1B** | 25.26 TPS；TTFT 875 ms | 未配置 | 未测试 |
| **Embedding 0.6B** | 617.5 ms p50 | 未配置 | — |
| **OCR 文字（p50）** | 1593 ms（参考） | 797 ms OpenVINO ✓；946 ms DirectML ✗ | 未测试 |
| **OCR 结构化（p50）** | 859 ms（参考） | 868 ms OpenVINO ✓；985 ms DirectML ✗ | 未测试 |
| **ASR（RTF）** | — | 0.341（DirectML）✓ | — |
| **Reranker base（p50）** | 148.5 ms ✓ | — | — |
| **Reranker v2-m3（p50）** | 546.5 ms ✓ | — | — |

Intel DirectML OCR **不可用**（CER 202%）。请改用 OpenVINO 路径。  
Intel iGPU LLM 加速尚未配置；所有 LLM 测试使用纯 CPU Ollama。

**→ 模式详情：**
- [CPU 模式 — LLM、Embedding、Reranker](./intel-windows-cpu.zh.md)
- [iGPU / OpenVINO / DirectML — OCR、ASR](./intel-windows-igpu.zh.md)

---

## 选型摘要

| 角色 | 推荐模型 | 执行模式 | 选型理由 |
|---|---|---|---|
| LLM 质量首选 | `qwen2.5-7b-intel-win` | CPU | 平台上最佳质量；MEASURED（延迟偏高，不适合交互） |
| LLM 日常首选 | `qwen2.5-3b-intel-win` | CPU | 轻量，8 并发已验证；TTFT 781 ms 适合交互 |
| LLM 轻量 | `llama3.2-1b-intel-win` | CPU | 32 并发，32k 上下文已验证 |
| Embedding | `qwen3-embedding-0.6b-intel-win` | CPU | PASS：hit@1 1.000，p50 617.5 ms |
| Reranker（默认） | `bge-reranker-base-intel-win` | CPU ONNX | p50 148.5 ms，多数场景质量充足 |
| Reranker（质量优先） | `bge-reranker-v2-m3-intel-win` | CPU ONNX | nDCG/MRR 等同，p50 546.5 ms — 仅排序质量是瓶颈时使用 |
| OCR | `rapidocr-intel-openvino` | iGPU OpenVINO | PASS：p50 797 ms；DirectML 不可用 |
| ASR | `sensevoice-small-intel-win` | DirectML | PASS：CER 7.69%，RTF 0.341 |
| VLM | *（不推荐）* | — | `llava-7b-intel-win` 可运行但 accuracy FAIL |

---

## 完整模型结果

| 模型 | 执行方式 | 角色 | 状态 | 关键指标 |
|---|---|---|---|---|
| `qwen2.5-7b-intel-win` | CPU（Ollama） | llm_quality | **MEASURED** | TPS 8.25；TTFT p50/p95 4820/8441 ms；PP/TG 112/9 t/s；max ctx 16k |
| `qwen2.5-3b-intel-win` | CPU（Ollama） | llm_baseline | **FAIL** | TPS 19.47；TTFT p50/p95 781/3495 ms；PP/TG 124/26 t/s；max ctx 16k |
| `llama3.2-1b-intel-win` | CPU（Ollama） | llm_nano | **FAIL** | TPS 25.26；TTFT p50/p95 875/3308 ms；PP/TG 130/35 t/s；max ctx 32k |
| `llava-7b-intel-win` | CPU（Ollama） | vlm_baseline | **FAIL** | TPS 10.02；TTFT p50 703 ms；accuracy FAIL |
| `qwen3-embedding-0.6b-intel-win` | CPU（Ollama） | embedding | **PASS** | hit@1 1.000；nDCG 1.000；p50 617.5 ms |
| `bge-reranker-base-intel-win` | CPU ONNX | reranker_default | **PASS** | nDCG 1.000；MRR 1.000；p50 148.5 ms |
| `bge-reranker-v2-m3-intel-win` | CPU ONNX | reranker_stronger | **PASS** | nDCG 1.000；MRR 1.000；p50 546.5 ms |
| `rapidocr-intel-openvino` | iGPU OpenVINO | ocr_openvino | **PASS** | CER 7.04%；p50 797 ms；结构化字段准确率 92.86%；结构化 p50 867.5 ms |
| `rapidocr-intel-directml` | iGPU DirectML | ocr_directml | **FAIL** | CER 202.35% — 不可用 |
| `sensevoice-small-intel-win` | DirectML | asr | **PASS** | CER 7.69%；RTF 0.341 |

**状态说明：** PASS = 所有阈值达标。FAIL = 一个或多个阈值未达标。MEASURED = 已采集延迟/吞吐；质量维度未完整认证。

---

## 已知局限

- **general_ability BLOCKED** — 目标机未安装 `datasets` 库；需解决后才能运行 general_ability 和 conditioned 维度。
- **conditioned BLOCKED** — 与 general_ability 同一根因。
- **Intel DirectML OCR 不可用** — `rapidocr-intel-directml` CER 202.35%；Intel iGPU 上 DirectML FP16 精度问题。请使用 OpenVINO 路径。
- **无合格 VLM** — `llava-7b-intel-win` accuracy FAIL。
- **LLM TTFT 偏高（7B）** — `qwen2.5-7b-intel-win` p50 TTFT 4820 ms 由纯 CPU 预填充驱动；交互场景首选 `qwen2.5-3b-intel-win`。
- **iGPU LLM 未测试** — Intel iGPU LLM 加速（通过 OpenVINO 或 IPEX）尚未配置。

---

## 校准历史

| 日期 | 事件 |
|---|---|
| 2026-06-19 | 初次完整校准：10 个模型全部测量；CPU LLM、OpenVINO OCR、DirectML ASR 已校准；general_ability/conditioned 因 datasets 未安装而 BLOCKED |
