# 模型选型

**最后更新：** 2026-07-08
**英文版本：** [model-selection.en.md](model-selection.en.md)
**目的：** 提供模型和平台选型的第一层可用结论。

## 范围

本报告汇总旧平台报告和运行日志证据。标记为本地数据的行，不作为官方厂商基线声明。旧报告没有覆盖的内容，优先从 `reports/runs` 和 `output/reports` 摘要中补齐。

## 选型摘要

| 角色 | 推荐模型/路径 | 平台 | 状态 | 原因 | 主要风险 | 证据 |
|---|---|---|---|---|---|---|
| K3 LLM 主模型 | `Qwen3-30B-A3B-Q4_0` | K3 32G llama.cpp | 通过但有限制 | K3 上 full-GA 覆盖最好；短请求和有界请求通过 | 长上下文同步高风险：1K realistic run 达到 223.834s 和 30.488GiB RSS | [K3 llama](../platforms/k3-riscv-32g/llama.zh.md) |
| K3 大 LLM 候选 | `Qwen3.6-35B-A3B-UD-Q4_K_XL` | K3 32G llama.cpp | LLM 通过，不是 VLM | 当前系统 runtime 可服务文本；1K/3K needle 通过 | `vision_backend=none`；不能作为单模型 LLM+VLM 方案 | [K3 llama](../platforms/k3-riscv-32g/llama.zh.md) |
| K3 单服务 LLM+VLM 候选 | `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | K3 mtmd | 通过，只适合异步 | 29/30 文档 case，字段准确率 0.9942，JSON 1.0 | 平均/p95 为 68.822/78.774s；长文本只能异步 | [K3 工作流风险](../platforms/k3-riscv-32g/workflow-risk.zh.md) |
| K3 同步 VLM 默认 | `Qwen3.5-2B.tar.gz` | K3 SMT media backend | 通过 | 30/30 文档 case；p95 12.239s | 依赖干净 TCM 状态；不能替代专用 OCR | [K3 ORT/SMT](../platforms/k3-riscv-32g/ort.zh.md) |
| K3 VLM 质量控制 | `Qwen3VL-4B + mmproj` | K3 mtmd | 通过但慢 | 30/30 文档 case；JSON 1.0 | realistic p95 可到 78.064s | [K3 llama](../platforms/k3-riscv-32g/llama.zh.md) |
| K3 OCR 默认 | `PP-OCRv5_mobile_det+rec.onnx` | K3 ORT 路径 | 行 OCR 通过 | 72 行 broad run CER 0.0039 | 完整文档版面组装仍是产品工程工作 | [K3 ORT/SMT](../platforms/k3-riscv-32g/ort.zh.md) |
| K3 ASR 默认 | `qwen3-asr-0.6B.tar.gz` | K3 SMT audio | 通过 | 与 1.7B 准确率相同，RTF/RSS 更低 | 评分需做中文繁简和数字归一化 | [K3 ORT/SMT](../platforms/k3-riscv-32g/ort.zh.md) |
| K3 embedding 默认 | `Bge-Small-Zh-V1.5-Q4_K_M` | K3 GGUF embedding | 通过 | p95 5.85ms；overall Hit@1 0.9722 | `Bge-Small-En` 在该 runtime 返回无效向量 | [K3 llama](../platforms/k3-riscv-32g/llama.zh.md) |
| K3 reranker 默认 | `Bge-Reranker-V2-M3-Q4_0` | K3 GGUF reranker | 通过 | top50 内 Hit@1 1.0 | 在线 top-k 建议 <=20 | [K3 llama](../platforms/k3-riscv-32g/llama.zh.md) |
| AMD Windows LLM | `qwen2.5-7b-amd-win` 或 `llama3.2-3b-amd-win` | Radeon 780M iGPU | 已测，有质量风险 | 7B 为 13.33 TPS；3B 为 28.99 TPS | 当前 harness 质量门禁失败；更适合性能/路径覆盖 | [AMD iGPU](../platforms/amd-windows/igpu.zh.md) |
| AMD Windows OCR | `rapidocr-amd-directml` | Radeon 780M DirectML | 通过 | p50 468.5ms，快于 CPU 和 NPU 路径 | NPU OCR 更慢，更适合批处理/热隔离 | [AMD iGPU](../platforms/amd-windows/igpu.zh.md) |
| AMD Windows reranker | `bge-reranker-base-amd-win` | CPU ONNX | 通过 | p50 78ms，nDCG/MRR 1.0 | v2-m3 慢 3.7 倍 | [AMD CPU](../platforms/amd-windows/cpu.zh.md) |
| Intel Windows CPU LLM | 交互用 `qwen2.5-3b-intel-win`，质量用 `qwen2.5-7b-intel-win` | Intel CPU | GA 通过，翻译有风险 | 3B TTFT p50 781ms；7B GA 分数更强 | 翻译门禁失败 | [Intel CPU](../platforms/intel-windows/cpu.zh.md) |
| Intel Windows iGPU/NPU OCR | OpenVINO OCR；支持静态 shape 时用 NPU PP-OCRv4 | Intel Arc / AI Boost NPU | 通过 | OpenVINO p50 797ms；NPU det+rec+cls 约 47ms | NPU 需要静态 shape 和 pipeline 集成 | [Intel NPU](../platforms/intel-windows/npu.zh.md) |
| RK1828 LLM/VLM | `qwen3-vl-2b-rk1820` | RK1828 PCIe NPU | 主要维度通过 | TTFT p50/p95 143/244ms；TPS 108.5 | runtime 上下文 768 token；conversation drift 失败 | [RK1828 NPU](../platforms/rk3588/rk1828-npu.zh.md) |
| RK3588 embedding | `minicpm-embed-rk3588` | RK3588 RKNPU3 | 通过 | hit@1/MRR/nDCG 1.0；p50 143ms | 当前部署中该路径只用于 embedding | [RK3588 RKNPU3](../platforms/rk3588/rk3588-rknpu.zh.md) |

## K3 推荐栈

| 工作流 | 推荐栈 | 结论 |
|---|---|---|
| 实时 RAG | BGE-Zh embedding -> BGE reranker top-k <=20 -> 有界 Qwen3-30B 回答 | 严格控制 context 和 token 后可用 |
| 文档 OCR | 先用 PP-OCRv5；只有视觉理解需要时再路由到 VLM | 不要用 VLM 替代 OCR |
| 同步 VLM | Qwen3.5-2B SMT | 质量/延迟最均衡 |
| 异步/高质量 VLM | Qwen3VL-4B、qwen30ba3b-mm 或外部 Qwen3.5-35B+mmproj | 需要队列、TTL、取消和可见任务状态 |
| ASR | qwen3-ASR 0.6B SMT | 默认优于 1.7B |
| 飞行手册/长文档 | 离线 text/OCR -> embedding -> reranker -> LLM 引用证据窗口 | 全手册直灌不是 K3 同步路径 |

## Windows 推荐栈

| 平台 | CPU | GPU/iGPU | NPU |
|---|---|---|---|
| AMD Windows | reranker 默认；OCR 基线 | Ollama Vulkan 做 LLM/embedding；DirectML 做 OCR | OCR 批处理/热隔离；LLM NPU 仍待验证 |
| Intel Windows | LLM CPU 默认；reranker CPU | OpenVINO iGPU 做 OCR/embedding/reranker 和实验性 LLM | 静态 shape OCR 和 Whisper encoder；embedding/reranker 动态 shape 失败 |

## RK 推荐栈

| 芯片路径 | 推荐角色 | 结论 |
|---|---|---|
| RK1828 PCIe NPU | LLM/VLM 和 ASR | `qwen3-vl-2b-rk1820` 和 `rk-asr-rk1820` 已校准 |
| RK3588 RKNPU3 | Embedding | `minicpm-embed-rk3588` 已校准 |
| RKNN3 缓存模型 | 后续 LLM/VLM/OCR 覆盖 | 46/46 artifact 已缓存，但服务加载和校准仍待完成 |

## 证据映射

| 证据类型 | 路径 |
|---|---|
| K3 原始证据映射 | [../evidence/k3-riscv-32g.evidence.zh.md](../evidence/k3-riscv-32g.evidence.zh.md) |
| K3 旧完整报告 | [../k3-riscv-32g.en.md](../k3-riscv-32g.en.md) |
| AMD 旧完整报告 | [../amd-windows.en.md](../amd-windows.en.md) |
| Intel 旧完整报告 | [../intel-windows.en.md](../intel-windows.en.md) |
| RK 旧完整报告 | [../rk3588.en.md](../rk3588.en.md) |

## 状态定义

| 状态 | 含义 |
|---|---|
| 通过 | 对声明工作负载的端到端或路径门禁通过。 |
| 通过但有限制 | 核心路径可用，但产品使用必须显式加边界。 |
| 已测，有质量风险 | 性能数据有效，但一个或多个质量门禁失败或未完成。 |
| 本地数据 | 没有官方厂商基线；只能作为项目内数据使用。 |
| 待验证 | artifact 已缓存或路径已知，但设备/服务校准未完成。 |
| 失败 | 对声明工作负载不可用。 |
