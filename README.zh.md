> [English](./README.md)

# local-ai-bench

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/qiurui144/local-ai-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/qiurui144/local-ai-bench/actions/workflows/ci.yml)

**local-ai-bench** 是面向本地 AI 部署的模型选型评测平台，适用于边缘设备、Windows 笔电和嵌入式 AI-box 硬件。它回答一个问题：

> *"模型 X 能否替换生产中的模型 Y？性能够不够，效果掉不掉？"*

在目标硬件上跑完评测，然后对比：

```bash
python run_benchmark.py --model qwen2.5-7b   --seeds 3
python run_benchmark.py --model qwen3-4b     --seeds 3
python run_benchmark.py --compare qwen2.5-7b qwen3-4b
# → REPLACEABLE / NOT_REPLACEABLE / INCONCLUSIVE
```

判定结果自动生成，有数据支撑：每个质量指标在 2σ 容差内比对，单 seed 结果强制封顶 INCONCLUSIVE，性能轴和质量轴分开评估。

---

## 覆盖范围

**13 个评测维度**，分为两条轴：

| 轴 | 维度 |
|---|---|
| **性能** | TTFT · 吞吐量 · Prefill/Decode 分离 · 并发 · 稳定性 |
| **模型质量** | 准确率 · 翻译（中↔英）· Embedding · Rerank · ASR · 通用能力（GSM8K / MMLU / HellaSwag）· 条件能力曲线 · 真实产品场景 |

每个维度输出 PASS / WARN / FAIL 判定，阈值可配置。最终 exit code（`0/1/2`）可直接接入 CI。

→ 完整维度参考：[DEVELOP.md § 维度说明](DEVELOP.md)

---

## 支持平台

| 平台 | 状态 | 结果报告 |
|---|---|---|
| AMD Linux（Ryzen 8845H + 780M iGPU） | ✅ 已校准 | [amd-linux.en.md](reports/amd-linux.en.md) |
| AMD Windows（同一 Ryzen 8845H 硬件；历史 dual-boot 结果） | ✅ 已校准 | [amd-windows.en.md](reports/amd-windows.en.md) |
| Intel Windows（Core Ultra 7 155H + Arc iGPU） | ✅ 已校准 | [intel-windows.en.md](reports/intel-windows.en.md) |
| RK3588 Linux（RKNN3 NPU） | 🔧 进行中 | — |
| K3 RISC-V | 🔧 进行中 | — |
| vLLM 服务器（Linux + NVIDIA GPU） | ✅ 支持 | — |

任何 **OpenAI 兼容端点** 均可使用——vLLM、Ollama、llama.cpp server、OpenAI、DashScope、DeepSeek。

---

## 快速上手

**前置条件：** Python 3.10+，一个 OpenAI 兼容的模型端点。

```bash
git clone https://github.com/qiurui144/local-ai-bench.git
cd local-ai-bench
pip install -r requirements.txt

# 验证端点可达
python3 scripts/probe_provider.py --model <你的模型名>

# 运行评测（跳过未配置的维度）
python run_benchmark.py --model <你的模型名> --skip stability,embedding,rerank,asr

# 离线单元测试（无需 GPU）
python -m pytest tests/ -q
```

→ 模型注册、Provider 配置、Windows/Ollama 快速上手：[DEVELOP.md § 环境配置](DEVELOP.md)

---

## 模型配置

模型在 `models.yaml` 中声明。必填字段仅 `name`、`provider`、`model_id`；各维度的阈值和 skip 列表为可选覆盖。

→ 完整 `models.yaml` 字段说明与示例：[DEVELOP.md § 模型配置](DEVELOP.md)

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/index.md](docs/index.md) | 文档目录、命名规则、公开报告与本地记录边界 |
| [DEVELOP.md](DEVELOP.md) | 开发者环境配置、架构说明、维度参考、模型配置 schema、贡献指南 |
| [RELEASE.md](RELEASE.md) | 版本历史、Breaking Changes、迁移说明 |
| [reports/amd-linux.en.md](reports/amd-linux.en.md) | AMD Linux 评测结果——当前 192.168.100.201 目标机状态与模型选型建议 |
| [reports/amd-windows.en.md](reports/amd-windows.en.md) | AMD Windows 评测结果——历史 dual-boot 结果与已校准阈值 |
| [reports/intel-windows.en.md](reports/intel-windows.en.md) | Intel Windows 评测结果 |
| [docs/contributing.md](docs/contributing.md) | 如何新增模型、维度、硬件目标 |
| [docs/academic-rigor.md](docs/academic-rigor.md) | 统计严谨性原则（多 seed、效应量、校准） |

---

## 许可证

[Apache License 2.0](LICENSE)
