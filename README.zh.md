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
| AMD Linux（Ryzen 8845H + 780M iGPU） | ✅ 合同报告已生成，存在 verdict caveat | [AMD Linux 平台报告](reports/platforms/amd-linux/index.zh.md) |
| AMD Windows（Ryzen 8845H + Radeon 780M + XDNA NPU） | ✅ 已校准 | [AMD Windows 平台报告](reports/platforms/amd-windows/index.zh.md) |
| Intel Windows（Core Ultra 7 155H + Arc iGPU + AI Boost NPU） | ✅ 已校准 | [Intel Windows 平台报告](reports/platforms/intel-windows/index.zh.md) |
| K3 RISC-V 16G（SpacemiT K3 X100） | ✅ 旧校准完成；合同复测待执行 | [K3 RISC-V 16G 平台报告](reports/platforms/k3-riscv-16g/index.zh.md) |
| K3 RISC-V 32G（SpacemiT K3 X100） | ✅ 当前 model_zoo 范围已校准 | [K3 RISC-V 32G 平台报告](reports/platforms/k3-riscv-32g/index.zh.md) |
| RK3588 + RK1828 NPU | ✅ 主路径已校准；RKNN3 缓存完成，单模型加载待验证 | [RK 平台报告](reports/platforms/rk3588/index.zh.md) |
| Intel Linux（OpenVINO/vLLM；CPU baseline 显式标记） | ✅ 合同报告已生成，存在 verdict caveat | [Intel Linux 平台报告](reports/platforms/intel-linux/index.zh.md) |
| vLLM 服务器（Linux + NVIDIA GPU） | ✅ 支持 | — |

任何 **OpenAI 兼容端点** 均可使用——vLLM、Ollama、llama.cpp server、OpenAI、DashScope、DeepSeek。

项目运行规则：

- 同一目标机器一次只跑一个模型。不同物理目标可以并行。
- LLM/VLM CPU-only 只作为显式 CPU baseline；正常 LLM/VLM 评测必须使用目标加速路径。
- 场景 L2 judge 必须运行在单独硬件或外部服务上。目标本机单模型运行默认只跑 L1 场景。
- 主机名、账号、密码和可复用连接串不得写入报告或文档；远端目标通过本地安全环境变量配置。

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

## 报告导航

优先从整理后的报告索引进入。旧根目录报告保留为证据，但模型选型入口已经按平台和执行路径拆分。

| 需求 | 报告 |
|---|---|
| 跨平台选择模型 | [模型选型](reports/selection/model-selection.zh.md) |
| 浏览所有整理后的报告 | [报告索引](reports/index.zh.md) |
| 查看 K3 证据和运行日志来源 | [K3 证据映射](reports/evidence/k3-riscv-32g.evidence.zh.md) |
| 比较 AMD/Intel CPU、GPU/iGPU、NPU 路径 | [AMD Windows](reports/platforms/amd-windows/index.zh.md)、[Intel Windows](reports/platforms/intel-windows/index.zh.md) |

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/index.md](docs/index.md) | 文档目录、命名规则、公开报告与本地记录边界 |
| [DEVELOP.md](DEVELOP.md) | 开发者环境配置、架构说明、维度参考、模型配置 schema、贡献指南 |
| [RELEASE.md](RELEASE.md) | 版本历史、Breaking Changes、迁移说明 |
| [reports/index.zh.md](reports/index.zh.md) | 整理后的报告索引，包含中英文平台报告链接 |
| [reports/selection/model-selection.zh.md](reports/selection/model-selection.zh.md) | K3、AMD、Intel、RK 的模型选型首入口 |
| [docs/k3-realistic-stress-plan.zh.md](docs/k3-realistic-stress-plan.zh.md) | 贴近真实业务使用方式的 K3 混合流量压力测试方案 |
| [docs/k3-source-runtime-and-long-context.md](docs/k3-source-runtime-and-long-context.md) | K3 源码 runtime 等价和飞行手册长文本门禁 |
| [docs/rockchip-rknn3-model-cache.md](docs/rockchip-rknn3-model-cache.md) | RKNN3 模型缓存与同步流程 |
| [docs/spacemit-model-zoo.md](docs/spacemit-model-zoo.md) | SpacemiT model_zoo 数据获取与 K3 模型调用方式索引 |
| [docs/contributing.md](docs/contributing.md) | 如何新增模型、维度、硬件目标 |
| [docs/academic-rigor.md](docs/academic-rigor.md) | 统计严谨性原则（多 seed、效应量、校准） |

---

## 许可证

[Apache License 2.0](LICENSE)
