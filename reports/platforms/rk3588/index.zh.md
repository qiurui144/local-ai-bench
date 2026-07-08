# RK3588 + RK1828

**最后更新：** 2026-07-08
**英文版本：** [index.en.md](index.en.md)
**旧报告来源：** [../../rk3588.en.md](../../rk3588.en.md)

## 范围

该平台有两条独立 NPU 路径：RK3588 片上 RKNPU3 用于 embedding，RK1828 PCIe NPU 用于 LLM/VLM/ASR。历史模型 ID 使用 `rk1820`，但实测设备是 RK1828。

## 执行路径摘要

| 路径 | Runtime/service | 最适合工作负载 | 状态 |
|---|---|---|---|
| [RK3588 RKNPU3](rk3588-rknpu.zh.md) | uvicorn embedding service | Embedding | 通过 |
| [RK1828 NPU](rk1828-npu.zh.md) | rkllm3 / ASR services | LLM/VLM 和 ASR | 主要行通过；有上下文风险 |

## 选型说明

| 角色 | 当前选择 | 结论 |
|---|---|---|
| LLM/VLM | `qwen3-vl-2b-rk1820` | TTFT/吞吐/翻译通过；conversation drift 因 768-token runtime 限制失败。 |
| ASR | `rk-asr-rk1820` | 3-seed ASR 通过；zh CER 被字符集差异放大。 |
| Embedding | `minicpm-embed-rk3588` | hit@1/MRR/nDCG 1.0；p50 143ms。 |
| RKNN3 缓存模型 | 46/46 artifacts 已缓存 | 已登记，但仍待服务加载和 harness 校准。 |

## 证据

| 详情 | 报告 |
|---|---|
| RK3588 RKNPU3 | [rk3588-rknpu.zh.md](rk3588-rknpu.zh.md) |
| RK1828 NPU | [rk1828-npu.zh.md](rk1828-npu.zh.md) |
| 旧完整报告 | [../../rk3588.en.md](../../rk3588.en.md) |
