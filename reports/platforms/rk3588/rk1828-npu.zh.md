# RK1828 NPU 路径

**最后更新：** 2026-07-08
**英文版本：** [rk1828-npu.en.md](rk1828-npu.en.md)
**旧报告来源：** [../../rk3588.en.md](../../rk3588.en.md)

## 范围

RK1828 PCIe NPU 路径运行 LLM/VLM 和 ASR 服务。模型 ID 保留历史 `rk1820` 后缀，但实测设备是 RK1828。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| LLM/VLM | `qwen3-vl-2b-rk1820` | TTFT p50/p95 143/244ms；TPS 108.5 | 主要维度通过 | 默认 RK NPU LLM/VLM |
| 翻译 | `qwen3-vl-2b-rk1820` | 所有测试方向通过 | 通过 | 在上下文限制内可用 |
| Conversation drift | `qwen3-vl-2b-rk1820` | 结构化提取最大下降 21.19% | 失败 | 768-token runtime 限制风险 |
| ASR | `rk-asr-rk1820` | RTF 0.0768 +/- 0.0045，CER 4.73% | 通过 | 默认 RK ASR |
| RKNN3 缓存 | v1.0.4 LLM/VLM/OCR 相关 artifacts | 46/46 已缓存 | 待验证 | 需要服务加载和 harness 校准 |

## RKNN3 缓存覆盖

| 分组 | 覆盖 | 状态 |
|---|---|---|
| LLM | Qwen2.5、Qwen3、Copaw Flash 行已登记 | 已缓存，待验证 |
| VLM | FastVLM、InternVL、Janus、MiniCPM、Qwen2.5-VL、Qwen3-VL 等 | 已缓存，待验证 |
| OCR/VLM | `paddleocr-vl-rk1820` | 已缓存，待验证 |

## 结论

RK 侧 LLM/VLM/ASR 决策使用 RK1828。主要 blocker 不是吞吐，而是当前 RKLLM 上下文限制和长对话 drift 行为。
