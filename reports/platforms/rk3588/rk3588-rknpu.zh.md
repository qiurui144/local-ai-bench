# RK3588 RKNPU3 路径

**最后更新：** 2026-07-08
**英文版本：** [rk3588-rknpu.en.md](rk3588-rknpu.en.md)
**旧报告来源：** [../../rk3588.en.md](../../rk3588.en.md)

## 范围

当前部署中，RK3588 片上 RKNPU3 路径用于 embedding。LLM/VLM/ASR 由 RK1828 PCIe NPU 路径服务，不在该路径上。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| Embedding | `minicpm-embed-rk3588` | hit@1 1.0，MRR 1.0，nDCG@10 1.0，p50 143ms | 通过 | 默认 RK embedding 路径 |
| LLM/VLM | RK3588 RKNPU3 | 未分配 | 不适用 | 使用 RK1828 NPU 路径 |
| ASR/TTS | RK3588 RKNPU3 | 未分配 | 不适用 | 使用 RK1828 NPU 路径 |

## 结论

在选型报告中，RK3588 RKNPU3 保持为 embedding-only 路径。不要把它的指标与 RK1828 LLM/VLM/ASR 结果混写。
