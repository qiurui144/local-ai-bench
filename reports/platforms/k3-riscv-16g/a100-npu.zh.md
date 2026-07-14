# K3 RISC-V 16G A100 NPU

**最后更新：** 2026-07-14
**英文版本：** [a100-npu.en.md](a100-npu.en.md)
**旧报告来源：** [../../k3-riscv.en.md](../../k3-riscv.en.md)

## 范围

本页把 K3 16G A100 NPU 状态从 X100/IME2 和 CPU ORT 路径中独立出来。

## 工作负载结果

| 工作负载 | 模型/路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| LLM acceleration | A100 NPU offload candidate | 未校准 | PENDING | 不作为当前模型选型证据 |
| Bottom models | A100 NPU candidate path | 未校准 | PENDING | 当前使用 CPU ORT / sherpa 旧路径 |

## 结论

当前仓库没有 K3 16G A100 NPU 的合同校准路径。该硬件条件需要显式保留，但不能标记为已覆盖。

## 证据

| 详情 | 报告 |
|---|---|
| 旧完整报告 | [../../k3-riscv.en.md](../../k3-riscv.en.md) |
