# K3 RISC-V 16G A100 NPU

**Last updated:** 2026-07-14
**Chinese version:** [a100-npu.zh.md](a100-npu.zh.md)
**Legacy source:** [../../k3-riscv.en.md](../../k3-riscv.en.md)

## Scope

This page records the K3 16G A100 NPU status separately from the X100/IME2 and CPU ORT paths.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| LLM acceleration | A100 NPU offload candidate | not calibrated | PENDING | Do not use as current model-selection evidence |
| Bottom models | A100 NPU candidate path | not calibrated | PENDING | Use CPU ORT / sherpa legacy path instead |

## Decision

There is no current calibrated K3 16G A100 NPU contract path in this repository. Keep it visible as a hardware condition, but do not mark it covered.

## Evidence

| Detail | Report |
|---|---|
| Legacy full report | [../../k3-riscv.en.md](../../k3-riscv.en.md) |
