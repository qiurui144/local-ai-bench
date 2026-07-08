# Reports Index

**Last updated:** 2026-07-08
**Canonical entry point:** [Model Selection](selection/model-selection.en.md)
**Chinese version:** [index.zh.md](index.zh.md)

This directory now uses paired English and Chinese report files. New platform reports are grouped by platform and execution path, while legacy root-level reports remain as evidence and compatibility references.

## Start Here

| Need | Report |
|---|---|
| Choose models across platforms | [Model Selection](selection/model-selection.en.md) |
| Review K3 32G model/runtime decisions | [K3 RISC-V 32G](platforms/k3-riscv-32g/index.en.md) |
| Review AMD Windows CPU/iGPU/NPU paths | [AMD Windows](platforms/amd-windows/index.en.md) |
| Review Intel Windows CPU/iGPU/NPU paths | [Intel Windows](platforms/intel-windows/index.en.md) |
| Review RK3588/RK1828 NPU paths | [RK3588 + RK1828](platforms/rk3588/index.en.md) |
| Trace K3 raw evidence and run logs | [K3 Evidence Map](evidence/k3-riscv-32g.evidence.en.md) |

## Platform Reports

| Platform | Overview | Execution path reports |
|---|---|---|
| AMD Windows | [Overview](platforms/amd-windows/index.en.md) | [CPU](platforms/amd-windows/cpu.en.md), [iGPU](platforms/amd-windows/igpu.en.md), [NPU](platforms/amd-windows/npu.en.md) |
| Intel Windows | [Overview](platforms/intel-windows/index.en.md) | [CPU](platforms/intel-windows/cpu.en.md), [iGPU](platforms/intel-windows/igpu.en.md), [NPU](platforms/intel-windows/npu.en.md) |
| K3 RISC-V 32G | [Overview](platforms/k3-riscv-32g/index.en.md) | [llama.cpp / GGUF](platforms/k3-riscv-32g/llama.en.md), [ORT / SMT](platforms/k3-riscv-32g/ort.en.md), [Workflow Risk](platforms/k3-riscv-32g/workflow-risk.en.md) |
| RK3588 + RK1828 | [Overview](platforms/rk3588/index.en.md) | [RK3588 RKNPU3](platforms/rk3588/rk3588-rknpu.en.md), [RK1828 NPU](platforms/rk3588/rk1828-npu.en.md) |

## Report Contract

| Rule | Requirement |
|---|---|
| Language split | Every canonical report has a separate `.en.md` and `.zh.md` file. |
| Structure parity | English and Chinese files use the same section order and equivalent tables. |
| Hardware paths | CPU, GPU/iGPU, and NPU paths stay separated when the platform has them. |
| Evidence | New summary rows link back to run reports, legacy reports, or output evidence directories. |
| Sensitive data | Reports must not include host IPs, account names, passwords, or reusable connection strings. |
| Legacy reports | Existing root-level reports are retained until a later archive pass; new readers should start from this index. |

## Legacy Compatibility

The following legacy reports are still useful as detailed evidence, but are no longer the preferred navigation layer:

| Legacy report | New canonical route |
|---|---|
| [model-matrix.en.md](model-matrix.en.md) | [Model Selection](selection/model-selection.en.md) |
| [amd-windows.en.md](amd-windows.en.md) | [AMD Windows](platforms/amd-windows/index.en.md) |
| [intel-windows.en.md](intel-windows.en.md) | [Intel Windows](platforms/intel-windows/index.en.md) |
| [k3-riscv-32g.en.md](k3-riscv-32g.en.md) | [K3 RISC-V 32G](platforms/k3-riscv-32g/index.en.md) |
| [rk3588.en.md](rk3588.en.md) | [RK3588 + RK1828](platforms/rk3588/index.en.md) |
