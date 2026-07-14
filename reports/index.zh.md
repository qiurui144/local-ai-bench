# 报告索引

**最后更新：** 2026-07-14
**标准入口：** [模型选型](selection/model-selection.zh.md)
**英文版本：** [index.en.md](index.en.md)

本目录现在采用中英文成对文件。新的平台报告按平台和执行路径分组，旧的根目录报告保留为证据和兼容引用。

## 从这里开始

| 需求 | 报告 |
|---|---|
| 跨平台选择模型 | [模型选型](selection/model-selection.zh.md) |
| 查看 AMD Linux CPU/iGPU/NPU 合同路径 | [AMD Linux](platforms/amd-linux/index.zh.md) |
| 查看 Intel Linux CPU/iGPU/NPU/OpenVINO 合同路径 | [Intel Linux](platforms/intel-linux/index.zh.md) |
| 查看 K3 16G 旧硬件路径和合同缺口 | [K3 RISC-V 16G](platforms/k3-riscv-16g/index.zh.md) |
| 查看 K3 32G 模型和运行时结论 | [K3 RISC-V 32G](platforms/k3-riscv-32g/index.zh.md) |
| 查看 AMD Windows CPU/iGPU/NPU 路径 | [AMD Windows](platforms/amd-windows/index.zh.md) |
| 查看 Intel Windows CPU/iGPU/NPU 路径 | [Intel Windows](platforms/intel-windows/index.zh.md) |
| 查看 RK3588/RK1828 NPU 路径 | [RK3588 + RK1828](platforms/rk3588/index.zh.md) |
| 追溯 K3 原始证据和运行日志 | [K3 证据映射](evidence/k3-riscv-32g.evidence.zh.md) |

## 平台报告

| 平台 | 概览 | 执行路径报告 |
|---|---|---|
| AMD Linux | [概览](platforms/amd-linux/index.zh.md) | [CPU](platforms/amd-linux/cpu.zh.md), [iGPU](platforms/amd-linux/igpu.zh.md), [NPU](platforms/amd-linux/npu.zh.md), [混合运行时](platforms/amd-linux/mixed.zh.md), [阻塞运行时](platforms/amd-linux/blocked-runtime.zh.md) |
| AMD Windows | [概览](platforms/amd-windows/index.zh.md) | [CPU](platforms/amd-windows/cpu.zh.md), [iGPU](platforms/amd-windows/igpu.zh.md), [NPU](platforms/amd-windows/npu.zh.md), [混合运行时](platforms/amd-windows/mixed.zh.md), [阻塞运行时](platforms/amd-windows/blocked-runtime.zh.md) |
| Intel Linux | [概览](platforms/intel-linux/index.zh.md) | [CPU](platforms/intel-linux/cpu.zh.md), [iGPU](platforms/intel-linux/igpu.zh.md), [NPU](platforms/intel-linux/npu.zh.md), [混合运行时](platforms/intel-linux/mixed.zh.md), [阻塞运行时](platforms/intel-linux/blocked-runtime.zh.md) |
| Intel Windows | [概览](platforms/intel-windows/index.zh.md) | [CPU](platforms/intel-windows/cpu.zh.md), [iGPU](platforms/intel-windows/igpu.zh.md), [NPU](platforms/intel-windows/npu.zh.md), [混合运行时](platforms/intel-windows/mixed.zh.md), [阻塞运行时](platforms/intel-windows/blocked-runtime.zh.md) |
| K3 RISC-V 16G | [概览](platforms/k3-riscv-16g/index.zh.md) | [X100 CPU + IME2](platforms/k3-riscv-16g/x100-ime2.zh.md), [CPU ORT / sherpa](platforms/k3-riscv-16g/cpu-ort.zh.md), [A100 NPU](platforms/k3-riscv-16g/a100-npu.zh.md) |
| K3 RISC-V 32G | [概览](platforms/k3-riscv-32g/index.zh.md) | [llama.cpp / GGUF](platforms/k3-riscv-32g/llama.zh.md), [ORT / SMT](platforms/k3-riscv-32g/ort.zh.md), [工作流风险](platforms/k3-riscv-32g/workflow-risk.zh.md), [合同补充](platforms/k3-riscv-32g/contract.zh.md) |
| RK3588 + RK1828 | [概览](platforms/rk3588/index.zh.md) | [RK3588 RKNPU3](platforms/rk3588/rk3588-rknpu.zh.md), [RK1828 NPU](platforms/rk3588/rk1828-npu.zh.md) |

## 报告约定

| 规则 | 要求 |
|---|---|
| 语言分离 | 每份标准报告都有独立的 `.en.md` 和 `.zh.md` 文件。 |
| 结构对齐 | 英文和中文文件使用相同章节顺序和等价表格。 |
| 硬件路径 | 平台存在 CPU、GPU/iGPU、NPU 时必须拆分保留。 |
| 运行时缺口 | 混合运行时和阻塞运行时行必须显式保留，不能折叠进 CPU/iGPU/NPU 页。 |
| 证据 | 新摘要行需要回链到运行报告、旧报告或 output 证据目录。 |
| 敏感信息 | 报告不得包含主机 IP、账号、密码或可复用连接串。 |
| 旧报告 | 旧根目录报告保留到后续归档批次；新读者应从本索引开始。 |

## 旧报告兼容

以下旧报告仍可作为详细证据，但不再作为首选导航层：

| 旧报告 | 新标准路径 |
|---|---|
| [model-matrix.en.md](model-matrix.en.md) | [模型选型](selection/model-selection.zh.md) |
| [amd-windows.en.md](amd-windows.en.md) | [AMD Windows](platforms/amd-windows/index.zh.md) |
| [intel-windows.en.md](intel-windows.en.md) | [Intel Windows](platforms/intel-windows/index.zh.md) |
| [k3-riscv.en.md](k3-riscv.en.md) | [K3 RISC-V 16G](platforms/k3-riscv-16g/index.zh.md) |
| [k3-riscv-32g.en.md](k3-riscv-32g.en.md) | [K3 RISC-V 32G](platforms/k3-riscv-32g/index.zh.md) |
| [rk3588.en.md](rk3588.en.md) | [RK3588 + RK1828](platforms/rk3588/index.zh.md) |
