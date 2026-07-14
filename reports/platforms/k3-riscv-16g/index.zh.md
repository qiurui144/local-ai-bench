# K3 RISC-V 16G

**最后更新：** 2026-07-14
**英文版本：** [index.en.md](index.en.md)
**旧报告来源：** [../../k3-riscv.en.md](../../k3-riscv.en.md)

## 范围

本页把 K3 16GB 的旧校准证据整理到标准 reports 结构中。NAS 合同把 `k3-riscv-16g` 列为 P3 复测目标，但当前仓库没有该目标的 v1 `parameter-matrix.json` / `run-summary.json` 产物。

因此，本页是 legacy calibrated evidence，不是当前 NAS contract product verdict。

## 合同基线

| 项目 | 值 |
|---|---|
| target | `k3-riscv-16g` |
| contract_artifacts | 当前仓库无 |
| report_status | `legacy_calibrated_contract_retest_pending` |
| legacy_source | [../../k3-riscv.en.md](../../k3-riscv.en.md) |

## 硬件路径摘要

| 路径 | Runtime | 工作负载 | 状态 |
|---|---|---|---|
| [X100 CPU + IME2](x100-ime2.zh.md) | llama.cpp / llama-server v8355 | LLM chat、translation/GA probe | 旧校准通过；合同复测待执行 |
| [CPU ORT / sherpa](cpu-ort.zh.md) | ONNX Runtime / sherpa-onnx | embedding、reranker、OCR、ASR | 旧校准通过；合同复测待执行 |
| [A100 NPU](a100-npu.zh.md) | A100 NPU offload | 候选加速路径 | 当前无合同校准路径 |

## 结论

K3 16G 仍是 Qwen2.5 3B/7B 级本地 LLM 和 OCR/ASR/retrieval 底层模型的旧校准平台。但在 P3 合同复测生成必需产物前，不能写成 NAS contract complete。

## 证据

| 详情 | 报告 |
|---|---|
| X100 CPU + IME2 路径 | [x100-ime2.zh.md](x100-ime2.zh.md) |
| CPU ORT / sherpa 路径 | [cpu-ort.zh.md](cpu-ort.zh.md) |
| A100 NPU 路径 | [a100-npu.zh.md](a100-npu.zh.md) |
| 旧完整报告 | [../../k3-riscv.en.md](../../k3-riscv.en.md) |
