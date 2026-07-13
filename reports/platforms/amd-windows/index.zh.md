# AMD Windows

**最后更新：** 2026-07-09
**英文版本：** [index.en.md](index.en.md)

## 范围

本页记录 AMD Windows 的合同基线测试结果。此次只按同一软件栈对齐 NAS 合同测项，不做 llama.cpp、Ollama、CPU-only 等跨软件栈横向比较。

目标设备为 Windows 11 / Ryzen 7 8845H / Radeon 780M / 27.75GB RAM。测试覆盖 LLM、RAG、embedding、reranker、VLM、OCR、ASR 共 10 个合同测项。

## 合同基线

| 项目 | 值 |
|---|---|
| target | `amd-win-x86` |
| run_id | `amd-20260709-baseline-contract-s1-final` |
| status | `complete` |
| row_count | 10 |
| blocked_test_items | 0 |
| sync_default | 4 |
| sync_bounded | 1 |
| not_recommended | 5 |

## 执行路径摘要

| 工作负载 | 基线模型/路径 | Runtime | Verdict |
|---|---|---|---|
| LLM / RAG answer | `qwen2.5-7b-amd-win` | Ollama AMD/Vulkan service path | `not_recommended` |
| VLM | `llava-7b-amd-win` | Ollama AMD/VLM service path | `not_recommended` |
| Embedding / RAG search-only | `bge-base-en-v1.5-igpu-amd-win` | ONNX Runtime DirectML service path | `sync_default` / `sync_bounded` |
| Reranker | `bge-reranker-base-igpu-amd-win` | ONNX Runtime DirectML service path | `sync_default` |
| OCR | `rapidocr-amd-directml` | DirectML OCR path | `sync_default` |
| ASR | `sensevoice-small-amd-win` | AMD Windows ASR path | `sync_default` |

## 结论

AMD Windows 当前可作为 NAS 产品基线的同步默认路径是 OCR、ASR、embedding 和 reranker。RAG search-only 可作为 bounded sync fallback。

`qwen2.5-7b-amd-win` 的 LLM/RAG answer 行与 `llava-7b-amd-win` 的 VLM 行已有性能和质量输出，但合同 verdict 为 `not_recommended`，原因是质量验证未达当前产品化门槛，不能作为默认同步能力发布。

## 证据

| 产物 | 路径 |
|---|---|
| 合同总报告 | [nas-contract-report.md](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/nas-contract-report.md) |
| 参数矩阵 | [parameter-matrix.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/parameter-matrix.json) |
| 运行摘要 | [run-summary.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/run-summary.json) |
| verdict 表 | [verdict-table.tsv](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/verdict-table.tsv) |
| 模型画像 | [model-profile.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/model-profile.json) |
| scheduler 合同 | [scheduler-contract.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/scheduler-contract.json) |
| 主测 summary | [amd-20260709-baseline-contract-s1-capped_summary.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-capped_summary.json) |
| VLM 补测 summary | [amd-20260709-baseline-contract-s1-capped-vlm-scenarios_summary.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-capped-vlm-scenarios_summary.json) |
