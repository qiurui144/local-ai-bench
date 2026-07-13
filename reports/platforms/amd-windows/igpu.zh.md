# AMD Windows iGPU 路径

**最后更新：** 2026-07-09
**英文版本：** [igpu.en.md](igpu.en.md)

## 范围

本页只记录 AMD Windows 合同基线中的 iGPU/DirectML 相关路径，不做不同软件栈之间的横向比较。LLM/VLM 走 Ollama AMD service path，embedding/reranker/OCR 走 DirectML 相关路径。

## 工作负载结果

| 工作负载 | 模型/路径 | p95 latency | 质量分 | Verdict |
|---|---|---:|---:|---|
| LLM chat | `qwen2.5-7b-amd-win` | 6312.8ms | - | `not_recommended` |
| LLM summary | `qwen2.5-7b-amd-win` | 6312.8ms | - | `not_recommended` |
| RAG answer | `qwen2.5-7b-amd-win` | 6312.8ms | - | `not_recommended` |
| VLM image QA | `llava-7b-amd-win` | 13666.2ms | 0.8889 | `not_recommended` |
| VLM document extract | `llava-7b-amd-win` | 13666.2ms | 0.0667 | `not_recommended` |
| Embedding retrieval | `bge-base-en-v1.5-igpu-amd-win` | 2453.0ms | 0.9866 | `sync_default` |
| RAG search-only fallback | `bge-base-en-v1.5-igpu-amd-win` | 2453.0ms | 0.9866 | `sync_bounded` |
| Reranker candidates | `bge-reranker-base-igpu-amd-win` | 4527.7ms | 1.0000 | `sync_default` |
| OCR pages | `rapidocr-amd-directml` | 518.1ms | 0.9296 | `sync_default` |

ASR `sensevoice-small-amd-win` 属于本次 AMD Windows 合同基线，p95 latency 437.0ms，质量分 0.9231，verdict 为 `sync_default`。

## 结论

iGPU/DirectML 路径中，embedding、reranker、OCR 已达到同步默认或同步有界能力。LLM 和 VLM 行完成了合同输出，但由于质量门槛未通过，当前不能作为默认产品能力，只能保留为后续 prompt、模型或数据集修正后的重测对象。

最终合同产物见 [nas-contract-report.md](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/nas-contract-report.md)，完整机器可读矩阵见 [parameter-matrix.json](../../../output/reports/windows-full-matrix/amd-20260709-baseline-contract-s1-final-contract/parameter-matrix.json)。
