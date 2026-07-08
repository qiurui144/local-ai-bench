# K3 RISC-V 32G 证据映射

**最后更新：** 2026-07-08
**英文版本：** [k3-riscv-32g.evidence.en.md](k3-riscv-32g.evidence.en.md)

## 范围

本文件将 K3 32G 报告结论映射到旧报告、运行报告和 output 证据目录。它也是从运行日志补齐缺失报告章节的来源说明。

## 标准报告

| 标准报告 | 目的 |
|---|---|
| [../platforms/k3-riscv-32g/index.zh.md](../platforms/k3-riscv-32g/index.zh.md) | K3 平台级摘要 |
| [../platforms/k3-riscv-32g/llama.zh.md](../platforms/k3-riscv-32g/llama.zh.md) | GGUF、llama.cpp、mtmd、embedding、reranker |
| [../platforms/k3-riscv-32g/ort.zh.md](../platforms/k3-riscv-32g/ort.zh.md) | ORT、TCM、SMT VLM/ASR、OCR |
| [../platforms/k3-riscv-32g/workflow-risk.zh.md](../platforms/k3-riscv-32g/workflow-risk.zh.md) | 产品工作流风险 |

## 来自运行日志的章节

| 报告章节 | 主要证据 | 说明 |
|---|---|---|
| LLM 全量模型矩阵 | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-llm-full-20260704.md` | 从运行日志补齐高参数到低参数 LLM 覆盖。 |
| VLM 全量模型矩阵 | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-vlm-full-20260704.md` 加后续旧报告证据 | 早期 TCM 失败由 ORT 路径中释放 TCM 后的 SMT 复测覆盖。 |
| 非 LLM 全量覆盖 | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-nonllm-full-20260704.md` | Embedding、reranker、ASR package、OCR package 覆盖。 |
| 高规格非 LLM | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-nonllm-highspec-20260704.md` | Jina embedding、BGE reranker、ASR/MinerU package 检查。 |
| Qwen3-30B 专项 | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-qwen30b-20260704.md` | 内存/context 风险和 no-think 行为。 |
| Qwen3.6 loader/runtime 分析 | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-qwen36-20260704.md` 和 `k3-riscv-32g-qwen36-spacemit-private-20260704.md` | 解释 private loader 问题和 upstream fallback。 |
| ORT 官方 vision 与 SMT 修复 | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-spacemit-ort.en.md` | 132/132 官方 ONNX vision 对齐和 TCM 修复来源。 |
| CPU/RVV/IME 路径 | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-cpu.en.md` | 路径拆分和 GGUF 模型覆盖来源。 |

## Output 证据目录

| 结论领域 | 证据路径 |
|---|---|
| 官方 LLM ModelZoo 复测 | `output/reports/k3-riscv-32g/official-modelzoo-llm-20260706_185656/` |
| 官方 VLM VisionEncoder probe | `output/reports/k3-riscv-32g/official-modelzoo-vlm-encoder-20260706_193214/` |
| 官方 ONNX vision 矩阵 | `output/reports/k3-riscv-32g/vision-official-20260704_195025/results.tsv` |
| VLM 文档抽取 | `output/reports/k3-riscv-32g/vlm-full-20260706_0955_allcases/` |
| 非 LLM 宽覆盖 | `output/reports/k3-riscv-32g/nonllm-broad-20260706_190649/` |
| realistic 工作流控制 | `output/reports/k3-riscv-32g/realistic-stress-combined-20260706_150439/` |
| 源码构建 llama.cpp 等价 | `output/reports/k3-riscv-32g/source-runtime-compare-20260707_101930/` |
| 源码构建 ORT 等价 | `output/reports/k3-riscv-32g/source-runtime-compare-20260707_111859/` |
| Qwen3.5-35B 外部 mmproj 文档 run | `output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-docfull-defaultimg-official-20260708_093315/` |
| 飞行手册缓存 | `drivers/long-context-suites/airplane-manual-collection/cases/aviation_manual_cases.jsonl` |
| 飞行手册 1K run | `output/reports/k3-riscv-32g/long-context-aviation-qwen3-4b-1k-20260707_113324/` |
| 飞行手册 0.6B safe-window run | `output/reports/k3-riscv-32g/long-context-aviation-qwen3-06b-1k-safe-20260707_120251/` |

## 对齐说明

| 领域 | 对齐结论 |
|---|---|
| LLM 官方基线 | ModelZoo 基线声明只使用 TCM enabled 官方 wrapper run。 |
| ORT 官方基线 | TCM 清理后 132/132 行对齐；stale TCM 是环境问题。 |
| VLM encoder 基线 | encoder 延迟对齐，但端到端 VLM 文档质量是独立产品指标。 |
| OCR / embedding / reranker | 本地数据；引用的 ModelZoo 页面没有可比较官方行。 |
| ASR | 产品本地数据可用；精确官方聚合仍需复测。 |

## 敏感信息规则

证据路径和报告必须使用连接信息占位符。不要把主机 IP、账号、密码或可复用连接串写入报告文件。
