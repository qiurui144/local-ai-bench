# K3 RISC-V 32G Evidence Map

**Last updated:** 2026-07-08
**Chinese version:** [k3-riscv-32g.evidence.zh.md](k3-riscv-32g.evidence.zh.md)

## Scope

This file maps K3 32G report claims to legacy reports, run reports, and output evidence directories. It is also the source of truth for which missing report sections were reconstructed from run logs.

## Canonical Reports

| Canonical report | Purpose |
|---|---|
| [../platforms/k3-riscv-32g/index.en.md](../platforms/k3-riscv-32g/index.en.md) | Platform-level K3 summary |
| [../platforms/k3-riscv-32g/llama.en.md](../platforms/k3-riscv-32g/llama.en.md) | GGUF, llama.cpp, mtmd, embedding, reranker |
| [../platforms/k3-riscv-32g/ort.en.md](../platforms/k3-riscv-32g/ort.en.md) | ORT, TCM, SMT VLM/ASR, OCR |
| [../platforms/k3-riscv-32g/workflow-risk.en.md](../platforms/k3-riscv-32g/workflow-risk.en.md) | Product workflow risk |

## Run-Log Derived Sections

| Report section | Primary evidence | Notes |
|---|---|---|
| LLM full model matrix | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-llm-full-20260704.md` | Filled high-to-low parameter LLM coverage from run log. |
| VLM full model matrix | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-vlm-full-20260704.md` plus later legacy report evidence | Early TCM failures were superseded by after-release SMT retests in the ORT path. |
| Non-LLM full coverage | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-nonllm-full-20260704.md` | Embedding, reranker, ASR package, OCR package coverage. |
| High-spec non-LLM | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-nonllm-highspec-20260704.md` | Jina embedding, BGE reranker, ASR/MinerU package inspection. |
| Qwen3-30B focused run | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-qwen30b-20260704.md` | Memory/context risk and no-think behavior. |
| Qwen3.6 loader/runtime analysis | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-qwen36-20260704.md` and `k3-riscv-32g-qwen36-spacemit-private-20260704.md` | Explains private loader issue and upstream fallback. |
| ORT official vision and SMT fix | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-spacemit-ort.en.md` | Source for 132/132 official ONNX vision alignment and TCM remediation. |
| CPU/RVV/IME path | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-cpu.en.md` | Source for path split and GGUF model coverage. |

## Output Evidence Directories

| Claim area | Evidence path |
|---|---|
| Official LLM ModelZoo retest | `output/reports/k3-riscv-32g/official-modelzoo-llm-20260706_185656/` |
| Official VLM VisionEncoder probe | `output/reports/k3-riscv-32g/official-modelzoo-vlm-encoder-20260706_193214/` |
| Official ONNX vision matrix | `output/reports/k3-riscv-32g/vision-official-20260704_195025/results.tsv` |
| VLM document extraction | `output/reports/k3-riscv-32g/vlm-full-20260706_0955_allcases/` |
| Non-LLM broad coverage | `output/reports/k3-riscv-32g/nonllm-broad-20260706_190649/` |
| Realistic workflow control | `output/reports/k3-riscv-32g/realistic-stress-combined-20260706_150439/` |
| Source-built llama.cpp equivalence | `output/reports/k3-riscv-32g/source-runtime-compare-20260707_101930/` |
| Source-built ORT equivalence | `output/reports/k3-riscv-32g/source-runtime-compare-20260707_111859/` |
| Qwen3.5-35B external mmproj document run | `output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-docfull-defaultimg-official-20260708_093315/` |
| Aviation manual cache | `drivers/long-context-suites/airplane-manual-collection/cases/aviation_manual_cases.jsonl` |
| Aviation 1K run | `output/reports/k3-riscv-32g/long-context-aviation-qwen3-4b-1k-20260707_113324/` |
| Aviation 0.6B safe-window run | `output/reports/k3-riscv-32g/long-context-aviation-qwen3-06b-1k-safe-20260707_120251/` |

## Alignment Notes

| Area | Alignment decision |
|---|---|
| LLM official baseline | Use only the TCM-enabled official wrapper run for ModelZoo baseline claims. |
| ORT official baseline | 132/132 rows aligned after TCM cleanup; stale TCM is an environment issue. |
| VLM encoder baseline | Encoder latency aligns, but end-to-end VLM document quality is a separate product metric. |
| OCR / embedding / reranker | Local-only because the cited ModelZoo page does not publish comparable rows. |
| ASR | Product-local data is usable; exact official aggregation still requires retest. |

## Sensitive Data Rule

Evidence paths and reports must use placeholders for connection information. Do not place host IPs, account names, passwords, or reusable connection strings in report files.
