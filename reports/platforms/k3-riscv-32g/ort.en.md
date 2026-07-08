# K3 ORT / SMT Path

**Last updated:** 2026-07-08
**Chinese version:** [ort.zh.md](ort.zh.md)
**Sources:** [SpacemiT ORT run report](../../runs/k3-riscv-32g/20260704/k3-riscv-32g-spacemit-ort.en.md), [legacy full report](../../k3-riscv-32g.en.md)

## Scope

This path covers SpacemiT ONNX Runtime EP, TCM state management, SMT VLM/ASR media backends, official ONNX vision alignment, PP-OCRv5 OCR, and ASR.

## Official ONNX Vision

| Area | Result | Decision |
|---|---|---|
| Runtime availability | `onnxruntime_perf_test`, `spacemit-ort 2.0.3`, `spacemit-tcm` present | PASS |
| Initial failure root cause | Stale TCM blocks held by dead PIDs | Not a model/runtime capability failure |
| Remediation | `spacemit-tcm-smi -c` released TCM blocks | Required run hygiene |
| Official matrix | 33 ONNX models x 1/2/4/8 core = 132/132 PASS | Aligned with ModelZoo |
| Worst deviation | mean -1.27%, max absolute -2.60ms | No retest needed unless runtime changes |

## SMT VLM / ASR

| Model | Local result after TCM release | Official/probe relation | Decision |
|---|---|---|---|
| `Qwen3.5-2B.tar.gz` | SMT vision PASS; image OCR request 8.356s; full doc 30/30, p95 12.239s | VisionEncoder row aligned | Sync VLM default |
| `Qwen3.5-4B.tar.gz` | Runtime PASS; full doc suite 8/30 | VisionEncoder row aligned | Not default due document quality |
| `Qwen3.5-0.8B.tar.gz` | Runtime PASS; doc suite 18/30 | VisionEncoder row aligned | Partial only |
| `qwen30ba3b-mm-q4_1.tar.gz` | 29/30 docs, p95 51.188s | VisionEncoder row aligned | High-spec async VLM |
| `fastvlm-mm-0.5b-q4_1.tar.gz` | Fast runtime, quality fail | VisionEncoder row aligned | Not production VLM |
| `qwen3-asr-0.6B.tar.gz` | RTF p50/p95 0.168/0.512; normalized CER avg 0.0192 | Official-style retest still needed | ASR default |
| `qwen3-asr-1.7B-dynq-q4km.tar.gz` | RTF p50/p95 0.358/1.486; same normalized CER avg | Slower than 0.6B | Not default |

## OCR

| Model/path | Key metric | Status | Decision |
|---|---:|---|---|
| `PP-OCRv5_mobile_det+rec.onnx` | 72-line broad run p50 2372.3ms, p95 2985.5ms, CER 0.0039 | PASS line OCR | Default OCR route |
| MinerU package | package inspected, no confirmed serving wrapper | PACKAGE PASS / E2E PENDING | Keep as future document parsing work |

## Known Limits

| Limit | Impact |
|---|---|
| TCM occupancy | Stale TCM blocks can make valid ORT/SMT runs fail; always log TCM before/after. |
| ASR official alignment | Current ASR result is useful product-local data but still needs exact official-style aggregation. |
| OCR scope | PP-OCRv5 is line OCR; layout reconstruction remains product work. |
| SMT quality split | Runtime PASS is not the same as document-quality PASS. |
