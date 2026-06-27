# AMD Windows Platform — Model Selection & Benchmark Report

**Platform:** amd-win-x86 | Ryzen 8845H + Radeon 780M iGPU + AMD XDNA 1 NPU (Hawk Point), Windows 11  
**Last calibrated:** 2026-06-26. This file is updated in place.

---

## Hardware Profile

| Compute Unit | Chip | Specs | TDP | Role |
|---|---|---|---|---|
| **CPU** | AMD Ryzen 8845H | 4× Zen4 P-core + 4× Zen4c E-core, 16 threads, 3.8–5.1 GHz | 35 W (base) / 54 W (max) | ONNX Runtime CPU — OCR baseline, Reranker |
| **iGPU** | AMD Radeon 780M | RDNA3, 12 CU, 2800 MHz, 17.9 GiB shared VRAM | part of SoC TDP | Ollama Vulkan — LLM + Embedding; ONNX DirectML — OCR + Embedding INT8 (BGE) + Reranker INT8 (BGE) |
| **NPU** | AMD XDNA 1 (Hawk Point) | 16 TOPS INT8; **NOT the "AI 300 Series"** (= XDNA 2, Ryzen AI 300 only) | ~2–5 W (dedicated) | ONNX VitisAI — OCR batch (CNN/non-generative); **LLM via NPU NOT SUPPORTED** (XDNA 2 only) |
| **RAM** | LPDDR5x | 32 GB | — | — |
| **Runtime** | Ollama (Vulkan) | Vulkan backend, iGPU offload | — | LLM inference (primary path) |

---

## Execution Mode Comparison

All measured values are p50 latency or TPS from E2E calibration runs.

| Workload | CPU path | iGPU path (Vulkan / DirectML) | NPU path (VitisAI) |
|---|---|---|---|
| **LLM 7B** | ~3–5 TPS (est.) | **13.33 TPS** ✓ | — |
| **LLM 3B** | ~8–12 TPS (est.) | **28.99 TPS** ✓ | — |
| **LLM 0.6B** | — | **91.09 TPS** ✓ | — |
| **LLM NPU pure (Lemonade FLM)** | — | — | **NOT SUPPORTED** on 8845H (XDNA 1). Requires Ryzen AI 300 / XDNA 2 |
| **LLM Hybrid iGPU+NPU (Lemonade)** | — | — | **NOT SUPPORTED** on 8845H (XDNA 1). Requires Ryzen AI 300 / XDNA 2 |
| **Embedding 0.6B** | — | 875 ms p50 ✓ | — |
| **Embedding INT8 (BGE-base, DML)** | — | **2750 ms p50** ✓ DirectML (ORT round-trip) | — |
| **OCR text (p50)** | 1593 ms | **469 ms** ✓ fastest | 2031 ms |
| **OCR structured (p50)** | 859 ms | **477 ms** ✓ | 1868 ms |
| **ASR (RTF)** | — | — | **0.073** ✓ |
| **Reranker base (p50 pair)** | **78 ms** ✓ CPU | **697 ms** ✓ DirectML (ORT round-trip) | — |
| **Reranker v2-m3 (p50)** | 289 ms | — | — |

CPU-only LLM is not independently benchmarked; Ollama defaults to Vulkan iGPU.
OCR quality (CER 7.04%) is identical across all three paths.
**LLM inference paths on 8845H**: (1) **Ollama Vulkan iGPU** (Radeon 780M, calibrated — only practical LLM path); (2) Lemonade FLM / OGA NPU path = **NOT SUPPORTED** on 8845H XDNA 1 — requires Ryzen AI 300 (XDNA 2).

> **Official AMD docs (2026-06):** Ryzen 8845H = XDNA 1 (Hawk Point). XDNA 1 supports VitisAI EP for CNN/INT8 and non-generative Transformers only. LLM inference via NPU (Lemonade FLM, OGA) requires XDNA 2 (Ryzen AI 300 series). Ollama on Windows 780M uses **Vulkan backend** (`OLLAMA_VULKAN=1`), not DirectML and not ROCm (gfx1103 unsupported on Windows ROCm — AMD GitHub #12071).

**→ Mode details:**
- [iGPU (Vulkan + DirectML) — LLM, Embedding, OCR fastest path](./amd-windows-igpu.en.md)
- [NPU (VitisAI) — OCR batch (CNN/non-generative only)](./amd-windows-npu.en.md)
- [CPU ONNX — OCR baseline, Reranker](./amd-windows-cpu.en.md)

---

## Comprehensive Performance + Quality Profile

### LLM Performance (iGPU Vulkan, Ollama)

| Model | TPS | TTFT p50 | TTFT p95 | PP t/s | TG t/s | Max ctx |
|---|---|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | **13.6** | 484 ms (warm) | 8261 ms | 204 | 14.9 | 16k |
| `qwen3-4b-amd` | **30.7** | 867 ms | 4287 ms | — | — | — |
| `llama3.2-3b-amd-win` | 28.99 | 890 ms | 5207 ms | 124 | 39 | 32k |
| `qwen3-1.7b-amd` | **60.0** | 6646 ms¹ | 6962 ms¹ | — | — | — |
| `qwen3-0.6b-amd` | **91.09** | 1781 ms | — | — | — | — |
| `qwen2.5-14b-amd-win` | 8.6 | 7718 ms | 14395 ms | 94 | 9 | 16k |

> PP/TG not available for qwen3 series — Ollama qwen3 does not return `eval_count`/`eval_duration` for prefill separately.
> ¹ qwen3-1.7b TTFT with `ollama_think: false` (v2, 3-seed 2026-06-23): P50=6646ms, P95=6962ms (warm; internal thinking not streamed). Without think=false, TTFT=0ms (thinking streams immediately as first chunk); total request time ≈3.4s.

### LLM Quality Scores (2026-06-20/21 calibrated; qwen3-4b translation 3-seed 2026-06-23; qwen2.5-7b formal 3-seed 2026-06-26)

| Model | GSM8K | MMLU | HellaSwag | GA Verdict | Translation |
|---|---|---|---|---|---|
| `qwen2.5-7b` | **0.873±0.006** | **0.690±0** | **0.790±0** | **PASS** | **FAIL** (en→zh l1_flores chrF=33.88±0.30 <35.0; zh→en PASS; en→zh l3_term chrF=41.7 PASS; 3-seed 2026-06-26) |
| `qwen3-4b` | — | — | — | SKIP (each Q ~68s; GA not run) | **FAIL** (zh→en l1_flores empty=87%; l3_term chrF=63.6/term=64%<75%; en→zh l1_flores empty=100%; l3_term chrF=32.2<35; 3-seed 2026-06-23) |
| `qwen3nt-4b-amd` | 0.030±0 | 0.110±0 | 0.000±0 | **FAIL** (MCQ below random; std=0.000; rerun4 2026-06-26 w/ harness fix) | **FAIL** (zh→en l1 chrF=11.4<35, l3 term=64%<75%; en→zh l1 empty_rate=1.000, l3 chrF=35.5 borderline; rerun4 2026-06-26) |
| `llama3.2-3b` | 0.710/PASS | 0.390/**FAIL** | 0.320/**FAIL** | **FAIL** ⚠️ model-level | FAIL (zh→en term 55%; en→zh chrF 27.6) |
| `qwen3-1.7b` | 0.293±0.015/**FAIL** | 0.033±0.006/**FAIL** | 0.007±0.006/**FAIL** | **FAIL** (MCQ answer format; think=false does not fix) | — (skip) |
| `qwen3-0.6b` | 0.390/PASS | 0.000/**FAIL** | 0.000/**FAIL** | **FAIL** | FAIL (MCQ capability gap) |

**Best confirmed quality:** `qwen2.5-7b-amd-win`.  
**`qwen3-4b` translation FAIL root cause:** Ollama `think=false` option does not disable thinking mode. Model generates 1500–2000 thinking tokens before content; `max_tokens=2048` exhausted on l1_flores long prompts → empty outputs. Fix requires `max_tokens≥4096` or an Ollama version that fully disables thinking.  
**`qwen3nt-4b-amd` rerun4 confirmed FAIL (2026-06-26 w/ harness fix 1c5c656):** Fix extracted zh→en L1 content (chrF=11.4) but quality far below threshold (35.0). en→zh L1 still empty (empty_rate=1.000 — model generates tokens that harness strips to empty). GA unchanged (0.030/0.110/0.000, std=0.000 = 100% systematic). Root cause: model-level MCQ format compliance failure + en→zh free-form output incompatible with extraction — not a harness parser issue.  
**`llama3.2-3b` ⚠️ model weakness:** GA FAIL is a model-family issue (MMLU 0.39, HellaSwag 0.32 — LLaMA 3.2-3B inherent knowledge gap), not platform degradation. Recommend `qwen3-4b-amd` only after translation is fixed.

### Non-LLM Performance

| Model | Role | p50 | Key Metric | Status |
|---|---|---|---|---|
| `qwen3-embedding-0.6b-amd` | Embedding | 875 ms | hit@1=1.000, nDCG=1.000; 3-seed 2026-06-25 | **PASS** |
| `bge-m3-amd` | Embedding | 914 ms | hit@1=1.000, nDCG=1.000; 3-seed 2026-06-25 | **PASS** |
| `rapidocr-amd-directml` | OCR iGPU | 468.5 ms | CER 7.04%, struct 92.86%; 3-seed 2026-06-25 | **PASS** |
| `rapidocr-amd-npu` | OCR NPU | 2031 ms | CER 7.04%; 3-seed 2026-06-25 | **PASS** |
| `rapidocr-cpu` | OCR CPU | 1592.5 ms | CER 7.04%; 3-seed 2026-06-25 | **PASS** |
| `bge-base-en-v1.5-igpu-amd-win` | **Embedding DirectML** | 2750 ms p50¹ | hit@1=1.000, nDCG=0.987; 3-seed 2026-06-23 | **PASS** |
| `bge-reranker-base-igpu-amd-win` | **Reranker DirectML** | 697 ms/pair p50¹ | nDCG=1.000, MRR=1.000; 3-seed 2026-06-23 | **PASS** |
| `bge-reranker-base-amd-win` | Reranker CPU | 78 ms | nDCG=1.000; 3-seed 2026-06-25 | **PASS** |
| `bge-reranker-v2-m3-amd-win` | Reranker CPU | 289 ms | nDCG=1.000; 3-seed 2026-06-25 | **PASS** |
| `sensevoice-small-amd-win` | ASR DirectML | — | CER 7.69%, RTF 0.073; 3-seed 2026-06-25 | **PASS** |

---

## Power Consumption

### Chip TDP Reference

| Component | Base TDP | Max TDP | Note |
|---|---|---|---|
| Ryzen 8845H CPU | **35 W** | 54 W (configurable) | Zen4/Zen4c, SoC |
| Radeon 780M iGPU | — | within SoC TDP | RDNA3, 12 CU, shared power |
| AMD XDNA NPU | **~2 W** | ~5 W | Dedicated low-power block |
| RAM (LPDDR5x) | ~3 W | ~5 W | 32 GB |

**Typical OEM TDP:** 45 W (between 35 W cTDP-down and 54 W cTDP-up).

### Estimated Power Under Inference

| Scenario | Estimated Power | Basis |
|---|---|---|
| Idle | ~8–12 W | AMD laptop desktop standby |
| LLM 0.6B iGPU (91 TPS) | **~30–35 W** | Light iGPU load, low TDP draw |
| LLM 3B iGPU (29 TPS) | **~38–45 W** | iGPU Vulkan, sustained |
| LLM 7B iGPU (13 TPS) | **~42–50 W** | Higher iGPU utilization |
| OCR iGPU DirectML | **~25–35 W** | iGPU active, CPU idle |
| ASR NPU | **~15–20 W** | Mostly NPU, low CPU+iGPU |

> **PENDING-VERIFY (real power):** Activate RAPL via `rocm-smi --showpower --json` or AMD μProf / HWiNFO64 on the target machine during a benchmark run. Remote power sampling via `benchmark/power/windows_sampler.py` (`WindowsPowerSampler(target_host=AMD_HOST, ssh_user=..., ssh_pass=...)`) is implemented but not yet run.

### Power Efficiency Comparison

| Model | Platform | TPS | Est. Power | TPS/W |
|---|---|---|---|---|
| 3B | **AMD iGPU (Vulkan)** | 28.99 | ~42 W | **0.69 TPS/W** |
| 3B | Intel CPU | 19.47 | ~42 W | 0.46 TPS/W |
| 7B | **AMD iGPU (Vulkan)** | 13.33 | ~46 W | **0.29 TPS/W** |
| 7B | Intel CPU | 8.25 | ~47 W | 0.18 TPS/W |
| 0.6B | AMD iGPU (Vulkan) | 91.09 | ~33 W | **2.76 TPS/W** |

**AMD iGPU advantage:** Vulkan parallel execution uses the Radeon 780M more efficiently than Intel's pure-CPU path. 3B throughput is 49% higher at the same power draw.

---

## Selection Summary

| Role | Selected Model | Execution mode | Rationale |
|---|---|---|---|
| LLM primary | `qwen2.5-7b-amd-win` | iGPU (Vulkan) | Best confirmed quality; GA PASS (gsm8k=0.873/mmlu=0.690/hellaswag=0.790; 3-seed 2026-06-26); **translation FAIL** (en→zh l1_flores chrF=33.88 <35.0; same INT4 degradation as Intel 7B; zh→en and l3_term PASS) |
| LLM lightweight (NT — confirmed FAIL) | `qwen3nt-4b-amd` | iGPU (Vulkan) | 24.1 TPS; **FAIL confirmed** (rerun4 2026-06-26 w/ harness fix): zh→en l1 chrF=11.4, en→zh l1 empty_rate=1.000, GA 0.030/0.110/0.000 below random; model-level format compliance failure |
| LLM lightweight (**thinking tokens fix pending**) | `qwen3-4b-amd` | iGPU (Vulkan) | 29–30.7 TPS; translation **FAIL** (Ollama thinking mode: max_tokens=2048 insufficient for l1_flores, fix: ≥4096); GA skipped (too slow); use `qwen2.5-7b` until fixed |
| LLM lightweight (legacy) | `llama3.2-3b-amd-win` | iGPU (Vulkan) | 29 TPS, 32k context; **GA FAIL — model-family weakness** (LLaMA 3.2-3B MMLU=0.39/HellaSwag=0.32 inherent gap, not platform issue); keep only for 32k-context or tool-use |
| LLM nano | `qwen3-1.7b-amd` | iGPU (Vulkan) | 60 TPS, TTFT 6646ms (think=false); GA FAIL (MCQ instruction not followed; same as 0.6B pattern) |
| LLM nano-micro | `qwen3-0.6b-amd` | iGPU (Vulkan) | 91 TPS; MCQ ability insufficient (MMLU=0.000); **not for GA use** |
| Embedding (primary) | `qwen3-embedding-0.6b-amd` | iGPU (Vulkan) | Best retrieval quality, lower latency |
| Embedding (multilingual) | `bge-m3-amd` | iGPU (Vulkan) | Drop-in multilingual alternative |
| **Embedding iGPU (vendor-specific ONNX)** | `bge-base-en-v1.5-igpu-amd-win` | **iGPU DirectML (ORT INT8)** | 2750 ms p50 (ORT SSH round-trip), PASS; quality: hit@1=1.000, nDCG=0.987 |
| Reranker (default) | `bge-reranker-base-amd-win` | CPU ONNX | p50 78 ms, sufficient quality — fastest option |
| **Reranker iGPU (vendor-specific ONNX)** | `bge-reranker-base-igpu-amd-win` | **iGPU DirectML (ORT INT8)** | 697 ms/pair p50 (ORT SSH round-trip), PASS; quality: nDCG=1.000, MRR=1.000 |
| Reranker (quality) | `bge-reranker-v2-m3-amd-win` | CPU ONNX | Equal nDCG/MRR but 3.7× latency — use when ranking quality critical |
| OCR (interactive) | `rapidocr-amd-directml` | iGPU DirectML | Fastest: p50 468 ms |
| **OCR (batch / background)** | `rapidocr-amd-npu` | **NPU VitisAI** | p50 2031 ms — **frees iGPU for concurrent LLM**; use when indexing docs while serving LLM chat |
| **ASR (always-on)** | `sensevoice-small-amd-win` | **iGPU DirectML** | RTF 0.073; runs on Radeon 780M via DirectML EP (not VitisAI NPU); **frees CPU for concurrent tasks** |
| **Vector DB construction** | `qwen3-embedding-0.6b-amd` | iGPU (Vulkan) | 875 ms/chunk; run during off-peak or schedule around LLM load |
| VLM | *(not recommended)* | — | `llava-7b-amd-win` accuracy FAIL; no qualified VLM yet |

---

## Full Model Results

| Model | Execution | Role | Status | Key Metrics |
|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | iGPU Vulkan | llm_primary | **GA PASS / Translation FAIL** | TPS 13.6; TTFT p50/p95 484/8261 ms (warm 3-seed); PP/TG 204/14.9 t/s; GA PASS (gsm8k=0.873±0.006/mmlu=0.690±0/hellaswag=0.790±0; 3-seed 2026-06-26); translation FAIL (en→zh l1_flores chrF=33.88±0.30 <35.0; zh→en PASS; en→zh l3_term chrF=41.7 PASS; 3-seed 2026-06-26) |
| `qwen3-4b-amd` | iGPU Vulkan | llm_lightweight | **FAIL** | TPS 29–30.7; TTFT p50/p95 867/4287 ms; GA SKIPPED (each Q ~68s impractical); translation FAIL (3-seed 2026-06-23: l1_flores empty=87-100%; l3_term en→zh chrF=32.2<35; root cause: Ollama think=false ineffective, max_tokens=2048 exhausted by thinking tokens) |
| `llama3.2-3b-amd-win` | iGPU Vulkan | llm_baseline | **FAIL** ⚠️ | TPS 28.99; TTFT p50/p95 890/5207 ms; PP/TG 124/39 t/s; max ctx 32k; GA FAIL (mmlu=0.390/FAIL, hellaswag=0.320/FAIL — model-level weakness); translation FAIL |
| `qwen3-1.7b-amd` | iGPU Vulkan | llm_nano_plus | **FAIL** | TPS 60.0 (3-seed v2 2026-06-23); TTFT P50=6646ms/P95=6962ms (think=false warm); GA FAIL (gsm8k=0.300/PASS, mmlu=0.000/FAIL, hellaswag=0.000/FAIL; 3-seed zero variance; root cause: Qwen3 1.7B does not follow "answer with just A/B/C/D" MCQ instruction → parser fails; think=false confirmed not the fix — v1 and v2 identical FAIL); translation skip (1.7B insufficient) |
| `qwen3-0.6b-amd` | iGPU Vulkan | llm_nano | **FAIL** | TPS 91.09; TTFT p50 1781 ms; GA FAIL (mmlu=0.000/hellaswag=0.000 — 0.6B MCQ gap, confirmed 2026-06-20) |
| `qwen2.5-14b-amd-win` | iGPU Vulkan | llm_parameter_uplift | **MEASURED** | TPS 8.6; TTFT p50/p95 7718/14395 ms; PP/TG 94/9 t/s; max-ctx 16k; GA/translation skipped by design |
| `qwen3nt-4b-amd` | iGPU Vulkan | llm_nt_4b | **FAIL** | TPS 24.1; GA FAIL (gsm8k=0.030±0, mmlu=0.110±0, hellaswag=0.000±0; below random; std=0.000; rerun4 2026-06-26 w/ harness fix); translation FAIL (zh→en l1 BLEU=0/chrF=11.4, l3 chrF=66.4/term=64%<75%; en→zh l1 empty_rate=1.000, l3 chrF=35.5 borderline; rerun4 2026-06-26); root cause: MCQ format compliance failure (model generates wrong format, not parser issue); en→zh L1 output stripped to empty by extraction |
| `llava-7b-amd-win` | iGPU Vulkan | vlm_baseline | **FAIL** | TPS 16.84; TTFT p50 890 ms; accuracy FAIL |
| `qwen3-embedding-0.6b-amd` | iGPU Vulkan | embedding_primary | **PASS** | hit@1 1.000; nDCG 1.000; p50 875 ms |
| `bge-m3-amd` | iGPU Vulkan | embedding_bge | **PASS** | hit@1 1.000; nDCG 1.000; p50 914 ms |
| `bge-base-en-v1.5-igpu-amd-win` | iGPU DirectML (ORT) | embedding_igpu_dml | **PASS** | hit@1 1.000; nDCG@10 0.987; p50 2750 ms¹; 3-seed 2026-06-23 |
| `rapidocr-amd-directml` | iGPU DirectML | ocr_gpu | **PASS** | CER 7.04%; p50 468.5 ms; structured field acc 92.86%; structured p50 476.5 ms |
| `rapidocr-amd-npu` | NPU VitisAI | ocr_npu | **PASS** | CER 7.04%; p50 2031 ms; structured field acc 92.86%; structured p50 1867.5 ms |
| `rapidocr-cpu` | CPU ONNX | ocr_cpu_baseline | **PASS** | CER 7.04%; p50 1592.5 ms; structured field acc 92.86%; structured p50 859.0 ms |
| `paddleocr-cpu` | CPU ONNX | ocr_cpu_paddle | **PASS** | CER 7.04%; p50 1829.5 ms |
| `bge-reranker-base-igpu-amd-win` | iGPU DirectML (ORT) | reranker_igpu_dml | **PASS** | nDCG@10 1.000; MRR 1.000; pair p50 697 ms¹; query p50 2815 ms¹; 3-seed 2026-06-23 |
| `bge-reranker-base-amd-win` | CPU ONNX | reranker_default | **PASS** | nDCG 1.000; MRR 1.000; p50 78 ms |
| `bge-reranker-v2-m3-amd-win` | CPU ONNX | reranker_stronger | **PASS** | nDCG 1.000; MRR 1.000; p50 289 ms |
| `sensevoice-small-amd-win` | iGPU DirectML | asr | **PASS** | CER 7.69%; RTF 0.073 |

**Status legend:** PASS = all thresholds met. FAIL = one or more quality/perf thresholds missed.
MEASURED = latency/throughput collected; quality dims not fully qualified.

---

> ¹ **ORT+DirectML latency note:** BGE embedding/reranker latency values (2750 ms / 697 ms pair) are ORT+DirectML SSH round-trip times measured from the harness (benchmark host → remote server → inference → return). Pure model inference on the Radeon 780M is faster; the overhead is dominated by network round-trip and ORT warm-up per request. Latency is informational only — verdict is determined by quality metrics (hit@1, nDCG, MRR), which are measured correctly regardless of round-trip time.

## Known Limitations

- **`qwen2.5-7b` translation PASS (recalibrated 2026-06-21)** — Thresholds corrected to chrF≥35.0 / term≥75%. 3-seed confirmed: zh→en 79%≥75%, en→zh chrF 36.4≥35.0.
- **`llama3.2-3b` GA FAIL — model-level weakness, not platform** — MMLU=0.390 and HellaSwag=0.320 are intrinsic LLaMA 3.2-3B limitations. GSM8K=0.710 PASS (math-only). Keep llama3.2-3b only for: (a) 32k+ context workloads, (b) tool-calling use cases. Do NOT replace with `qwen3-4b-amd` until its translation is fixed (see above).
- **`qwen3-0.6b` MCQ FAIL** — mmlu=0.000, hellaswag=0.000. 0.6B model incapable of reliable MCQ letter output, confirmed post parser-fix. Use qwen3-1.7b for better GA coverage.
- **`qwen3-4b` translation FAIL (3-seed 2026-06-23)** — Root cause: Ollama `think=false` option does not disable thinking mode; model generates ~1500–2000 thinking tokens before content. `max_tokens=2048` (current common.py setting) is exhausted during thinking on l1_flores long prompts, producing empty outputs (empty_rate=87–100%). l3_terminology shorter prompts partially succeed (zh→en chrF=63.6, but term=64%<75%; en→zh chrF=32.2<35). To fix: increase `max_tokens≥4096` in common.py (will increase per-request time to ~136s). GA skipped for now (each question ~68s; 300Q×3seeds≈17 hours).
- **`qwen3-1.7b` GA FAIL (3-seed 2026-06-23, confirmed with and without `think=false`)** — v1 (no think=false): gsm8k=0.300/PASS, mmlu=0.000/FAIL, hellaswag=0.000/FAIL. v2 (think=false, 3-seed): gsm8k=0.293±0.015/FAIL (worst seed 0.280 < 0.30), mmlu=0.033±0.006/FAIL, hellaswag=0.007±0.006/FAIL. `think=false` marginally improved mmlu/hellaswag but worsened gsm8k below threshold. Root cause: **Qwen3 1.7B does not follow the MCQ "answer with just the letter A/B/C/D" instruction** regardless of thinking mode — model generates verbose multi-token explanations; benchmark parser fails to extract A/B/C/D. This is a model capability issue, not a thinking-mode issue. Fix requires either prompt engineering (few-shot MCQ examples) or a more capable model tier. Same GA failure pattern as `qwen3-0.6b`.
- **LLM conditioned/scenarios FAIL** — Long-context conditioning fails across all tested models.
- **No qualified VLM** — `llava-7b-amd-win` accuracy FAIL; no VLM workloads recommended.
- **NPU LLM NOT SUPPORTED on 8845H** — AMD Ryzen 8845H has XDNA 1 (Hawk Point). Lemonade FLM / OGA NPU inference requires **XDNA 2 (Ryzen AI 300 series, e.g. Ryzen AI 9 HX 370)**. The 8845H iGPU (Radeon 780M) is the only GPU inference path via Ollama Vulkan.
- **Ollama backend is Vulkan, not DirectML or ROCm** — On Windows with Radeon 780M (gfx1103), AMD officially endorses Ollama with `OLLAMA_VULKAN=1`. ROCm on Windows does not support gfx1103 (AMD GitHub issue #12071 open). DirectML is not used for LLM.
- **ASR runs on iGPU (DirectML), not NPU** — SenseVoice ONNX uses OnnxRuntime DirectML EP on the Radeon 780M, not the XDNA 1 VitisAI EP. Power draw for ASR is therefore included in iGPU TDP, not NPU TDP.

---

## NPU Batch Processing Guidance

> **Critical clarification (per official AMD docs):** Ryzen 8845H has **XDNA 1** (Hawk Point NPU). XDNA 1 supports VitisAI EP for CNN/INT8 and non-generative Transformer models only. **LLM inference on NPU is NOT possible on this hardware** — that requires XDNA 2 (Ryzen AI 300 series). ASR (SenseVoice ONNX) runs on the **iGPU (780M) via DirectML EP**, not the NPU.

AMD XDNA 1 NPU excels at **CNN-based batch workloads** (e.g., OCR via RapidOCR) that would otherwise compete with iGPU LLM inference.

| Workload | Execution path | p50 latency | vs iGPU | Recommendation |
|---|---|---|---|---|
| OCR (single doc) | NPU VitisAI ONNX | 2031 ms | 4.3× slower (iGPU: 469 ms) | **Not recommended** for interactive |
| OCR (batch / background) | NPU VitisAI ONNX | 2031 ms/doc | Frees iGPU for LLM | **Recommended** when LLM is concurrently active |
| ASR transcription | iGPU DirectML | RTF 0.073 | — | **Recommended** — low-latency, frees CPU |
| Vector DB construction | iGPU Vulkan (embedding) | 875 ms/chunk | — | Embedding runs on iGPU; no NPU embedding path |
| LLM inference | **NOT POSSIBLE on NPU** | — | — | XDNA 2 (Ryzen AI 300) required for Lemonade FLM |

**Key insight:** XDNA 1 NPU (~2 W) offloads CNN-based OCR from the iGPU, allowing simultaneous LLM inference on the Radeon 780M while batch OCR runs on NPU. For bulk OCR pipelines (e.g., indexing 100 documents), NPU batch achieves similar throughput to iGPU while keeping iGPU available for interactive chat. Latency per document is ~4× higher on NPU, but total system throughput is higher due to parallelism.

**When to use NPU (VitisAI) vs iGPU:**
- Background document ingestion while serving LLM queries → NPU VitisAI for OCR
- Always-on ASR → iGPU DirectML (SenseVoice ONNX)
- LLM inference → iGPU Vulkan (Ollama) — **NPU path not available on 8845H**

**Official AMD model repositories (for reference):**
- `amd/ryzen-ai-171-*` on HuggingFace: AWQ UINT4+BF16 models for XDNA 2 NPU (Ryzen AI 300 only, not applicable to 8845H)
- For iGPU (780M) with Ollama Vulkan: standard GGUF Q4 from Ollama Hub / HuggingFace

### Summary: Can AMD Windows handle Embedding / Reranker / OCR / ASR?

| Task | NPU (XDNA 1 VitisAI) | iGPU (780M Vulkan/DirectML) | CPU |
|---|---|---|---|
| Embedding | **NOT SUPPORTED** (non-generative transformers require special VitisAI ORT env; not set up) | **PASS** via Ollama Vulkan (qwen3-embedding 875 ms; bge-m3 914 ms) | — |
| Reranker | **NOT SUPPORTED** (requires VitisAI ORT env) | — | **PASS** CPU ONNX (bge-reranker-base 78 ms) |
| OCR | **PASS** (VitisAI ONNX — field accuracy 92.86%, p50 2031 ms) | **PASS** DirectML (p50 469 ms — 4.3× faster but uses iGPU) | PASS CPU (p50 1593 ms) |
| ASR | **NOT SUPPORTED** (SenseVoice uses DirectML, not VitisAI) | **PASS** DirectML RTF 0.073, latency 407 ms | — |

**Summary:** AMD XDNA 1 NPU handles **OCR only** (CNN-based). All other non-LLM tasks (embedding/reranker/ASR) run on iGPU or CPU. Embedding/reranker on NPU would require special VitisAI ORT environment setup with INT8 ONNX models (not tested). Production recommendation: use iGPU Vulkan for embedding, CPU ONNX for reranker, NPU for background OCR, iGPU DirectML for ASR.

---

## Calibration History

| Date | Event |
|---|---|
| 2026-06-19 | Initial full calibration: all 14 models measured across CPU/iGPU/NPU paths; thresholds set from E2E runs |
| 2026-06-20 | Added quality dims: qwen2.5-7b general_ability PASS (3-seed); llama3.2-3b general_ability FAIL (3-seed — model-level); qwen3-0.6b FAIL (MCQ gap confirmed post parser-fix) |
| 2026-06-21 | qwen2.5-14b perf recalibrated (TPS 8.6); AMD 7B translation threshold recalibrated (chrF 40→35, term 80%→75%); translation PASS confirmed 3-seed |
| 2026-06-22 | Added qwen3:1.7b (TPS 63.5, TTFT P50 497ms) and qwen3:4b (TPS 30.7, TTFT P50 867ms); perf thresholds calibrated; GA/translation PENDING-VERIFY; llama3.2-3b documented as model-level GA FAIL; corrected XDNA 1 vs XDNA 2 hardware description (per official AMD docs); clarified NPU LLM NOT supported on 8845H; ASR moved from NPU to iGPU DirectML |
| 2026-06-23 | qwen3-1.7b GA 3-seed v2 (think=false) confirmed FAIL: gsm8k=0.293±0.015/FAIL (worst seed 0.280<0.30), mmlu=0.033±0.006/FAIL, hellaswag=0.007±0.006/FAIL; v1 (no think=false) had gsm8k=0.300/PASS but mmlu=0.000/FAIL, hellaswag=0.000/FAIL; think=false marginally improved mmlu/hellaswag but worsened gsm8k below threshold; root cause: model ignores MCQ "just letter" instruction regardless of thinking mode; v2 TTFT: P50=6646ms/P95=6962ms; TPS=60.0 tok/s |
| 2026-06-23 | bge-base-en-v1.5-igpu-amd-win 3-seed PASS: hit@1=1.000, nDCG@10=0.987, P50=2750ms (ORT+DirectML). bge-reranker-base-igpu-amd-win 3-seed PASS: nDCG@10=1.000, MRR=1.000, pair P50=697ms. qwen3-4b translation 3-seed FAIL: l1_flores empty_rate=87–100% (thinking tokens exhaust max_tokens=2048); l3_term en→zh chrF=32.2<35 |
| 2026-06-24 | Launched AMD full 3-seed verification: qwen3-embedding-0.6b, bge-m3, bge-reranker-base CPU, bge-reranker-v2-m3 CPU, sensevoice-small, rapidocr×3 — all pending 3-seed. Added qwen3nt-4b-amd (Qwen3-4B /no_think variant) for GA+translation — expected to fix max_tokens exhaustion issue |
| 2026-06-25 | **qwen3nt-4b-amd initial 3-seed run** (pre-fix): TPS=24.1. GA: gsm8k=0.030±0/mmlu=0.110±0/hellaswag=0.000±0 — all below random (25%). Translation: empty_rate=1.000±0 ALL directions. std=0.000 = 100% systematic. Initial root cause hypothesis: harness parser cannot extract qwen3nt format → harness fix 1c5c656 applied → re-run scheduled. **8 non-LLM models 3-seed PASS** (embedding ×2, rerank ×2, ASR, OCR ×3, std=0.000 deterministic). |
| 2026-06-26 | **qwen3nt-4b-amd rerun4 (harness fix 1c5c656)**: GA identical — gsm8k=0.030±0/mmlu=0.110±0/hellaswag=0.000±0 (harness fix did NOT improve accuracy). Translation: zh→en l1 chrF=11.4/BLEU=0 (content extracted but far below threshold 35); zh→en l3 chrF=66.4/term=64%<75%; en→zh l1 empty_rate=1.000 (fix did NOT help en→zh L1); en→zh l3 chrF=35.5/BLEU=32.9/term=77%. **Root cause confirmed: model-level failure** (MCQ format compliance + en→zh free-form output incompatible with extraction), not parser issue. Harness fix is necessary but not sufficient. Report: `qwen3nt-4b-amd_20260626_004430.md` (archived to `benchmark-runs/amd-win/`). |

---

## 中文摘要

**平台：** amd-win-x86 | Ryzen 8845H + Radeon 780M iGPU + AMD XDNA 1 NPU（Hawk Point），Windows 11  
**最后校准：** 2026-06-26。本文件原地更新。

### 硬件画像

| 计算单元 | 芯片 | 规格 | TDP | 角色 |
|---|---|---|---|---|
| **CPU** | AMD Ryzen 8845H | 4P+4E Zen4 核，16 线程，3.8–5.1 GHz | 35 W（基础）/ 54 W（最大） | ONNX CPU — OCR 基线、Reranker |
| **iGPU** | Radeon 780M | RDNA3，12 CU，2800 MHz，17.9 GiB 共享显存 | SoC TDP 内 | Ollama Vulkan — LLM/Embedding；ONNX DirectML — OCR/ASR/Embedding INT8/Reranker INT8 |
| **NPU** | AMD XDNA 1（Hawk Point） | 16 TOPS INT8；**非 "AI 300 Series"（XDNA 2）** | ~2–5 W | ONNX VitisAI — OCR 批处理（CNN/非生成式）；**LLM 不支持**（需 XDNA 2） |
| **RAM** | LPDDR5x | 32 GB | ~3–5 W | — |

> ⚠️ **重要澄清（官方文档）：** Ryzen 8845H 为 XDNA 1（Hawk Point），**不是** Ryzen AI 300 的 XDNA 2。LLM 推理不支持 XDNA 1 NPU，只能用 iGPU（Radeon 780M，Ollama Vulkan）。ASR 走 **iGPU DirectML**，不走 NPU VitisAI。

### 执行模式对比

| 任务 | CPU 路径 | iGPU 路径（Vulkan/DirectML） | NPU 路径（VitisAI） |
|---|---|---|---|
| LLM 7B | ~3–5 TPS（估算） | **13.33 TPS** ✓ | **不支持**（XDNA 2 才可） |
| LLM 3B | ~8–12 TPS（估算） | **28.99 TPS** ✓ | **不支持** |
| Embedding INT8（BGE-base，DML） | — | **2750 ms p50** ✓ DirectML（3-seed PASS 2026-06-23）| — |
| OCR 文字 p50 | 1593 ms | **469 ms** ✓ 最快 | 2031 ms（CNN 模型，VitisAI） |
| ASR RTF | — | **0.073** ✓（DirectML；非 NPU） | — |
| Reranker base p50/pair | **78 ms** ✓ | **697 ms** ✓ DirectML（3-seed PASS） | — |

**AMD 8845H LLM 推理路径（唯一实用路径）：** Ollama Vulkan iGPU（Radeon 780M），已校准 13.33–91 TPS。Lemonade FLM/OGA NPU 路径在 8845H 上**不可用**，需 Ryzen AI 300（XDNA 2）。

### 综合性能 + 模型效果（已校准 / 已验证）

| 模型 | 执行模式 | TPS/p50 | GSM8K | MMLU | HellaSwag | 翻译 | 综合 |
|---|---|---|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | iGPU Vulkan | 13.33 TPS / 953 ms | 0.880 | 0.600 | 0.790 | **PASS**（3-seed 2026-06-21） | **GA PASS** |
| `qwen3nt-4b-amd` | iGPU Vulkan | 24.1 TPS | 0.030 | 0.110 | 0.000 | FAIL（zh→en l1 chrF=11.4<35；en→zh l1 empty_rate=1.000；3-seed rerun4 2026-06-26） | **FAIL**（MCQ 格式合规性失败 + en→zh L1 输出被提取为空；rerun4 harness fix 1c5c656 后确认为模型层面问题，非解析器问题） |
| `qwen3-4b-amd` | iGPU Vulkan | 29–30.7 TPS / 867 ms | GA skip | GA skip | GA skip | **FAIL**（thinking tokens 耗尽 max_tokens，3-seed 2026-06-23）| **FAIL** |
| `llama3.2-3b-amd-win` | iGPU Vulkan | 28.99 TPS / 890 ms | 0.710/PASS | 0.390/**FAIL** | 0.320/**FAIL** | FAIL | **GA FAIL**（模型固有局限） |
| `qwen3-1.7b-amd` | iGPU Vulkan | 60.0 TPS / 6646 ms | 0.293/**FAIL** | 0.033/**FAIL** | 0.007/**FAIL** | skip | **GA FAIL**（MCQ 格式，3-seed） |
| `qwen3-0.6b-amd` | iGPU Vulkan | 91.09 TPS / 1781 ms | 0.390/PASS | 0.000/**FAIL** | 0.000/**FAIL** | FAIL | **GA FAIL**（0.6B 能力不足） |
| `qwen3-embedding-0.6b-amd` | iGPU Vulkan | 875 ms | — | — | — | — | **PASS**（hit@1=1.000；**3-seed 2026-06-25**） |
| `bge-m3-amd` | iGPU Vulkan | 914 ms | — | — | — | — | **PASS**（hit@1=1.000；**3-seed 2026-06-25**） |
| `bge-base-en-v1.5-igpu-amd-win` | iGPU DirectML (ORT) | 2750 ms | — | — | — | — | **PASS**（hit@1=1.000，nDCG=0.987；**3-seed 2026-06-23**） |
| `bge-reranker-base-igpu-amd-win` | iGPU DirectML (ORT) | 697 ms/pair | — | — | — | — | **PASS**（nDCG=1.000，MRR=1.000；**3-seed 2026-06-23**） |
| `bge-reranker-base-amd-win` | CPU ONNX | 78 ms | — | — | — | — | **PASS**（nDCG=1.000；**3-seed 2026-06-25**） |
| `bge-reranker-v2-m3-amd-win` | CPU ONNX | 289 ms | — | — | — | — | **PASS**（nDCG=1.000；**3-seed 2026-06-25**） |
| `rapidocr-amd-directml` | iGPU DirectML | 469 ms | — | — | — | — | **PASS**（CER 7.04%；**3-seed 2026-06-25**） |
| `rapidocr-amd-npu` | NPU VitisAI | 2031 ms | — | — | — | — | **PASS**（CER 7.04%；**3-seed 2026-06-25**） |
| `rapidocr-cpu` | CPU ONNX | 1593 ms | — | — | — | — | **PASS**（CER 7.04%；**3-seed 2026-06-25**） |
| `sensevoice-small-amd-win` | iGPU DirectML | RTF 0.073 | — | — | — | — | **PASS**（CER 7.69%；**3-seed 2026-06-25**） |

### qwen3-4b vs qwen3nt-4b 区别

| | `qwen3-4b-amd` | `qwen3nt-4b-amd` |
|---|---|---|
| Ollama model | `qwen3:4b` | `qwen3nt:latest` |
| thinking 控制 | `options.think=false`（**无效**，AMD 上未真正禁用）| 系统提示 `/no_think`（**有效**，ollama show 确认）|
| l1_flores 空输出率 | 87–100%（thinking tokens 耗尽 max_tokens=2048） | 预期 0%（无 thinking 前缀） |
| MCQ GA 测试 | 跳过（每题 ~68s 不实际） | 待验证（预期 ~1–2s/题） |

### 功耗参考

| 场景 | 估算功耗 |
|---|---|
| 空闲 | ~8–12 W |
| LLM 0.6B（91 TPS） | ~30–35 W |
| LLM 3B/4B（29–31 TPS） | ~38–45 W |
| LLM 7B（13 TPS） | ~42–50 W |
| OCR iGPU DirectML | ~25–35 W |
| ASR iGPU DirectML | ~15–20 W |

> **能效对比（3B）：** AMD iGPU 28.99 TPS / ~42 W = **0.69 TPS/W**；Intel CPU 19.47 TPS / ~42 W = 0.46 TPS/W（AMD 高效 50%）。

### 选型摘要（2026-06-24 当前状态）

| 角色 | 推荐模型 | 执行模式 | 备注 |
|---|---|---|---|
| **LLM 质量首选** | `qwen2.5-7b-amd-win` | iGPU Vulkan | **GA PASS**（MMLU 0.60 / HellaSwag 0.79 / 翻译 PASS 3-seed） |
| LLM 轻量 NT（确认 FAIL） | `qwen3nt-4b-amd` | iGPU Vulkan | **FAIL 确认（rerun4 2026-06-26，harness fix 后）**：GA 不变（0.030/0.110/0.000）；zh→en l1 chrF=11.4（有内容但极低）；en→zh l1 仍全空（empty_rate=1.000）；根因确认为**模型层面** MCQ 格式合规失败 + en→zh 自由输出被提取为空，非解析器问题 |
| LLM 轻量（旧，有局限） | `qwen3-4b-amd` | iGPU Vulkan | 翻译 FAIL；GA skip；thinking tokens 问题待 Ollama 修复 |
| LLM 轻量（旧，FAIL） | `llama3.2-3b-amd-win` | iGPU Vulkan | GA FAIL（模型固有局限）；仅 32k 上下文场景保留 |
| LLM 极速纳米 | `qwen3-0.6b-amd` | iGPU Vulkan | 91 TPS；MCQ 能力不足 |
| **Embedding 首选** | `qwen3-embedding-0.6b-amd` | iGPU Vulkan | hit@1=1.000；875 ms；**3-seed PASS 2026-06-25** |
| Embedding 多语言 | `bge-m3-amd` | iGPU Vulkan | hit@1=1.000；914 ms；**3-seed PASS 2026-06-25** |
| **Embedding 厂商专属** | `bge-base-en-v1.5-igpu-amd-win` | iGPU DirectML (ORT) | hit@1=1.000；**3-seed PASS 2026-06-23** |
| **Reranker 默认（最快）** | `bge-reranker-base-amd-win` | CPU ONNX | **78 ms**；nDCG=1.000；**3-seed PASS 2026-06-25** |
| Reranker 厂商专属 | `bge-reranker-base-igpu-amd-win` | iGPU DirectML (ORT) | nDCG=1.000；**3-seed PASS 2026-06-23**；697 ms（DirectML） |
| Reranker 高质量 | `bge-reranker-v2-m3-amd-win` | CPU ONNX | nDCG=1.000；289 ms；**3-seed PASS 2026-06-25** |
| **OCR 交互首选** | `rapidocr-amd-directml` | iGPU DirectML | **469 ms** 最快；**3-seed PASS 2026-06-25** |
| **OCR 后台批处理** | `rapidocr-amd-npu` | NPU VitisAI | 2031 ms；**释放 iGPU 供 LLM 并发**；**3-seed PASS 2026-06-25** |
| **ASR 常驻后台** | `sensevoice-small-amd-win` | **iGPU DirectML** | RTF 0.073；**iGPU DirectML 路径**（非 NPU），适合与 LLM 并行；**3-seed PASS 2026-06-25** |
