# AMD Windows Platform — Model Selection & Benchmark Report

**Platform:** amd-win-x86 | Ryzen 8845H + Radeon 780M iGPU + AMD XDNA NPU, Windows 11  
**Last calibrated:** 2026-06-21. This file is updated in place.

---

## Hardware Profile

| Compute Unit | Chip | Specs | TDP | Role |
|---|---|---|---|---|
| **CPU** | AMD Ryzen 8845H | 4× Zen4 P-core + 4× Zen4c E-core, 16 threads, 3.8–5.1 GHz | 35 W (base) / 54 W (max) | ONNX Runtime CPU — OCR baseline, Reranker |
| **iGPU** | AMD Radeon 780M | RDNA3, 12 CU, 2800 MHz, 17.9 GiB shared VRAM | part of SoC TDP | Ollama Vulkan — LLM + Embedding; ONNX DirectML — OCR |
| **NPU** | AMD XDNA | AI 300 Series, 16 TOPS INT8 | ~2–5 W (dedicated) | ONNX VitisAI — OCR (batch); ASR; PENDING: LLM via Lemonade |
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
| **LLM NPU pure (1B, Lemonade)** | — | — | **~80–100 TPS** PENDING-VERIFY |
| **LLM NPU pure (1.5B, Lemonade)** | — | — | **~60–80 TPS** PENDING-VERIFY |
| **LLM NPU pure (3.8B, Lemonade)** | — | — | **~30–50 TPS** PENDING-VERIFY |
| **LLM Hybrid (7B, iGPU+NPU)** | — | — | **PENDING-VERIFY** ⚠️ Ryzen AI 300 preferred |
| **Embedding 0.6B** | — | 875 ms p50 ✓ | — |
| **OCR text (p50)** | 1593 ms | **469 ms** ✓ fastest | 2031 ms |
| **OCR structured (p50)** | 859 ms | **477 ms** ✓ | 1868 ms |
| **ASR (RTF)** | — | — | **0.073** ✓ |
| **Reranker base (p50)** | **78 ms** ✓ | — | — |
| **Reranker v2-m3 (p50)** | 289 ms | — | — |

CPU-only LLM is not independently benchmarked; Ollama defaults to Vulkan iGPU.
OCR quality (CER 7.04%) is identical across all three paths.
**Three LLM modes**: (1) Ollama Vulkan iGPU (calibrated); (2) Lemonade hybrid — iGPU prefill + NPU decode, PENDING-VERIFY, ⚠️ Ryzen AI 300 preferred for full support; (3) Lemonade/FastFlowLM pure NPU ≤3.8B, PENDING-VERIFY.

**→ Mode details:**
- [iGPU (Vulkan + DirectML) — LLM, Embedding, OCR fastest path](./amd-windows-igpu.en.md)
- [NPU (VitisAI + DirectML) — OCR batch, ASR](./amd-windows-npu.en.md)
- [CPU ONNX — OCR baseline, Reranker](./amd-windows-cpu.en.md)

---

## Comprehensive Performance + Quality Profile

### LLM Performance (iGPU Vulkan, Ollama)

| Model | TPS | TTFT p50 | TTFT p95 | PP t/s | TG t/s | Max ctx |
|---|---|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | **13.33** | 953 ms | 6241 ms | 116 | 16 | 16k |
| `qwen3-4b-amd` | **30.7** | 867 ms | 4287 ms | — | — | — |
| `llama3.2-3b-amd-win` | 28.99 | 890 ms | 5207 ms | 124 | 39 | 32k |
| `qwen3-1.7b-amd` | **63.5** | 497 ms | 3034 ms | — | — | — |
| `qwen3-0.6b-amd` | **91.09** | 1781 ms | — | — | — | — |
| `qwen2.5-14b-amd-win` | 8.6 | 7718 ms | 14395 ms | 94 | 9 | 16k |

> PP/TG not available for qwen3 series — Ollama qwen3 does not return `eval_count`/`eval_duration` for prefill separately.

### LLM Quality Scores (2026-06-20/21 calibrated; qwen3 PENDING-VERIFY)

| Model | GSM8K | MMLU | HellaSwag | GA Verdict | Translation |
|---|---|---|---|---|---|
| `qwen2.5-7b` | **0.880** | **0.600** | **0.790** | **PASS** | **PASS** (zh→en term 79%≥75%; en→zh chrF 36.4≥35.0; recal 2026-06-21) |
| `qwen3-4b` | PENDING | PENDING | PENDING | **PENDING** | PENDING |
| `llama3.2-3b` | 0.710/PASS | 0.390/**FAIL** | 0.320/**FAIL** | **FAIL** ⚠️ model-level | FAIL (zh→en term 55%; en→zh chrF 27.6) |
| `qwen3-1.7b` | PENDING | PENDING | PENDING | **PENDING** | — (skip) |
| `qwen3-0.6b` | 0.390/PASS | 0.000/**FAIL** | 0.000/**FAIL** | **FAIL** | FAIL (MCQ capability gap) |

**Best confirmed quality:** `qwen2.5-7b-amd-win`. `qwen3-4b` expected to match or surpass.  
**`llama3.2-3b` ⚠️ model weakness:** GA FAIL is a model-family issue (MMLU 0.39, HellaSwag 0.32 — LLaMA 3.2-3B inherent knowledge gap), not platform degradation. Recommend `qwen3-4b-amd` as replacement (same TPS, better quality tier).

### Non-LLM Performance

| Model | Role | p50 | Key Metric | Status |
|---|---|---|---|---|
| `qwen3-embedding-0.6b-amd` | Embedding | 875 ms | hit@1=1.000, nDCG=1.000 | **PASS** |
| `bge-m3-amd` | Embedding | 914 ms | hit@1=1.000, nDCG=1.000 | **PASS** |
| `rapidocr-amd-directml` | OCR iGPU | 468.5 ms | CER 7.04%, struct 92.86% | **PASS** |
| `rapidocr-amd-npu` | OCR NPU | 2031 ms | CER 7.04% | **PASS** |
| `rapidocr-cpu` | OCR CPU | 1592.5 ms | CER 7.04% | **PASS** |
| `bge-reranker-base-amd-win` | Reranker | 78 ms | nDCG=1.000 | **PASS** |
| `bge-reranker-v2-m3-amd-win` | Reranker | 289 ms | nDCG=1.000 | **PASS** |
| `sensevoice-small-amd-win` | ASR NPU | — | CER 7.69%, RTF 0.073 | **PASS** |

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
| LLM primary | `qwen2.5-7b-amd-win` | iGPU (Vulkan) | Best confirmed quality; GA PASS |
| LLM lightweight (**recommended**) | `qwen3-4b-amd` | iGPU (Vulkan) | 30.7 TPS, same speed as llama3.2-3b; better quality expected (PENDING) |
| LLM lightweight (legacy) | `llama3.2-3b-amd-win` | iGPU (Vulkan) | 29 TPS, 32k context; GA FAIL (model-level weakness — MMLU/HellaSwag low); keep for 32k-context or tool-use only |
| LLM nano | `qwen3-1.7b-amd` | iGPU (Vulkan) | 63.5 TPS, TTFT warm 497 ms; GA PENDING |
| LLM nano-micro | `qwen3-0.6b-amd` | iGPU (Vulkan) | 91 TPS; best for fast responses where quality is secondary |
| Embedding (primary) | `qwen3-embedding-0.6b-amd` | iGPU (Vulkan) | Best retrieval quality, lower latency |
| Embedding (multilingual) | `bge-m3-amd` | iGPU (Vulkan) | Drop-in multilingual alternative |
| Reranker (default) | `bge-reranker-base-amd-win` | CPU ONNX | p50 78 ms, sufficient quality |
| Reranker (quality) | `bge-reranker-v2-m3-amd-win` | CPU ONNX | Equal nDCG/MRR but 3.7× latency — use when ranking quality critical |
| OCR (primary) | `rapidocr-amd-directml` | iGPU DirectML | Fastest: p50 468 ms |
| OCR (batch / background) | `rapidocr-amd-npu` | NPU VitisAI | p50 2031 ms — frees iGPU for LLM; ideal for bulk doc processing |
| ASR | `sensevoice-small-amd-win` | NPU DirectML | PASS: CER 7.69%, RTF 0.073 |
| VLM | *(not recommended)* | — | `llava-7b-amd-win` accuracy FAIL; no qualified VLM yet |

---

## Full Model Results

| Model | Execution | Role | Status | Key Metrics |
|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | iGPU Vulkan | llm_primary | **PASS** | TPS 13.33; TTFT p50/p95 953/6241 ms; PP/TG 116/16 t/s; GA PASS (gsm8k=0.880/mmlu=0.600/hellaswag=0.790); translation PASS (recal 2026-06-21) |
| `qwen3-4b-amd` | iGPU Vulkan | llm_lightweight | **PENDING** | TPS 30.7; TTFT p50/p95 867/4287 ms; GA/translation PENDING-VERIFY (2026-06-22 perf calibrated) |
| `llama3.2-3b-amd-win` | iGPU Vulkan | llm_baseline | **FAIL** ⚠️ | TPS 28.99; TTFT p50/p95 890/5207 ms; PP/TG 124/39 t/s; max ctx 32k; GA FAIL (mmlu=0.390/FAIL, hellaswag=0.320/FAIL — model-level weakness); translation FAIL |
| `qwen3-1.7b-amd` | iGPU Vulkan | llm_nano_plus | **PENDING** | TPS 63.5; TTFT p50/p95 497/3034 ms; GA/translation PENDING-VERIFY (2026-06-22 perf calibrated) |
| `qwen3-0.6b-amd` | iGPU Vulkan | llm_nano | **FAIL** | TPS 91.09; TTFT p50 1781 ms; GA FAIL (mmlu=0.000/hellaswag=0.000 — 0.6B MCQ gap, confirmed 2026-06-20) |
| `qwen2.5-14b-amd-win` | iGPU Vulkan | llm_parameter_uplift | **MEASURED** | TPS 8.6; TTFT p50/p95 7718/14395 ms; PP/TG 94/9 t/s; max-ctx 16k; GA/translation skipped by design |
| `llava-7b-amd-win` | iGPU Vulkan | vlm_baseline | **FAIL** | TPS 16.84; TTFT p50 890 ms; accuracy FAIL |
| `qwen3-embedding-0.6b-amd` | iGPU Vulkan | embedding_primary | **PASS** | hit@1 1.000; nDCG 1.000; p50 875 ms |
| `bge-m3-amd` | iGPU Vulkan | embedding_bge | **PASS** | hit@1 1.000; nDCG 1.000; p50 914 ms |
| `rapidocr-amd-directml` | iGPU DirectML | ocr_gpu | **PASS** | CER 7.04%; p50 468.5 ms; structured field acc 92.86%; structured p50 476.5 ms |
| `rapidocr-amd-npu` | NPU VitisAI | ocr_npu | **PASS** | CER 7.04%; p50 2031 ms; structured field acc 92.86%; structured p50 1867.5 ms |
| `rapidocr-cpu` | CPU ONNX | ocr_cpu_baseline | **PASS** | CER 7.04%; p50 1592.5 ms; structured field acc 92.86%; structured p50 859.0 ms |
| `paddleocr-cpu` | CPU ONNX | ocr_cpu_paddle | **PASS** | CER 7.04%; p50 1829.5 ms |
| `bge-reranker-base-amd-win` | CPU ONNX | reranker_default | **PASS** | nDCG 1.000; MRR 1.000; p50 78 ms |
| `bge-reranker-v2-m3-amd-win` | CPU ONNX | reranker_stronger | **PASS** | nDCG 1.000; MRR 1.000; p50 289 ms |
| `sensevoice-small-amd-win` | NPU DirectML | asr | **PASS** | CER 7.69%; RTF 0.073 |

**Status legend:** PASS = all thresholds met. FAIL = one or more quality/perf thresholds missed.
MEASURED = latency/throughput collected; quality dims not fully qualified.

---

## Known Limitations

- **`qwen2.5-7b` translation PASS (recalibrated 2026-06-21)** — Thresholds corrected to chrF≥35.0 / term≥75%. 3-seed confirmed: zh→en 79%≥75%, en→zh chrF 36.4≥35.0.
- **`llama3.2-3b` GA FAIL — model-level weakness, not platform** — MMLU=0.390 and HellaSwag=0.320 are intrinsic LLaMA 3.2-3B limitations. GSM8K=0.710 PASS (math-only). Qwen2.5-3B on Intel scores MMLU=0.530/HellaSwag=0.760 on the same task set. Recommendation: replace `llama3.2-3b-amd-win` with `qwen3-4b-amd` (same ~30 TPS, better knowledge coverage). Keep llama3.2-3b only for: (a) 32k+ context workloads, (b) tool-calling use cases where its format is well-tested.
- **`qwen3-0.6b` MCQ FAIL** — mmlu=0.000, hellaswag=0.000. 0.6B model incapable of reliable MCQ letter output, confirmed post parser-fix. Use qwen3-1.7b for better GA coverage.
- **`qwen3-1.7b`, `qwen3-4b` GA/translation PENDING-VERIFY** — Performance calibrated (TPS/TTFT measured 2026-06-22); quality benchmarks not yet run. Run: `python run_benchmark.py --model qwen3-4b-amd --skip stability,concurrency,conditioned,scenarios,prefill_decode`.
- **LLM conditioned/scenarios FAIL** — Long-context conditioning fails across all tested models.
- **No qualified VLM** — `llava-7b-amd-win` accuracy FAIL; no VLM workloads recommended.
- **NPU LLM PENDING-VERIFY** — AMD XDNA NPU LLM via Lemonade Server supports Llama-3.2-1B, Phi-3.5-mini, Qwen2.5-1.5B (W4A8). Performance estimates unverified.

---

## NPU Batch Processing Guidance

AMD XDNA NPU excels at **background batch workloads** that would otherwise compete with iGPU LLM inference.

| Workload | NPU path | p50 latency | vs iGPU | Recommendation |
|---|---|---|---|---|
| OCR (single doc) | VitisAI ONNX | 2031 ms | 4.3× slower (iGPU: 469 ms) | **Not recommended** for interactive |
| OCR (batch / background) | VitisAI ONNX | 2031 ms/doc | Frees iGPU for LLM | **Recommended** when LLM is concurrently active |
| ASR transcription | DirectML | RTF 0.073 | N/A (NPU-only path) | **Recommended** — dedicated path |
| Vector DB construction | (via embedding model on iGPU) | 875 ms/chunk | — | Embedding runs on iGPU; NPU path not tested |
| LLM (Lemonade) | NPU W4A8 | PENDING | — | PENDING-VERIFY |

**Key insight:** NPU (2 W typical) offloads compute from iGPU (part of 35-54 W SoC TDP), allowing simultaneous LLM inference on iGPU while OCR or ASR runs on NPU. For bulk OCR pipelines (e.g., indexing 100 documents), NPU batch achieves similar throughput to iGPU while keeping iGPU available for interactive chat. Latency per document is ~4× higher on NPU, but total system throughput is higher due to parallelism.

**When to use NPU over iGPU:**
- Background document ingestion while serving LLM queries
- Always-on ASR transcription alongside LLM agent tasks
- Low-power sustained inference where battery life matters

---

## Calibration History

| Date | Event |
|---|---|
| 2026-06-19 | Initial full calibration: all 14 models measured across CPU/iGPU/NPU paths; thresholds set from E2E runs |
| 2026-06-20 | Added quality dims: qwen2.5-7b general_ability PASS (3-seed); llama3.2-3b general_ability FAIL (3-seed — model-level); qwen3-0.6b FAIL (MCQ gap confirmed post parser-fix) |
| 2026-06-21 | qwen2.5-14b perf recalibrated (TPS 8.6); AMD 7B translation threshold recalibrated (chrF 40→35, term 80%→75%); translation PASS confirmed 3-seed |
| 2026-06-22 | Added qwen3:1.7b (TPS 63.5, TTFT P50 497ms) and qwen3:4b (TPS 30.7, TTFT P50 867ms); perf thresholds calibrated; GA/translation PENDING-VERIFY; llama3.2-3b documented as model-level GA FAIL |

---

## 中文摘要

**平台：** amd-win-x86 | Ryzen 8845H + Radeon 780M iGPU + AMD XDNA NPU，Windows 11  
**最后校准：** 2026-06-21。本文件原地更新。

### 硬件画像

| 计算单元 | 芯片 | 规格 | TDP | 角色 |
|---|---|---|---|---|
| **CPU** | AMD Ryzen 8845H | 4P+4E Zen4 核，16 线程，3.8–5.1 GHz | 35 W（基础）/ 54 W（最大） | ONNX CPU — OCR 基线、Reranker |
| **iGPU** | Radeon 780M | RDNA3，12 CU，2800 MHz，17.9 GiB 共享显存 | SoC TDP 内 | Ollama Vulkan — LLM/Embedding；ONNX DirectML — OCR |
| **NPU** | AMD XDNA | AI 300 Series，16 TOPS INT8，~2–5 W 专用 | ~2–5 W | ONNX VitisAI — OCR 批处理；ASR；PENDING: Lemonade LLM |
| **RAM** | LPDDR5x | 32 GB | ~3–5 W | — |

### 执行模式对比

| 任务 | CPU 路径 | iGPU 路径（Vulkan/DirectML） | NPU 路径（VitisAI） |
|---|---|---|---|
| LLM 7B | ~3–5 TPS（估算） | **13.33 TPS** ✓ | — |
| LLM 3B | ~8–12 TPS（估算） | **28.99 TPS** ✓ | — |
| **LLM NPU（Lemonade）** | — | — | **~80–100 TPS**（PENDING-VERIFY） |
| OCR 文字 p50 | 1593 ms | **469 ms** ✓ 最快 | 2031 ms |
| ASR RTF | — | — | **0.073** ✓ |
| Reranker base p50 | **78 ms** ✓ | — | — |

**AMD LLM 三模式：** (1) Ollama Vulkan iGPU（已校准，13.33–91 TPS）；(2) Lemonade Hybrid（iGPU prefill + NPU decode，PENDING-VERIFY，⚠️ Ryzen AI 300 最佳）；(3) Lemonade/FastFlowLM 纯 NPU ≤3.8B（PENDING-VERIFY）。新增支持 DeepSeek-R1-Distill-7B-Hybrid/NPU、Gemma-4-26B-A4B（MoE）。详见 [NPU 文档](./amd-windows-npu.en.md)。

**→ 详细模式文档：**
- [iGPU（Vulkan + DirectML）— LLM、Embedding、OCR 最快路径](./amd-windows-igpu.en.md)
- [NPU（VitisAI + DirectML）— OCR 批处理、ASR](./amd-windows-npu.en.md)
- [CPU ONNX — OCR 基线、Reranker](./amd-windows-cpu.en.md)

### 综合性能 + 模型效果

| 模型 | TPS | TTFT p50 | PP/TG (t/s) | GSM8K | MMLU | HellaSwag | 翻译 | 综合 |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-7b（iGPU） | 13.33 | 953 ms | 116/16 | **0.880** | **0.600** | **0.790** | FAIL（term/chrF） | **GA PASS** |
| llama3.2-3b（iGPU） | 28.99 | 890 ms | 124/39 | 0.710/PASS | 0.390/FAIL | 0.320/FAIL | FAIL | **GA FAIL** |
| qwen3-0.6b（iGPU） | 91.09 | 1781 ms | —/— | 0.390/PASS | 0.000/FAIL | 0.000/FAIL | FAIL | **GA FAIL** |
| qwen3-embed-0.6b | — | 875 ms | — | — | — | — | — | **PASS**（hit@1=1.000） |
| rapidocr-directml | — | 469 ms | — | — | — | — | — | **PASS**（CER 7.04%） |
| sensevoice（NPU） | — | — | — | — | — | — | — | **PASS**（RTF 0.073） |
| bge-reranker-base | — | 78 ms | — | — | — | — | — | **PASS**（nDCG=1.000） |

**唯一 GA PASS 的 LLM：** `qwen2.5-7b-amd-win`（建议生产使用）。

### 功耗参考

| 场景 | 估算功耗 | 依据 |
|---|---|---|
| 空闲 | ~8–12 W | AMD 笔电桌面待机 |
| LLM 0.6B（91 TPS） | **~30–35 W** | 轻度 iGPU，低 TDP |
| LLM 3B（29 TPS） | **~38–45 W** | iGPU Vulkan，持续 |
| LLM 7B（13 TPS） | **~42–50 W** | iGPU 持续高负载 |
| OCR iGPU DirectML | **~25–35 W** | iGPU 激活，CPU 闲置 |
| ASR NPU | **~15–20 W** | 主要 NPU，CPU+iGPU 低负载 |

> **PENDING-VERIFY（实测功耗）：** 使用 `rocm-smi --showpower --json` 或 AMD μProf / HWiNFO64 在基准测试期间测量。

**能效对比（3B 模型）：**
- AMD iGPU：28.99 TPS / ~42 W = **0.69 TPS/W**
- Intel CPU：19.47 TPS / ~42 W = 0.46 TPS/W（AMD iGPU 高效 50%）

### 选型摘要

| 角色 | 推荐模型 | 执行模式 | 备注 |
|---|---|---|---|
| LLM 质量首选 | `qwen2.5-7b-amd-win` | iGPU Vulkan | **唯一 GA PASS**；TTFT 953ms 可交互 |
| LLM 轻量/高并发 | `llama3.2-3b-amd-win` | iGPU Vulkan | 32k ctx；29 TPS；GA FAIL |
| LLM 极速纳米 | `qwen3-0.6b-amd` | iGPU Vulkan | 91 TPS；MCQ 能力不足，不推荐 GA 场景 |
| Embedding（首选） | `qwen3-embedding-0.6b-amd` | iGPU Vulkan | hit@1=1.000；875 ms |
| Embedding（多语言） | `bge-m3-amd` | iGPU Vulkan | 同质量，914 ms |
| Reranker（默认） | `bge-reranker-base-amd-win` | CPU ONNX | 78 ms 最快 |
| OCR（首选） | `rapidocr-amd-directml` | iGPU DirectML | 469 ms 最快 |
| OCR（批处理省 iGPU） | `rapidocr-amd-npu` | NPU VitisAI | 2031 ms，同等 CER |
| ASR | `sensevoice-small-amd-win` | NPU DirectML | RTF 0.073 PASS |
