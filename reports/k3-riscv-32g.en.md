# K3 RISC-V 32G Platform — Model Selection & Benchmark Report

**Platform:** k3-riscv-32g | SpacemiT K3 X100, 32GB RAM, Bianbu Linux
**Source scope:** SpacemiT official [AI SDK](https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/application_tools/ai-sdk.md) for invocation/application routes and official [ModelZoo](https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/compute_stack/ai_compute_stack/modelzoo.md) for performance baselines, checked 2026-07-07; plus public ModelZoo archive artifacts requested for Qwen3.5/Qwen3 VLM, Qwen3/Qwen3.5/LFM2 LLM, qwen3-ASR, PP-OCRv5, embedding, and reranker coverage; and an external HF GGUF+mmproj Qwen3.5-35B-A3B multimodal MoE candidate.
**Last calibrated:** 2026-07-08. This file is updated in place.

---

## Hardware Profile

| Compute Unit | Chip / Runtime | Specs | Power | Role |
|---|---|---|---|---|
| **CPU / RVV / IME** | SpacemiT K3 X100 | riscv64, 32GB RAM, 8-thread benchmark runs | PENDING real board power | GGUF LLM/VLM text path through SpacemiT private `llama.cpp` and upstream K3 `llama.cpp` fallback |
| **SpacemiT ORT EP** | `spacemit-onnxruntime` / `spacemit-tcm` | ONNX Runtime provider `spacemit`; TCM must be free before ORT/SMT runs | PENDING | Official ModelZoo ONNX vision path and PP-OCRv5 OCR path |
| **Private llama.cpp** | `llama.cpp-tools-spacemit 0.1.1+6` | `llama-bench`, `llama-server`; SMT media backend for VLM/ASR tar packages | PENDING | Primary K3 model-serving runtime |
| **Source-built A100 runtimes** | SpacemiT `llama.cpp` + SpacemiT ORT | Cross-built with SpacemiT toolchain v1.2.4; deployed only for system/source comparison | PENDING | Optimization baseline after source/system performance-equivalence gate |
| **Upstream K3 llama.cpp** | Local K3 build | Used only when private runtime cannot load a GGUF | PENDING | Compatibility control for Qwen3.6-35B |
| **Storage policy** | Local cache + K3 working cache | Canonical model artifacts stay under `drivers/spacemit-ai/model_zoo`; K3 keeps only hot model copies after tests | — | Prevents K3 root fs exhaustion during broad ModelZoo sweeps |

---

## Official Reference Baseline

Application invocation follows the SpacemiT AI SDK page updated `2026-06-30 20:04:26`: LLM can be integrated through `llama-server` plus `llm_chat` or gateway `POST /v1/chat/completions`; VLM can be integrated through SDK demos or gateway `POST /v1/vlm/models/load` and `POST /v1/vlm/chat/completions`; ASR and vision use their component demos or gateway routes. This report's raw benchmark scripts still call direct llama-server/ORT endpoints so the same model artifact can be retested reproducibly.

Official performance baselines follow the SpacemiT ModelZoo page updated `2026-06-09 18:06:37`: vision rows use `onnxruntime_perf_test`; LLM rows use the ModelZoo `llama-bench` command with `-p 128 -n 128 -mmp 0 -fa 1 -ub 128`; VLM/ASR rows use the SMT llama-server path. Embedding, reranker, and PP-OCRv5 OCR conclusions below are local measurements because the cited ModelZoo page does not publish those rows.

---

## Official Baseline Alignment Gate

Retest rule: for throughput rows, local data must be at least 95% of the official ModelZoo value under the same artifact and command. For latency/RTF rows, local data must be no worse than 105% of the official value under the same artifact and command. If artifact, quantization, command, input, or metric differs, the row is **RETEST_REQUIRED** and must not be used as an official-baseline claim.

| Area | Current alignment | Finding | Action |
|---|---|---|---|
| Official ONNX vision | **ALIGNED** | 132/132 K3 rows retested with `onnxruntime_perf_test`; no row is >5% slower than official. Worst observed ratio is about 1.009x on `yolo12n` 1-core. | No full retest needed unless SpacemiT updates ModelZoo or runtime changes. |
| LLM ModelZoo rows | **ALIGNED** | 8/8 official rows passed the exact `llama-bench -p 128 -n 128 -mmp 0 -fa 1 -ub 128` retest in `output/reports/k3-riscv-32g/official-modelzoo-llm-20260706_185656/alignment-summary.tsv`. The aligned run left `PRIVATE_ENV` empty so TCM stayed enabled and released TCM before every model. | Keep `scripts/run_k3_32g_official_modelzoo_llm_retest.sh` as the reproducible path. Do not use `SPACEMIT_DISABLE_TCM=1` for ModelZoo baseline claims. |
| VLM VisionEncoder rows | **ALIGNED_PROBE** | 10/10 official 4/8-core VisionEncoder rows passed the 105% latency gate in `output/reports/k3-riscv-32g/official-modelzoo-vlm-encoder-20260706_193214/results.tsv`. The probe uses `onnxruntime_perf_test -e spacemit` on each tar package's vision ONNX because the ModelZoo page publishes the server startup command but not a standalone encoder benchmark command. | Keep encoder latency and end-to-end document extraction conclusions separate. |
| qwen3-ASR 0.6B | **PARTIAL / RETEST_REQUIRED** | Local broad run has RTF p50 0.168 vs official 0.186, but p95 is 0.513 and the input/metric aggregation differs. | Retest exact official qwen3-ASR path/input. If p50/mean official-style RTF exceeds 0.195, run full ASR retest. |
| sensevoice | **RETEST_REQUIRED** | Official ModelZoo publishes sensevoice RTF, but current K3 result is package inspection only. | Find/confirm K3 serving or CLI wrapper, then run official-style ASR retest. |
| OCR / embedding / reranker | **LOCAL_ONLY** | The cited ModelZoo page does not publish OCR, embedding, or reranker baseline rows. Current data is product-local coverage only. | Do not compare to official baselines. Retest only against local quality/latency gates; BGE-En remains failed due invalid vectors. |
| Power and scheduler/gateway stress | **NOT_ALIGNED** | No official ModelZoo baseline; current raw model-server tests do not exercise the scheduler/gateway required by `docs/k3-realistic-stress-plan.md`. | Keep as product risk workstream, not ModelZoo baseline alignment. |

Exact LLM baseline retest wrapper: run
`scripts/run_k3_32g_official_modelzoo_llm_retest.sh` with the K3 connection
values supplied through the local secure environment. Do not record host,
account, or password values in reports.

TCM control: a qwen3-0.6B run with `SPACEMIT_DISABLE_TCM=1` measured about
PP128/TG128 `437/37` token/s; the aligned run with TCM enabled measured about
`502/53` token/s. Treat the disabled-TCM result as invalid for official
ModelZoo alignment.

---

## Source Runtime Equivalence Gate

SpacemiT source runtime work follows the public cross-compiler guide plus the
SpacemiT `llama.cpp` and `onnxruntime` source trees. Local artifacts are stored
under `builds/spacemit-a100/`; dependency mirrors stay under `drivers/`.

### `llama.cpp` source vs system

Gate: source/system TPS must be at least 0.95. Result directory:
`output/reports/k3-riscv-32g/source-runtime-compare-20260707_101930/`.

| Model | Test | System TPS | Source TPS | Source/System | Status |
|---|---:|---:|---:|---:|---|
| `qwen3-0.6B` | pp128 | 497.11 | 499.41 | 1.005 | PASS |
| `qwen3-0.6B` | tg128 | 52.58 | 54.21 | 1.031 | PASS |
| `qwen3-4B` | pp128 | 76.81 | 76.16 | 0.992 | PASS |
| `qwen3-4B` | tg128 | 10.85 | 10.98 | 1.012 | PASS |
| `qwen3-30B-A3B` | pp128 | 55.50 | 55.23 | 0.995 | PASS |
| `qwen3-30B-A3B` | tg128 | 12.28 | 12.50 | 1.017 | PASS |

### ORT source vs system

Gate: source/system average latency must be no worse than 1.05. Result
directory:
`output/reports/k3-riscv-32g/source-runtime-compare-20260707_111859/`.

| Model | Core | System ms | Source ms | Source/System | Status |
|---|---:|---:|---:|---:|---|
| `fastvlm-vision` | 8 | 158.942 | 155.254 | 0.977 | PASS |
| `qwen3vl30b-vision` | 8 | 4623.870 | 4653.055 | 1.006 | PASS |

Conclusion: the source-built `llama.cpp` and ORT artifacts are
performance-equivalent to the system SpacemiT runtimes on these K3/A100 rows.
They can be used as the optimization baseline, with the same source/system gate
kept for every future runtime change.

---

## Execution Mode Comparison

All values are measured on K3 32GB unless explicitly marked as official baseline.

| Workload | Private llama.cpp / GGUF path | Upstream K3 llama.cpp path | SpacemiT ORT / SMT media path |
|---|---|---|---|
| **LLM 35B MoE, Qwen3.6** | **PASS as LLM / FAIL as VLM**: current system runtime loads `Qwen3.6-35B-A3B-UD-Q4_K_XL`; PP512 30.49 tok/s, TG128 6.75 tok/s, 1K/3K needle PASS; model reports `vision_backend=none` and image input returns HTTP 500 unsupported | **MEASURED legacy fallback**: PP512 20.87 tok/s, TG128 6.57 tok/s; 1K/4K PASS, 8K/32K timeout | — |
| **LLM 30B MoE, Qwen3** | **PASS with limits**: PP512 33.69 tok/s, TG128 9.80 tok/s; GA PASS; realistic 1K prompt takes 223.834s and 30.488GiB RSS | not needed | — |
| **LLM 35B MoE, Qwen3.5** | **PASS as LLM / FAIL as VLM**: current system runtime loads `Qwen3.5-35B-A3B-Q4_0`; PP512 29.31 tok/s, TG128 6.48 tok/s, 1K/3K needle PASS; model reports `vision_backend=none` and image input returns HTTP 500 unsupported | not needed | — |
| **VLM 35B MoE, Qwen3.5 external GGUF+mmproj** | **PASS smoke as single LLM+VLM service**: `Qwen3.5-35B-A3B-Q4_K_M.gguf` + `mmproj-F16.gguf` loads with `capabilities=["completion","multimodal"]`, `vision_backend=mtmd`; no-think API TTFT 1.118s, decode128 29.192s, 1K needle PASS 60.104s, receipt-title image PASS 8.489s | not tested | — |
| **LLM 24B MoE, LFM2** | **PASS(perf)**: PP512 55.48 tok/s, TG128 15.36 tok/s; quality matrix not fully qualified | not needed | — |
| **VLM SMT tar models** | **PASS with quality split**: Qwen3.5-2B 30/30 doc cases; qwen30ba3b-mm 29/30; smaller models mixed | — | SMT media backend required for tar VLM/ASR packages |
| **VLM GGUF+mmproj** | **PASS quality control**: Qwen3VL-4B + mmproj 30/30 doc cases, but slow; SmolVLM runtime only and quality FAIL | — | — |
| **Official ONNX vision** | — | — | **PASS**: 33 official ONNX models x 1/2/4/8 core settings = 132/132 PASS after stale TCM release |
| **OCR** | — | — | **PASS line OCR**: PP-OCRv5 `vision/ppocr` det+rec ONNX, 72-sample CER 0.0039 |
| **ASR** | qwen3-ASR 0.6B/1.7B SMT audio paths PASS | — | SMT media backend; 0.6B is default |
| **Embedding / Reranker** | BGE-Zh/Nomic/Qwen embedding PASS; BGE reranker default; Qwen reranker slow/offline only | — | BGE-En embedding returns invalid vectors on this runtime |

**Mode details:**
- CPU / RVV / IME evidence: `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-cpu.en.md`
- SpacemiT ORT evidence: `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-spacemit-ort.en.md`

---

## Comprehensive Performance + Quality Profile

### LLM Performance

Official ModelZoo LLM rows use `llama-bench -p 128 -n 128 -mmp 0 -fa 1 -ub 128`. Local rows add PP512/TG128, endpoint probes, context checks, resource/RSS observations, and selected quality suites.

| Model | Official PP128 | Official TG128 | Local PP512 | Local TG128 | Endpoint / TTFT | Max practical context | Status |
|---|---:|---:|---:|---:|---|---|---|
| `Qwen3-30B-A3B-Q4_0` | 55.67 | 12.32 | 33.69 | 9.80 | TTFT p50/p95 672/2354ms; realistic retest TTFT 0.866s, decode128 16.479s | 1K PASS but 223.834s and 30.488GiB RSS; 3K sync manually aborted; 32K impractical | **PASS with strict long-context limits** |
| `Qwen3.5-35B-A3B-Q4_0` | — | — | 29.31 | 6.48 | no-think API PASS: TTFT 1.120s, decode128 28.105s | 1K PASS 37.263s; 3K PASS 97.110s; VLM image input unsupported | **PASS LLM, NOT VLM** |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL` | — | — | 30.49 | 6.75 | TTFT 1.017s; decode128 20.277s | 1K PASS 34.741s; 3K PASS 82.782s; VLM image input unsupported | **PASS LLM, NOT VLM** |
| `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | — | — | — | 4.57* | no-think API TTFT 1.071s; decode128 29.374s | 1K PASS 58.824s; 3K PASS 297.122s at 4K ctx; VLM image input supported | **PASS single-service, long-context async only** |
| `LFM2-24B-A2B-Q4_0` | — | — | 55.48 | 15.36 | TTFT p50 174ms; aggregate 14.5 TPS | 1K/3K high-spec probe PASS | **PASS(perf), quality mixed** |
| `Qwen3-8B-Q4_K_M` | — | — | 25.97 | 4.24 | TTFT p50 281ms; aggregate 4.3 TPS | 1K/4K PASS but slow | **PASS(perf), quality mixed** |
| `Qwen3-4B-Q4_K_M` | 76.44** | 11.03** | 42.14 | 7.30 | TTFT 0.63s; decode128 17.26s | 1K/3K PASS | **PASS smoke** |
| `Qwen3.5-2B-Q4_0` | 112.22*** | 16.15*** | 87.02 | 14.47 | TTFT 0.35s; decode128 9.60s | 1K/3K PASS | **PASS smoke** |
| `Qwen3-0.6B-Q4_0` | 499.75 | 53.35 | 198.51 | 37.53 | TTFT 0.10s; decode128 3.12s | 1K/3K PASS | **PASS smoke** |

`*` Legacy upstream K3 `llama.cpp` fallback result is superseded for current system-runtime LLM serving, but remains compatibility evidence.
`**` Official row is Q4_0; local archive run used Q4_K_M.
`***` Official table labels Qwen3.5-2B as Q4_1, but the linked/file-tested artifact is Q4_0.
`*` For the external Qwen3.5-35B-A3B multimodal candidate, decode speed is from
OpenAI-compatible server timing/logs, not `llama-bench`; run the official bench
wrapper before using it as a ModelZoo-style throughput claim.

### LLM Quality Scores

| Model | GSM8K | MMLU | HellaSwag | Translation | Conditioned / Context | Scenario / Drift | Verdict |
|---|---:|---:|---:|---|---|---|---|
| `Qwen3-30B-A3B-Q4_0` | 0.950 PASS | 0.750 PASS | 0.850 PASS | PASS | 1K/3K correct but very slow; cache behavior caveat | scenario judge unavailable; adversarial stability weak | **Best K3 LLM, bounded use only** |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL` | not full GA | not full GA | not full GA | not full matrix | 1K/3K no-think API PASS; image input unsupported | not full matrix | **LLM candidate only** |
| `Qwen3.5-35B-A3B-Q4_0` | not full GA | not full GA | not full GA | not full matrix | 1K/3K no-think API PASS; image input unsupported | not full matrix | **LLM control only** |
| `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | not full GA | not full GA | not full GA | not full matrix | no-think API 1K/3K needle PASS; default thinking returns reasoning-only final-empty output | VLM doc 29/30, field acc 0.9942 | **Single-model candidate; async-only for VLM/long text** |
| `LFM2-24B-A2B-Q4_0` | BLOCKED | BLOCKED | BLOCKED | FAIL terminology threshold | 1K high-spec PASS; standard 3K rejected in one run | FAIL/DRIFT | **Perf PASS, GA not qualified** |
| `Qwen3-8B-Q4_K_M` | BLOCKED | BLOCKED | BLOCKED | WARN synthetic fallback | 1K/4K PASS but slow | FAIL/DRIFT | **Perf PASS, GA not qualified** |
| Smaller official/archive LLMs | smoke only | smoke only | smoke only | smoke/mixed | 1K/3K smoke mixed | not full matrix | **Smoke coverage only** |

### Long-Context / Aviation Manuals

The long-context suite now includes
`shiroinekotfs/airplane-manual-collection`, cached as extracted PDF text and
windowed benchmark cases. The current core cache generated 15 cases from 5
usable aviation manuals.

| Model / context | Cases | Quality | Latency / resource | Verdict |
|---|---:|---|---|---|
| `Qwen3-4B-Q4_0`, aviation 1K window | 1 | PASS, score 1.0 | 175.453s E2E; 1263 prompt tokens; prefill 7.74 tok/s; RSS about 5GB | **Pipeline PASS, sync UX FAIL** |
| `Qwen3-4B-Q4_0`, aviation 3K window | 0 completed | not scored | first request exceeded 5 minutes before completion; server log reached 1024 prompt tokens at 9.51 tok/s before cancellation | **Direct 3K sync not acceptable** |
| `Qwen3-0.6B-Q4_0`, aviation 1K naive window | 6 attempted, 4 measured | measured score 0.75; 2 context-overflow errors | mean 62.747s; p95/max 74.659s | **Estimator FAIL, data useful** |
| `Qwen3-0.6B-Q4_0`, aviation 1K safety-budget window | 6 | FAIL overall, score 0.5958; no context-overflow errors | mean 57.438s; p95/max 143.945s; prompts 796-1911 tokens | **Stable API, quality below gate** |
| `Qwen3.5-35B-A3B-Q4_0`, aviation run | not rerun in this pass | not scored | current 2026-07-07 pass covered LLM API plus image-input probe only | **Separate long-context retest required if selected** |

Conclusion: aviation manuals meet the project need for real long-document
coverage, but K3 32G must not ingest whole manuals directly. The product path is
offline text/OCR extraction, embedding index, bounded reranker top-k, and a
small cited evidence window sent to the LLM. Character-count clipping is not
sufficient because it undercounted the serving tokenizer; use tokenizer-aware or
conservative adaptive clipping after retrieval. 0.6B is useful for pipeline smoke
tests but not for answer quality. 4B is acceptable only for triage; 30B/35B and
high-quality VLM checks must be async with queueing, TTL, cancellation, and
memory admission control.

### VLM / Multimodal

| Model | Vision path | TTFT | 1K context | Receipt image | Doc extraction cases | Field acc | JSON parse | Latency avg / p95 | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `Qwen3.5-2B.tar.gz` | SMT | 0.391s | PASS, 13.726s | 8.439s | 30/30 | 1.0000 | 0.9667 | 10.798s / 12.239s | **Best practical sync VLM** |
| `Qwen3VL-4B-Instruct-Q4_K_M + mmproj` | mtmd | 0.532s | PASS, 26.846s | 38.266s | 30/30 | 1.0000 | 1.0000 | 38.680s / 44.307s | **Best quality control, slow** |
| `qwen30ba3b-mm-q4_1.tar.gz` | SMT | 0.714s | PASS, 34.883s | 48.915s | 29/30 | 0.9942 | 1.0000 | 48.026s / 51.188s | **High-spec, very slow** |
| `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | mtmd | 1.058s | PASS, 58.824s | 8.009s | 29/30 | 0.9942 | 1.0000 | 68.822s / 78.774s | **Quality PASS, async-only single-model candidate** |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL` with `image_url` | none | 1.017s text TTFT | 1K/3K text PASS | HTTP 500 unsupported | 0 | — | — | immediate fail | **NOT VLM** |
| `Qwen3.5-35B-A3B-Q4_0` with `image_url` | none | 1.120s text TTFT | 1K/3K text PASS | HTTP 500 unsupported | 0 | — | — | immediate fail | **NOT VLM** |
| `Qwen3.5-0.8B.tar.gz` | SMT | 0.274s | PASS, 9.540s | 7.513s | 18/30 | 0.8728 | 0.5667 | 7.269s / 9.844s | **PARTIAL** |
| `Qwen3.5-4B.tar.gz` | SMT | 0.792s | PASS, 31.668s | 21.334s | 8/30 | 0.7977 | 1.0000 | 21.413s / 24.575s | **FAIL doc suite** |
| `fastvlm-mm-0.5b-q4_1.tar.gz` | SMT | 0.081s | FAIL recall | 3.213s | 0/30 | 0.4451 | 0.2333 | 4.243s / 4.789s | **FAIL quality** |
| `SmolVLM-256M + mmproj` | mtmd | 0.198s | FAIL HTTP400 | 41.426s | 0/30 | 0.1850 | 0.0000 | 42.388s / 44.856s | **FAIL quality** |

Realistic-control retest on `Qwen3VL-4B + mmproj` repeated 10 document cases with 10/10 pass and field accuracy 1.0, but latency increased to avg 63.979s and p95 78.064s. Use it as a quality-control or async model, not a synchronous document-upload default.

External Qwen3.5-35B-A3B multimodal MoE results:
`output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-smoke-official-20260707_190232/`
confirmed the model loads as `multimodal`, but default thinking mode produced
reasoning-only responses with empty final `content` for text probes.
`output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-thinkingoff-official-20260707_190728/`
set `CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}'` and passed text,
1K needle recall, and a receipt-title image probe. Load-to-ready was about
2m05s at 2K context. The full document run in
`output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-docfull-defaultimg-official-20260708_093315/`
used `--parallel 1 --cache-ram 512` and completed 30/30 calls with 29/30 case
pass, 172/173 fields correct, and JSON parse rate 1.0. The only miss was
`vat_invoice/c12` seller. A control run with `--image-min-tokens 1024
--image-max-tokens 1024` was aborted after a single image request stalled with
repeated `non-consecutive token position` warnings; do not use fixed 1024 image
tokens on this runtime until the mtmd path is debugged. The 4K context run in
`output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-context4k-official-20260708_101309/`
passed 1K and 3K needle recall, but the 3K probe took 297.122s with 2777 prompt
tokens. This model can be a single LLM+VLM quality candidate, but document
extraction and long text are async-only with admission control.

Official ModelZoo VisionEncoder probe:

| Model | 4-core local / official ms | 8-core local / official ms | Status |
|---|---:|---:|---|
| `fastvlm-0.5B` | 257.37 / 256.47 | 156.82 / 164.50 | **PASS** |
| `Qwen3-VL-30B-A3B` | 7875.04 / 7928.13 | 4613.13 / 4753.55 | **PASS** |
| `Qwen3.5-0.8B` | 339.94 / 340.42 | 234.66 / 245.61 | **PASS** |
| `Qwen3.5-2B` | 900.35 / 901.56 | 769.25 / 794.03 | **PASS** |
| `Qwen3.5-4B` | 898.31 / 904.73 | 768.92 / 798.71 | **PASS** |

### Non-LLM Performance

ModelZoo publishes official ASR and generic vision rows. OCR, embedding, and reranker rows here are local K3 32G measurements against public archive artifacts and are not official ModelZoo performance rows.

| Model | Role | Latency / Resource | Key Metric | Status |
|---|---|---|---|---|
| `PP-OCRv5_mobile_det+rec.onnx` | OCR | 72-line broad run p50 2372.3ms, p95 2985.5ms | CER 0.0039, NED avg 0.0035 | **PASS line OCR** |
| `qwen3-asr-0.6B.tar.gz` | ASR | RTF p50/p95 0.168/0.512; RSS 1.812GiB | 12 samples, 8 scored; normalized CER avg 0.0192 | **PASS default** |
| `qwen3-asr-1.7B-dynq-q4km.tar.gz` | ASR | RTF p50/p95 0.358/1.486; RSS 3.570GiB | same normalized CER avg 0.0192 | **PASS but not default** |
| `Bge-Small-Zh-V1.5-Q4_K_M` | Embedding | p50 5.26ms, p95 5.85ms; batch64 6.26ms/text; RSS 0.068GiB | overall Hit@1 0.9722; finite vectors | **PASS default** |
| `Nomic-Embed-Text-V2-Moe-Q4_0` | Embedding | p50 19.41ms, p95 21.21ms; batch64 31.25ms/text | overall Hit@1 0.9722; finite vectors | **PASS alternate** |
| `Qwen3-Embedding-0.6B-Q4_0` | Embedding | p50 40.49ms, p95 45.27ms; batch64 101.23ms/text | overall Hit@1 0.9722; finite vectors | **PASS slower alternate** |
| `Jina-Embeddings-V5-Text-Small-Retrieval-Q4_K_M` | Embedding | p50 46.87ms, p95 57.12ms; batch64 125.47ms/text | overall Hit@1 0.8333; zh Hit@1 0.6667 | **PASS, not zh default** |
| `Bge-Small-En-V1.5-Q4_K_M` | Embedding | service starts; output invalid | finite vector ratio 0.0; retrieval Hit@1 0.0 | **FAIL** |
| `Bge-Reranker-V2-M3-Q4_0` | Reranker | top3 p95 308ms; top10 727ms; top20 1333ms; top50 3467ms; RSS 0.802GiB | Hit@1 1.0 through top50 | **PASS default** |
| `Qwen3-Reranker-0.6B-Q4_0` | Reranker | top3 p95 1370ms; top10 3910ms; top20 7379ms; top50 18804ms | top50 Hit@1 drops to 0.8333 | **PASS slow/offline only** |
| `mineru2.5-pro-2605-1.2B-original` | Document parsing | package inspected and extracted | no confirmed K3 serving wrapper in current SpacemiT runtime | **PACKAGE PASS / E2E PENDING** |
| `sensevoice.tar.gz` | ASR package | official K3 RTF available in ModelZoo | no local standalone CLI/serving wrapper found | **PACKAGE PASS / E2E PENDING** |

---

## Power Consumption

### Chip TDP Reference

| Component | Reference | Current status | Role |
|---|---|---|---|
| SpacemiT K3 X100 SoC | board/input power not sampled in this pass | **PENDING-VERIFY** | LLM/VLM/embedding/reranker GGUF runtime and ORT/SMT media workloads |
| 32GB system memory | unified memory | measured through RSS only | Long-context LLM risk control |
| TCM / SpacemiT ORT path | allocation state available through `spacemit-tcm-smi` | measured before/after runs | Official ONNX vision and PP-OCRv5 |

Real board power was not sampled during this pass. Do not compare K3 TPS/W with AMD/Intel reports until board/input power is collected during sustained `llama-bench`, `llama-server`, `onnxruntime_perf_test`, VLM, OCR, ASR, embedding, and reranker workloads.

### Estimated Power Under Inference

| Scenario | Current status | Required next measurement |
|---|---|---|
| Qwen3-30B short decode | performance and RSS measured | board/input power during 10-minute decode |
| Qwen3-30B long-context prefill | latency and RSS risk measured | board/input power plus scheduler queue impact |
| Qwen3.5-2B VLM | document-extraction quality measured | board/input power during 30-case VLM suite |
| PP-OCRv5 line OCR | 72-sample OCR measured | board/input power during batch OCR |
| qwen3-ASR 0.6B | RTF and RSS measured | board/input power during audio batch |
| Embedding/reranker | latency and RSS measured | board/input power during top-k sweep |

### Power Efficiency Comparison

| Workload | Throughput / Latency | Power | Efficiency status |
|---|---:|---|---|
| `Qwen3-30B-A3B-Q4_0` short decode | decode128 16.479s; TG128 9.80 tok/s | not sampled | **PENDING** |
| `Qwen3.5-2B.tar.gz` VLM | doc p95 12.239s | not sampled | **PENDING** |
| `PP-OCRv5_mobile_det+rec.onnx` | OCR p95 2985.5ms | not sampled | **PENDING** |
| `qwen3-asr-0.6B.tar.gz` | RTF p95 0.512 | not sampled | **PENDING** |
| `Bge-Small-Zh-V1.5-Q4_K_M` | embedding p95 5.85ms | not sampled | **PENDING** |
| `Bge-Reranker-V2-M3-Q4_0` | top20 p95 1333ms | not sampled | **PENDING** |

---

## Selection Summary

| Role | Selected Model | Execution mode | Rationale |
|---|---|---|---|
| LLM primary | `Qwen3-30B-A3B-Q4_0` | private llama.cpp GGUF | Best K3 full-GA coverage; bounded short prompts pass; long-context sync use is high risk |
| LLM large candidate | `Qwen3.6-35B-A3B-UD-Q4_K_XL` | current system llama.cpp GGUF | Strong LLM serving path: PP512 30.49 tok/s, TG128 6.75 tok/s, 1K/3K needle PASS; not a VLM |
| LLM large API control | `Qwen3.5-35B-A3B-Q4_0` | current system llama.cpp GGUF | no-think API retest passes PP512/TG128, TTFT/decode/1K/3K; not full GA; not a VLM |
| LLM throughput control | `LFM2-24B-A2B-Q4_0` | private llama.cpp GGUF | Strongest large-model throughput; quality not fully qualified |
| VLM practical default | `Qwen3.5-2B.tar.gz` | SMT media backend | Best latency/quality balance: 30/30 doc cases, p95 12.239s |
| VLM quality control | `Qwen3VL-4B + mmproj` | GGUF+mmproj mtmd | 30/30 quality, but slow; async/high-quality path only |
| OCR default | `PP-OCRv5_mobile_det+rec.onnx` | SpacemiT ORT / CPUExecutionProvider in local run | Dedicated OCR route from `vision/ppocr`; 72-line CER 0.0039 |
| ASR default | `qwen3-asr-0.6B.tar.gz` | SMT media backend | Same normalized CER as 1.7B on broad set with lower RTF/RSS |
| Embedding default | `Bge-Small-Zh-V1.5-Q4_K_M` | GGUF embedding server | Fastest usable broad result; overall Hit@1 0.9722, p95 5.82ms |
| Embedding alternate | `Nomic` / `Qwen3-Embedding-0.6B` | GGUF embedding server | Similar broad quality but slower; use when quality requirements justify latency |
| Reranker default | `Bge-Reranker-V2-M3-Q4_0` | GGUF reranker server | Hit@1 1.0 through top50; keep interactive top-k <=20 |
| Reranker slow/offline | `Qwen3-Reranker-0.6B-Q4_0` | GGUF reranker server | Too slow for default use; top50 p95 18.67s and quality drops |
| Document parsing | MinerU package | package inspection only | Package exists; current K3 runtime path not qualified for E2E serving |

---

## Full Model Results

| Area | Evidence |
|---|---|
| LLM full archive pass | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-llm-full-20260704.md` |
| VLM full archive pass | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-vlm-full-20260704.md` |
| Embedding/Reranker/ASR/OCR pass | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-nonllm-full-20260704.md` |
| Qwen3.6 private package analysis | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-qwen36-spacemit-private-20260704.md` |
| Qwen3-30B focused run | `reports/runs/k3-riscv-32g/20260704/k3-riscv-32g-qwen30b-20260704.md` |
| Official vision EP full matrix | `output/reports/k3-riscv-32g/vision-official-20260704_195025/results.tsv` |
| VLM 30-case document extraction | `output/reports/k3-riscv-32g/vlm-full-20260706_0955_allcases/` |
| Qwen3.5-35B no-think API retest | `output/reports/k3-riscv-32g/qwen35-35b-nothink-api-20260706_140027/` |
| Qwen3.5-35B LLM plus image-input probe | `output/reports/k3-riscv-32g/qwen35-35b-llm-vlm-20260707/` |
| Realistic workflow control | `output/reports/k3-riscv-32g/realistic-stress-combined-20260706_150439/` |
| Non-LLM broad coverage retest | `output/reports/k3-riscv-32g/nonllm-broad-20260706_154340/` |
| Non-LLM broad coverage rerun | `output/reports/k3-riscv-32g/nonllm-broad-20260706_190649/` |
| Official VLM VisionEncoder probe | `output/reports/k3-riscv-32g/official-modelzoo-vlm-encoder-20260706_193214/` |
| Source-built `llama.cpp` equivalence | `output/reports/k3-riscv-32g/source-runtime-compare-20260707_101930/` |
| Source-built ORT equivalence | `output/reports/k3-riscv-32g/source-runtime-compare-20260707_111859/` |
| Qwen3.6 LLM plus image-input probe | `output/reports/k3-riscv-32g/qwen36-35b-llm-vlm-20260707/` |
| Aviation-manual long-context cache | `drivers/long-context-suites/airplane-manual-collection/cases/aviation_manual_cases.jsonl` |
| Aviation-manual 1K closed-loop run | `output/reports/k3-riscv-32g/long-context-aviation-qwen3-4b-1k-20260707_113324/` |
| Aviation-manual 3K sync-risk run | `output/reports/k3-riscv-32g/long-context-aviation-qwen3-4b-20260707_112710/` |
| Aviation-manual 0.6B naive-window retest | `output/reports/k3-riscv-32g/long-context-aviation-qwen3-06b-1k-20260707_115623/` |
| Aviation-manual 0.6B safety-budget retest | `output/reports/k3-riscv-32g/long-context-aviation-qwen3-06b-1k-safe-20260707_120251/` |

---

## Known Limitations

- **No scheduler/gateway was active during these runs.** Raw OpenAI-compatible servers cannot prove `/capacity`, queue wait, async job fairness, cancellation, TTL, or backpressure required by `docs/k3-realistic-stress-plan.md`.
- **Qwen3-30B long-context serving is high risk on 32GB.** The 1K realistic needle already took 223.834s with 30.488GiB RSS; 3K sync was manually aborted after prolonged full-load wait.
- **Aviation-manual long text must be retrieval-windowed and tokenizer-aware.** A 1K aviation window on Qwen3-4B scored 1.0 but took 175.453s; a 3K window did not complete within the observation window. Qwen3-0.6B safety-budget windows avoided overflows but scored only 0.5958. Direct manual ingestion is not a sync product path.
- **PP-OCRv5 is qualified for line OCR, not full document layout assembly.** Broad coverage includes 72 line samples with fonts, low contrast, noise, and slight rotation; robust multi-line detector postprocess and layout reconstruction remain open.
- **BGE-En embedding is not usable on this K3 runtime.** It starts but returns invalid/non-finite vectors.
- **Qwen3.6-35B is not a single-model LLM+VLM solution.** The current system runtime serves it as a text model, and `/v1/models` reports `vision_backend=none`; an `image_url` chat request returns unsupported-image HTTP 500.
- **Qwen3.5-35B is not a single-model LLM+VLM solution.** The current system runtime serves it as a text model, and `/v1/models` reports `vision_backend=none`; an `image_url` chat request returns unsupported-image HTTP 500.
- **Qwen3 reranker is too slow for online default use.** BGE reranker should be the default; top-k must be bounded.
- **TCM state matters.** Stale TCM occupancy caused earlier ORT/SMT failures; every ORT/VLM/ASR run must log TCM before and after.
- **K3 root fs must be treated as a working cache.** Canonical artifacts stay in local `drivers/spacemit-ai/model_zoo`; tested non-hot K3-side model copies were deleted after evidence capture.
- **Power and TPS/W are not yet validated.** Board-level power sampling is still missing.

---

## K3 Workflow Guidance

| Workflow | Recommended path | Risk control |
|---|---|---|
| Realtime RAG | BGE-Zh embedding -> top-k retrieval -> BGE reranker -> bounded Qwen3-30B answer | Keep reranker interactive top-k <=20; cap LLM tokens and context |
| Document upload, sync | PP-OCRv5 for text OCR; Qwen3.5-2B only when visual reasoning is needed | Do not route ordinary OCR through VLM; keep VLM timeout visible to user |
| Document upload, async/high quality | Qwen3VL-4B or qwen30ba3b-mm as background jobs | Requires queue, TTL, cancellation, and status API |
| ASR | qwen3-asr-0.6B SMT path | Normalize Traditional/Simplified Chinese and Chinese numerals before scoring |
| Long-context LLM | offline extract/OCR -> embedding -> reranker -> cited evidence window -> async Qwen3-30B/35B verifier | No whole-manual ingestion; mandatory admission control, queue isolation, TTL, cancellation, and memory guard |
| Batch model sweeps | local `drivers/spacemit-ai/model_zoo` as canonical cache; K3 as working cache | Delete non-hot K3 copies after evidence capture |

### Summary: Can K3 32G handle Embedding / Reranker / OCR / ASR?

| Task | Recommended model | Online verdict | Main risk |
|---|---|---|---|
| Embedding | `Bge-Small-Zh-V1.5-Q4_K_M` | **YES** | Use BGE-Zh as default; BGE-En returns invalid vectors on this runtime |
| Reranker | `Bge-Reranker-V2-M3-Q4_0` | **YES with top-k bound** | top50 is usable but slower; interactive paths should prefer top20 or lower |
| OCR | `PP-OCRv5_mobile_det+rec.onnx` | **YES for line OCR** | Full layout reconstruction and detector postprocess still need product hardening |
| ASR | `qwen3-asr-0.6B.tar.gz` | **YES** | Normalize Chinese variants/numerals; 1.7B has no accuracy gain in this run |
| VLM | `Qwen3.5-2B.tar.gz` sync, `Qwen3VL-4B + mmproj` async | **YES with latency class split** | Do not use VLM as OCR replacement; high-quality VLM is too slow for default sync |
| LLM | `Qwen3-30B-A3B-Q4_0` | **YES for bounded short requests** | Long-context sync requests approach memory and latency limits |
| Aviation manuals | OCR/text extraction + BGE-Zh embedding + BGE reranker + tokenizer-aware cited LLM window | **YES only as RAG/async workflow** | 1K Qwen3-4B window takes 175s; 0.6B safety window is stable but below quality gate; 3K direct window is not sync-acceptable |

---

## Calibration History

| Date | Event |
|---|---|
| 2026-07-04 | Qwen3.6 upstream fallback and private-runtime failure captured; Qwen3-30B, LFM2-24B, Qwen3-8B, VLM, embedding, reranker, ASR/OCR package coverage completed. |
| 2026-07-04 | Official ONNX vision EP full matrix completed: 33 models x 4 core settings = 132/132 PASS after stale TCM release. |
| 2026-07-06 09:55-11:44 | Full VLM document-extraction suite completed: Qwen3VL 30/30, Qwen3.5-2B 30/30, qwen30ba3b-mm 29/30, Qwen3.5-0.8B 18/30, smaller controls failed quality. |
| 2026-07-06 14:00-14:04 | Qwen3.5-35B no-think API retest completed: TTFT 1.024s, decode128 22.119s, 1K/3K needle PASS. |
| 2026-07-06 14:50-15:19 | Realistic-control retest completed: Qwen3-30B short/decode/1K; Qwen3VL-4B 10/10 doc cases; PP-OCRv5, qwen3-ASR, BGE-Zh embedding, BGE reranker. Scheduler/gateway absent, so queueing/admission risk remains open. |
| 2026-07-06 15:43-15:58 | Non-LLM broad coverage completed: 72-sample OCR; qwen3-ASR 0.6B/1.7B perturbations; five embedding specs; two reranker specs at top3/10/20/50. |
| 2026-07-06 18:56-19:04 | Official LLM ModelZoo retest rerun with TCM enabled: 8/8 PP128/TG128 rows aligned. Disabled-TCM control was invalid for official baseline claims. |
| 2026-07-06 19:06-19:22 | Non-LLM broad coverage rerun completed with sanitized connection config: OCR, qwen3-ASR 0.6B/1.7B, five embedding specs, two rerankers. |
| 2026-07-06 19:32-20:15 | Official VLM VisionEncoder probe completed: 10/10 4/8-core encoder latency rows passed the 105% gate. |
| 2026-07-06 15:59-16:00 | K3-side cleanup completed: non-hot tested model copies deleted; canonical cache remains local; K3 final state was no model server, TCM 8/8 free, and 75GB root free. |
| 2026-07-07 10:19-11:21 | Source-built SpacemiT `llama.cpp` and ORT equivalence completed. `llama.cpp` passed 6/6 PP/TG rows; ORT passed FastVLM and Qwen3VL-30B VisionEncoder rows. |
| 2026-07-07 11:22-11:37 | Aviation-manual long-context integration completed. 15 core cases generated; Qwen3-4B 1K window closed-loop PASS but took 175.453s; 3K direct sync window was cancelled as latency-risk evidence. |
| 2026-07-07 11:56-12:09 | Aviation-manual 0.6B retests completed. Naive 1K window exposed tokenizer overflow errors; safety-budget rerun completed 6/6 calls without overflow but failed the quality gate with score 0.5958. |
| 2026-07-07 16:34-16:42 | `Qwen3.6-35B-A3B-UD-Q4_K_XL` dual LLM/VLM probe completed on K3 32GB. LLM path passed PP512/TG128, TTFT/decode, and 1K/3K needles; VLM path failed because the served model reports `vision_backend=none` and rejects `image_url`. |
| 2026-07-07 17:24-17:35 | `Qwen3.5-35B-A3B-Q4_0` dual LLM/VLM probe completed on K3 32GB from `/root/models/spacemit-ai`. LLM path passed PP512/TG128, TTFT/decode, and 1K/3K needles; VLM path failed because the served model reports `vision_backend=none` and rejects `image_url`. CLI smoke hit the 300s timeout, so server/API path is the reliable invocation path. |
| 2026-07-08 09:33-10:10 | External `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` full VLM document extraction completed with thinking disabled and `--parallel 1 --cache-ram 512`: 29/30 case pass, 172/173 fields correct, JSON parse 1.0, avg/p95 68.822/78.774s. A fixed 1024 image-token control was aborted as unstable/too slow. |
| 2026-07-08 10:13-10:21 | Same external Qwen3.5-35B-A3B multimodal service retested at 4K context: TTFT 1.071s, decode128 29.374s, 1K needle PASS 58.824s, 3K needle PASS 297.122s, image smoke PASS 8.403s. |

---

## 中文摘要

**平台：** k3-riscv-32g | SpacemiT K3 X100，32GB RAM，Bianbu Linux
**最后校准：** 2026-07-08。本文件原地更新。

### 硬件画像

| 计算单元 | 运行时 | 角色 |
|---|---|---|
| CPU / RVV / IME | SpacemiT private `llama.cpp` + upstream K3 fallback | LLM/VLM text path, embedding/reranker GGUF path |
| Source-built runtime | SpacemiT `llama.cpp` + ORT 源码构建 | 已和系统 runtime 做等价门禁，可作为后续优化基线 |
| SpacemiT ORT / TCM | `spacemit-onnxruntime`, `spacemit-tcm` | 官方 ONNX vision、PP-OCRv5 OCR |
| SMT media backend | private `llama-server --media-backend smt` | Qwen3.5 VLM tar、qwen3-ASR tar |
| Storage | local `drivers/spacemit-ai/model_zoo` + K3 working cache | 本地是 canonical cache，K3 只保留热模型 |

### 执行模式对比

| 任务 | 推荐路径 | 当前结论 |
|---|---|---|
| LLM | private llama.cpp GGUF | `Qwen3-30B-A3B-Q4_0` 是质量首选，但长文本同步高风险；`Qwen3.6-35B-A3B-UD-Q4_K_XL` 和 `Qwen3.5-35B-A3B-Q4_0` 可作为大模型候选/对照 |
| 单模型 LLM+VLM | Qwen3.5-35B / Qwen3.6 image_url 实测 | 不成立：两者均为 `vision_backend=none`，图片输入返回 HTTP 500 unsupported |
| VLM | SMT tar 或 GGUF+mmproj | 同步默认 `Qwen3.5-2B`；`Qwen3VL-4B` 质量好但慢 |
| OCR | PP-OCRv5 ONNX | 专用 OCR 路径，不用 VLM 代替；72 样本 CER 0.0039 |
| ASR | qwen3-ASR 0.6B SMT | 默认 0.6B，1.7B 没有准确率优势且更慢 |
| Embedding | BGE-Zh GGUF | 默认 `Bge-Small-Zh`，p95 5.85ms |
| Reranker | BGE reranker GGUF | 默认 BGE；Qwen3 reranker 只适合慢速/离线 |
| 长文本/飞行手册 | OCR/text extraction + embedding + reranker + cited LLM window | 不能全手册直灌；必须检索窗口化和异步化 |

### 综合性能 + 模型效果

| 模型 | 角色 | 关键指标 | 结论 |
|---|---|---|---|
| `Qwen3-30B-A3B-Q4_0` | LLM primary | GA PASS；TTFT 0.866s；decode128 16.479s；1K 223.834s / 30.488GiB RSS | **短请求可用，长文本必须异步** |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL` | LLM large candidate | PP512 30.49 tok/s；TG128 6.75 tok/s；TTFT 1.017s；decode128 20.277s；1K/3K needle PASS；`vision_backend=none` | **LLM 可用，不是 VLM** |
| `Qwen3.5-35B-A3B-Q4_0` | LLM large control | PP512 29.31 tok/s；TG128 6.48 tok/s；TTFT 1.120s；decode128 28.105s；1K/3K PASS；`vision_backend=none` | **LLM 可用，不是 VLM** |
| `Qwen3.5-2B.tar.gz` | VLM practical | 30/30 文档 case；p95 12.239s | **同步 VLM 默认** |
| `Qwen3VL-4B + mmproj` | VLM quality | 30/30 文档 case；realistic p95 78.064s | **异步/质量控制** |
| `PP-OCRv5_mobile_det+rec.onnx` | OCR | 72 行样本 CER 0.0039；p95 2985.5ms | **OCR 默认** |
| `qwen3-asr-0.6B` | ASR | normalized CER avg 0.0192；RTF p95 0.512 | **ASR 默认** |
| `Bge-Small-Zh-V1.5-Q4_K_M` | Embedding | overall Hit@1 0.9722；p95 5.85ms | **Embedding 默认** |
| `Bge-Reranker-V2-M3-Q4_0` | Reranker | top50 Hit@1 1.0；top20 p95 1333ms；top50 p95 3467ms | **Reranker 默认，控制 top-k** |
| `Qwen3-4B-Q4_0` + aviation 1K window | 长文本闭环 | score 1.0；E2E 175.453s；prefill 7.74 tok/s | **链路可用，同步体验不可用** |
| `Qwen3-4B-Q4_0` + aviation 3K window | 长文本风险 | 首条超过 5 分钟未完成，1024 prompt token 约 9.51 tok/s 后取消 | **不能作为同步默认** |
| `Qwen3-0.6B-Q4_0` + aviation safety window | 长文本烟测 | 6/6 调用成功无超窗；score 0.5958；p95/max 143.945s | **调用稳定，质量不达标** |

### 功耗参考

本轮没有采集板级输入功耗，因此 K3 不能和 AMD/Intel 报告做 TPS/W 横向比较。已经具备性能、延迟、RSS、TCM 状态证据；下一步需要在 LLM 短解码、LLM 长上下文、VLM 文档抽取、PP-OCRv5、qwen3-ASR、embedding 和 reranker 压测期间同步采集板级功耗。

### 选型摘要

| 角色 | 推荐模型 | 说明 |
|---|---|---|
| LLM | `Qwen3-30B-A3B-Q4_0` | 只做短请求或异步长文本；不能直接承载同步长上下文 |
| LLM 大模型候选 | `Qwen3.6-35B-A3B-UD-Q4_K_XL` | LLM 指标通过；不能作为单模型 LLM+VLM |
| LLM 大模型对照 | `Qwen3.5-35B-A3B-Q4_0` | LLM 指标通过；不能作为单模型 LLM+VLM |
| 同步 VLM | `Qwen3.5-2B.tar.gz` | 质量/延迟最均衡 |
| 异步高质量 VLM | `Qwen3VL-4B + mmproj` | 质量好但慢 |
| OCR | `PP-OCRv5_mobile_det+rec.onnx` | 专用 OCR；当前是 line OCR，不是完整版面解析 |
| ASR | `qwen3-asr-0.6B.tar.gz` | 0.6B 优于 1.7B 作为默认 |
| Embedding | `Bge-Small-Zh-V1.5-Q4_K_M` | 最快可用；Nomic/Qwen3 embedding 可备选 |
| Reranker | `Bge-Reranker-V2-M3-Q4_0` | 在线 top-k 建议 <=20 |
| 飞行手册长文本 | OCR/text extraction + BGE embedding + BGE reranker + LLM cited window | 只做 RAG/异步，不做全手册直灌 |

### 风险结论

本轮 raw model-server 测试没有 scheduler/gateway，因此不能证明 `docs/k3-realistic-stress-plan.md` 要求的 `/capacity`、队列等待、异步任务、公平性、取消、TTL 和背压。K3 32G 的关键风险是：Qwen3-30B 长文本会接近吃满内存；Qwen3.5-35B 和 Qwen3.6-35B 当前都只验证为 LLM 候选，不能作为单模型 LLM+VLM 方案；飞行手册类长文本即便 1K 窗口也达到分钟级延迟，必须 RAG/异步；窗口裁剪必须按 tokenizer 或保守安全系数处理，单纯字符估算会导致超窗或证据丢失；VLM 高质量模型延迟过高；reranker 延迟随 top-k 线性放大；OCR 已经应走 PP-OCRv5，不能再用 VLM 顶替；K3 本地磁盘只能作为工作缓存，模型归档应放在本地 `drivers/spacemit-ai/model_zoo`。
