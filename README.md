> [中文文档](./README.zh.md)

# local-ai-bench

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/qiurui144/local-ai-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/qiurui144/local-ai-bench/actions/workflows/ci.yml)

A reproducible **performance × model-quality benchmark platform** for AI-box model selection. It evaluates **VLM** (vision-language) and **LLM** (text) models served via **vLLM** (or any OpenAI-compatible endpoint) across **13 registered dimensions** — latency, throughput, concurrency and stability on the performance axis; accuracy, translation, embedding/rerank retrieval, ASR, **general ability** (gsm8k / mmlu / hellaswag) and real-scenario quality on the model-quality axis; plus **conditioned** capability curves (context-length ladder + prefix-cache cold/warm) — and ships a **complete RAG / LLM validation framework** for production deployments.

Built for the question: *"Can model X replace model Y in production — is the performance enough, does the quality drop?"* `--compare baseline candidate` turns saved reports into an automated **REPLACEABLE / NOT_REPLACEABLE / INCONCLUSIVE** verdict with per-metric 2σ evidence, and the `conditioned` dimension measures capability as a **curve over conditions** (input length, cache state, hardware-attributed reports) rather than a single point.

📋 Version history: [RELEASE.md](RELEASE.md) · 🛠 Developer guide & architecture: [DEVELOP.md](DEVELOP.md)

---

## Validation framework (new in v0.2)

In addition to the vLLM harness, this repository now ships a full
academic-grade validation framework under `benchmark/rigor/` and
`benchmark/rag/`. It implements:

- **Rigor foundation** (`benchmark/rigor/`): statistical tests, effect
  sizes, multi-seed runner, reproducibility snapshots, probability
  calibration (ECE/Brier/Platt/Isotonic), inter-rater reliability
  (Cohen/Fleiss/Krippendorff), ablation orchestrator, cross-validation,
  power analysis, and OOD/subgroup assessment.
- **RAG methodology** (`benchmark/rag/`): all 12 chapters of the RAG
  evaluation playbook — component pipeline traces, offline/online
  alignment, retrieval metrics (NDCG/MRR/MAP/bpref/ERR/RBP),
  reranker assessment + rank fusion (RRF/Borda/CombSUM/CombMNZ),
  answer relevance, claim-level groundedness with RAGAS strict
  faithfulness, LLM-judge prompts with G-Eval CoT, judge
  calibration with position/verbosity/self-preference bias
  detection, judge attack hardening (injection, leakage,
  perturbation), regression CI with flake controller, canary
  rollout with rollback policy, and drift detection with PSI / JS
  divergence / temporal cohorts / auto-curation.
- **Appendices**: 3 JSON schemas, 5 YAML rubrics and 8 runnable labs
  in `benchmark/rag/{schemas,rubrics,labs}/`; 6 production case
  studies in `docs/case-studies/`; plus a 120-question interview bank
  and a capstone system-design document in `docs/`.

Companion docs:

- `docs/ACADEMIC-RIGOR.md` — the 12 principles the framework enforces.
- `docs/BASELINES.md` — reference baselines and threshold defaults.
- `docs/REPRODUCIBILITY.md` — pinning policy and snapshot format.
- `docs/CITATION.md` — bibtex for citing this work and underlying methods.
- `docs/CONTRIBUTING.md` — the contributing guide (harness + methodology).
- `docs/capstone-system-design.md` — reference end-to-end architecture.
- `reports/2026-06-19-all-model-matrix-results.en.md` and
  `reports/2026-06-19-all-model-matrix-results.json` — latest full Windows
  model matrix, benchmark evidence summary, and model-selection guidance.
- `reports/2026-06-19-git-readiness.en.md` — pre-push quality, security, and
  open-source readiness checklist.

Run the validation tests:

```bash
python -m pytest tests/rigor tests/rag -q
```

Run a lab as a worked example:

```bash
python -m benchmark.rag.labs.lab2_retrieval_metrics
python -m benchmark.rag.labs.lab4_groundedness_audit
python -m benchmark.rag.labs.lab8_drift_detection
```

The legacy `algo-base/llama-benchmark` dataset infrastructure is absorbed under `benchmark/llama_benchmark/`.

---

## What it measures

| Dimension | What | Why it matters |
|---|---|---|
| **Accuracy** | Classification precision, entity recall, fact recall, **must-not-say violations** against a golden set | Catches digit-shift errors (e.g. ¥120 vs ¥1200) that pure perplexity misses |
| **TTFT** | First-token latency P50 / P95 (streaming) | UX baseline — anything > 2s feels broken |
| **Throughput** | Aggregate tokens-per-second under sustained load | Capacity planning |
| **Concurrency** | Success rate + P50/P95 across 1 / 5 / 10 / 30 / 50 concurrent requests | Production load shape |
| **Stability** | 30-min sustained run; latency drift between first 5 min and last 5 min | Memory leaks, KV-cache thrashing |
| **Token budget** | Input/output token distribution + truncation rate (measured inside the accuracy dimension) | Cost monitoring + silent truncation detection |
| **PP / TG split** | Prefill (PP) vs decode (TG) tokens-per-second measured separately (llama-bench style) | Aggregate throughput hides two different hardware regimes — prefill is compute-bound, decode is bandwidth-bound |
| **Translation** | zh↔en MT quality (SacreBLEU / chrF / COMET) + per-language-pair latency, across 3 task levels | Validates a model's bilingual deployment readiness |
| **Embedding** | Retrieval recall@k / MRR / nDCG@10 + single-query latency P50 (resident) + RSS dual distinction + numerical validation | The core AI-box RAG retriever — quality, real chat-query latency/memory, and a zero/NaN-vector gate |
| **Rerank** | Standalone reranker nDCG@10 / MRR + per-pair latency (distinct from the RAG-internal reranker) | Second-stage re-rank quality vs its latency cost (real-time vs offline) |
| **ASR** | Chinese CER / WER / RTF over an audio manifest (ONNX backend) | Speech transcription accuracy + real-time capability |
| **General ability** | gsm8k (math reasoning) / mmlu (knowledge, 4 subjects) / hellaswag (commonsense) accuracy via revision-pinned HF datasets, reusing the absorbed `llama_benchmark` datasets through an in-process adapter. HellaSwag is scored as A–D choice-letter accuracy over the chat API (a deterministic approximation, **not** length-normalized logprob — the method is recorded in the report). Unreachable or synthetic-fallback datasets → `BLOCKED`, never a fake score. **Thresholds are tier-aware**: 3-7B default (gsm8k≥0.55/mmlu≥0.55/hellaswag≥0.60); 1.5B (gsm8k≥0.30/mmlu≥0.45/hellaswag≥0.50); ≤0.6B (gsm8k≥0.20/mmlu≥0.40/hellaswag≥0.45) — set per-model in `models.yaml::benchmarks.general_ability.thresholds` | A model that aces your golden set can still have lost general reasoning vs the model it replaces |
| **Conditioned** | Capability as a **curve**, not a point: task quality + needle recall + TTFT/TPS across a context-length ladder (1k → 32k, capped by the model's max len) + prefix-cache cold/warm A/B (TTFT speedup + output-consistency check — the cache must never change answers) | "Good at 1k tokens" says nothing about 16k; cache-hot demos hide cold-start latency |
| **Scenarios** | 8 real-scenario tasks (S1-S8): WeChat-screenshot intent (VLM), case-logic contradiction detection, article knowledge grading, **instruction following**, **structured extraction**, **function calling**, **VLM document extraction**, **adversarial stability** — L1 deterministic + L2 multi-seed LLM judge with anchor calibration. 265 curated cases. Judge model selected automatically by priority (7B > 14B > 3B > 1.5B > 0.6B; highest vram within tier). Runs independently — does **not** require `conditioned` to have completed. | Measures what standardized suites can't: behavior on the product's true input distribution |
| **Conversation drift** | Quality curve across 0 / 5 / 10 / 20 prior conversation turns (filler corpus). Max drop > 15% → DRIFT/FAIL. Prerequisite for long-session deployment. | "Good at 1k tokens" says nothing about turn-50 degradation |

Pass/Warn/Fail is determined by thresholds in `golden/expectations.json::*_acceptance_criteria` and `models.yaml::benchmarks.*.thresholds` — exit code `0` PASS / `1` WARN / `2` FAIL, ready for CI consumption. A named model (`--model <name>`) that errors (e.g. endpoint not ready) exits `2`; under `--model all` a down model is skipped per its "run the models that are up" contract — but if **zero** models produce any measurement the run exits `2`: an empty run can never report success. A quality dimension reporting `BLOCKED` (missing prerequisites — e.g. the `general_ability` / `conditioned` datasets are unreachable on an offline host) counts as WARN (exit `1`), never a silent pass; `--skip general_ability,conditioned` restores the pre-v0.3 default-run behavior.

**Replaceability verdict (`--compare BASELINE CANDIDATE`):** the north-star question gets an automated answer from already-saved reports (offline — no rerun):

```bash
python run_benchmark.py --model qwen2.5-vl-7b-fp16   --seeds 3 --skip stability
python run_benchmark.py --model qwen3-vl-8b-instruct --seeds 3 --skip stability
python run_benchmark.py --compare qwen2.5-vl-7b-fp16 qwen3-vl-8b-instruct
```

Verdict and exit code: `REPLACEABLE` (0) / `INCONCLUSIVE` (1) / `NOT_REPLACEABLE` (2), with per-metric Δ / σ / significance evidence written to `output/reports/compare_*.{json,md}`. The discipline is hard-coded, not configurable: **REPLACEABLE** requires every shared quality metric within 2σ **and** the candidate passing its own performance thresholds; any significant quality regression → **NOT_REPLACEABLE**; **single-seed data is capped at INCONCLUSIVE** (a ranking from one run is noise — produce `--seeds 3` data first). Reports from different `harness_version` or `condition` are refused, legacy reports without `schema_version` are refused, and a `hardware_profile` mismatch forces the performance side INCONCLUSIVE (quality is still compared) — every report carries a schema-v1 envelope (`schema_version` / `harness_version` / `hardware_profile` / `condition`) to make these checks possible.

**HTML visual report:** every benchmark run automatically produces a self-contained `output/reports/<model>_<ts>.html` alongside the JSON/Markdown, viewable in any browser. It includes a quality radar chart (9 axes), performance table, per-scenario bar chart, and conversation-drift line chart. The `--compare` mode produces `compare_*.html` with side-by-side radar overlay and REPLACEABLE/INCONCLUSIVE/NOT_REPLACEABLE verdict badge.

**Multi-seed runs (`--seeds N`, default 1):** single-run quality numbers from a sampling LLM are statistically noisy — a ranking can flip between runs, so a single number is not a claim (CLAUDE.md §2.3: report mean ± std, never a lone score). With `--seeds N` the full suite is re-run N times per model and the report gains a top-level `multi_seed` block (`n_seeds` plus per-metric `mean`/`std`/`ci95_lower`/`ci95_upper` over the numeric quality metrics present in all N runs, via `benchmark/rigor/multi_seed_runner.aggregate`), rendered as a "Multi-seed" section in the Markdown report. Honesty notes: v1 does **not** pin a per-call sampling seed — the variance source is the model's own temperature noise across identical reruns; and the exit-code verdict is the **worst** verdict across the N runs (a FAIL in any run is a FAIL — verdicts are never averaged).

---

## Translation scenario (`benchmark/translation/`)

Evaluates **machine-translation quality and latency** for any LLM serving an
OpenAI-compatible endpoint, on the zh↔en pair. Enabled per model via the
`translation_capable: true` hint in `models.yaml`; it is one more dimension in
the standard `run_benchmark.py` flow (skip with `--skip translation`).

### Metrics

| Metric | Package | Compute | Notes |
|---|---|---|---|
| **SacreBLEU** | `sacrebleu` | CPU | Corpus BLEU, reproducible tokenization (`zh` tokenizer for Chinese targets). Pure-Python fallback if the package is absent. |
| **chrF** | `sacrebleu` | CPU | Character n-gram F-score — tokenization-free, the robust backstop for Chinese. |
| **COMET** | `unbabel-comet` | **GPU-recommended** | Neural quality estimate (`Unbabel/wmt22-comet-da`). Auto-skipped with `"COMET requires GPU/DGX"` when no CUDA GPU / package — never crashes the run. |
| **Term-match rate** | — | CPU | L3 exact-match rate for a required terminology glossary. |

Every metric is numerically validated (non-empty hypotheses, `0 ≤ BLEU/chrF ≤ 100`, finite / non-NaN-Inf) so a silently broken model surfaces as a FAIL rather than a plausible-looking number.

### Task levels

- **L1 — single sentence**: straight zh↔en sentence translation (raw adequacy + fluency).
- **L2 — context consistency**: a 3–5 sentence passage translated as a block, so pronoun reference / tense / named entities stay consistent across sentence boundaries.
- **L3 — terminology**: domain text where required technical terms (`prompt` / `embedding` / `向量化` …) must be rendered with the canonical translation; scored by exact-match term rate.

### Datasets

- **Flores-200** zh↔en (devtest split, 100-sentence subset by default) — pulled at runtime from the non-gated pure-parquet HF mirror [`haoranxu/FLORES-200`](https://huggingface.co/datasets/haoranxu/FLORES-200) (the ALMA paper's eval mirror; 1012 devtest sentences), pinned to a commit SHA. No `trust_remote_code`, no auth token — pure data, no upstream code execution. Override with `FLORES_DATASET=<repo>` / `FLORES_REVISION=<commit-sha>`. (`facebook/flores` is gated and script-based — unloadable on `datasets>=3` without a granted token.) Offline / air-gapped hosts fall back to a small built-in synthetic set (set `TRANSLATION_OFFLINE=1` to force it); the fallback is logged loudly, recorded as `dataset_sources` in the report, and caps the translation verdict at `WARN` — synthetic scores never masquerade as Flores-200 results.
- **Custom product-domain corpus** — `datasets/translation/custom_zh_en.jsonl` (~60 hand-authored, synthetic, no PII pairs, AI-infra / engineering / support domains; L3 glossaries inline). Replace with your own reviewed corpus using the same JSONL schema (`{src, tgt, domain, glossary}`).

### Usage

```bash
# Translation dimension only, on the LLM primary
python run_benchmark.py --model qwen3-30b-a3b-instruct-2507-fp8 \
    --skip accuracy,ttft,throughput,concurrency,stability

# Force offline Flores fallback (air-gapped host)
TRANSLATION_OFFLINE=1 python run_benchmark.py --model qwen3-30b-a3b-instruct-2507-fp8 \
    --skip accuracy,ttft,throughput,concurrency,stability
```

Thresholds live in `models.yaml::benchmarks.translation.thresholds` (`bleu_min`,
`chrf_min`, `term_match_rate_min`, …); golden cases in
`golden/expectations.json::translation_cases`. CPU-only tests (no vLLM/GPU
needed):

```bash
python -m pytest tests/translation -q
```

---

## Real-scenario dimension (`benchmark/scenarios/`)

Three tasks drawn from the product's true input distribution, one more dimension
in the standard `run_benchmark.py` flow (skip with `--skip scenarios`):

- **S1 `wechat_intent`** (VLM) — read synthetic WeChat-style chat screenshots, extract the message content and classify the chat intent (8 intent labels).
- **S2 `case_logic`** (LLM) — find contradictions across case-narrative segments (`time_conflict` / `causal_break` / `fact_mismatch`) plus an overall consistency label.
- **S3 `article_knowledge`** (LLM) — judge self-media article claims as `accurate` / `inaccurate` / `unverifiable` and assign an A–D knowledge grade.

**Scoring** is two-layer: **L1** deterministic metrics (accuracy / recall / F1 against
the case labels) + **L2** LLM-judge, run multi-seed (N=3) with paired-anchor
calibration from `golden/scenarios.json`. The judge model is set in
`models.yaml::benchmarks.scenarios.judge_model` (auto-picked when `null`) and is
**enforced to differ from the model under test**; `num_cases` caps the per-scenario
case count.

**Data is dual-track**, with provenance recorded per case in
`datasets/scenarios/<scenario>/cases.jsonl` (5 seed cases each):

- **Synthetic** — `scripts/render_wechat_case.py` (PIL renderer) generates screenshot cases. `provenance: "synthetic"` caps the scenario verdict at `WARN` — synthetic-only data never produces a PASS.
- **Curated** — `scripts/curate_scenario_case.py` ingests reviewed real-world cases (`provenance: "curated"` / `"dataset"`), unlocking full PASS verdicts.

`scripts/check_no_real_images.sh` is the PII control on the image fixtures: every
png under `fixtures/scenarios/` must be named after a `dialogs.json` renderer id
(provenance whitelist) and every `payload.image` referenced in `cases.jsonl` must
exist — real screenshots never enter git.

A missing `cases.jsonl` makes the scenario `BLOCKED` (counted as WARN, never a
fake PASS).

**Note:** out of the box this dimension ships with 5 synthetic seed cases per
scenario, so a default run reports `WARN` (exit 1) **by design** until curated or
dataset-track cases are added via `scripts/curate_scenario_case.py` —
synthetic-only scores are intentionally never a clean PASS. Use
`--skip scenarios` to restore the previous exit behavior.

---

## AI-box capabilities (`benchmark/embedding`, `benchmark/rerank`, `benchmark/asr`)

Core retrieval + speech capabilities synced from the K23 edge AI-box evaluation,
adapted from edge `llama.cpp`/`ONNX` to served **OpenAI-compatible** endpoints
(vLLM / sglang / llama.cpp server / Ollama). Each is an optional dimension in
the standard `run_benchmark.py` flow, enabled per model via a `*_capable: true`
hint in `models.yaml`, and skippable (`--skip embedding,rerank,asr`).

### Embedding (`benchmark/embedding/`)

Retrieval-quality + latency/memory characterisation of an embedding deployment —
the retriever stage of a RAG pipeline.

| Metric | Compute | Notes |
|---|---|---|
| **recall@1 / @5 / @10** | CPU (NumPy) | per query: embed query + candidate docs, cosine top-k, measure where the gold doc lands |
| **MRR** | CPU | mean reciprocal rank of the first relevant doc |
| **nDCG@10** | CPU | normalized DCG, binary relevance |
| **Single-query latency P50** | resident model | timed against a resident endpoint — the real chat-query path (not per-process CLI load) |
| **RSS dual distinction** | local proc | *batch RSS* (inflated by logical-batch KV) vs *resident-query RSS* (≈ weights + small KV — the real chat memory); reported `available: false` for remote endpoints |
| **Numerical validation** | CPU | NaN / Inf / **zero-vector** / dim-drift check — a zero vector is a hard FAIL (the classic "fast but wrong" trap) |

Datasets: a Chinese retrieval set in `datasets/retrieval/cmteb_zh_subset.jsonl`
(you provide; e.g. a C-MTEB subset), with a built-in synthetic Chinese fallback
for offline / unit-test runs (flagged `source="builtin"`, never mistaken for a
real score). See [`datasets/retrieval/README.md`](datasets/retrieval/README.md).

### Rerank (`benchmark/rerank/`)

A **standalone** reranker benchmark (distinct from the RAG-internal reranker in
`benchmark/rag/reranker.py`): score every (query, candidate-doc) pair via the
served endpoint (yes/no relevance, logprob when available), re-rank, and report
**nDCG@10 / MRR** over the same retrieval gold as embedding, plus **per-pair
latency P50** and a positive-vs-negative score-separation sanity check. Per-pair
latency is reported but not gated — a high-quality reranker can be offline-only.

### ASR (`benchmark/asr/`)

Chinese **CER / WER / RTF** over an audio manifest. CER (character error rate) is
the primary metric for Chinese; RTF < 1.0 means real-time capable. The
transcription backend is pluggable ONNX (default sherpa-onnx SenseVoice); when
the runtime / model / dataset is absent the dimension reports `blocked` and the
verdict is `SKIP` rather than crashing. Audio files are not shipped — see
[`datasets/asr/README.md`](datasets/asr/README.md) for the manifest schema.

### Usage

```bash
# Embedding-only run on the embedding primary
python run_benchmark.py --model qwen3-embedding-0.6b \
    --skip accuracy,ttft,throughput,prefill_decode,concurrency,stability,translation,rerank,asr

# CPU-only tests (no vLLM / GPU / model needed) — metric math + numerical gates:
python -m pytest tests/embedding tests/rerank tests/asr tests/performance -q
```

> **vLLM / GPU caveat**: the metric/validation logic is fully unit-tested on CPU
> with injected fakes. End-to-end numbers (recall on a real embedder, rerank
> nDCG, ASR CER, PP/TG tok/s) require a served endpoint + GPU/ONNX backend and
> are **not** run here — run `run_benchmark.py` against a live deployment to
> produce them.

---

## Reference model matrix (`models.yaml`)

The harness is provider-agnostic — anything serving an **OpenAI-compatible** endpoint works (vLLM, sglang, lmdeploy, llama.cpp server, Ollama 0.21+, …). Out of the box we ship a **10-model reference matrix**: 4 VLM/LLM chat models plus 6 embedding / rerank / ASR models.

| Role | Model | Quant | Port | VRAM | Min HW |
|---|---|---|---|---|---|
| 🌟 VLM primary | Qwen3-VL-8B-Instruct | BF16 | 8001 | 20 GB | A100-40G |
| 📍 VLM baseline | Qwen2.5-VL-7B-Instruct | BF16 | 8002 | 18 GB | A100-40G |
| 🌟 LLM primary | Qwen3-30B-A3B-Instruct-2507-FP8 (MoE) | FP8 | 9001 | 35 GB | H100-80G |
| 🌟🌟🌟 LLM flagship | Qwen3-235B-A22B-Instruct-2507-FP8 (MoE) | FP8 | 9002 | 240 GB | 8×H100-80G |
| 🌟 Embedding primary | Qwen3-Embedding-0.6B | — | 9101 | 4 GB | A100-40G |
| Embedding high-acc | Qwen3-Embedding-4B | — | 9102 | 12 GB | A100-40G |
| Reranker (generative) | Qwen3-Reranker-4B | — | 9201 | 12 GB | A100-40G |
| 🌟 Reranker (real-time) | bge-reranker-v2-m3 | Q8_0 | 9202 | 1 GB | K3-X100 (CPU) |
| Reranker (latency floor) | bge-reranker-base | Q4_K_M | 9203 | 1 GB | K3-X100 (CPU) |
| ASR primary | SenseVoiceSmall | INT8 | — (local ONNX) | 1 GB | CPU |

> Note: `scripts/prepare_offline.sh` MODEL_SET tiers (minimal/standard/full) cover only the 4 chat models; the embedding / rerank / ASR models currently have no offline-download path (see RELEASE.md Known issues).

**No-downgrade design**: if you have DGX-class hardware, run real models. The flagship MoE entries activate only ~3B / ~22B params per forward pass, so they're competitive with much smaller dense models on latency while preserving quality.

Drop in your own models by appending to `models.yaml` — only `(name, hf_repo, port, role)` are required; other fields are documentation hints.

---

## Quick start

### Prerequisites

- Linux (Ubuntu 22.04 or 24.04 tested) with **CUDA-capable GPU**
- Python 3.10+
- ~50 GB free disk for the default 80-GB-of-models matrix (or 16 GB for the minimal set)

### Quick start on Windows (Ollama or llama.cpp)

No GPU server required — run Ollama or llama.cpp locally on a Windows machine and point the harness at it from any machine on the same network.

**Step 1 — On the Windows machine (PowerShell, one-time)**

```powershell
# Enable OpenSSH for remote management (optional but recommended)
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd; Set-Service -Name sshd -StartupType Automatic
New-NetFirewallRule -Name sshd -DisplayName "OpenSSH" -Enabled True -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow

# Install Ollama
winget install Ollama.Ollama --silent

# Pull a model
ollama pull qwen2.5:7b

# Open Ollama port (so other machines can reach it)
New-NetFirewallRule -Name Ollama -DisplayName "Ollama" -Enabled True -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
```

**Step 2 — On your dev machine, configure models.yaml**

```yaml
models:
  - name: qwen2.5-7b-win-ollama
    provider: ollama
    port: 11434
    base_url_override: http://192.168.1.100:11434/v1   # Windows machine IP
    model_id: qwen2.5:7b
    task_type: text_only
    notes: "Windows Ollama — qwen2.5:7b"
```

**Step 3 — Probe and benchmark**

```bash
# Verify the endpoint is reachable and functional
python3 scripts/probe_provider.py --model qwen2.5-7b-win-ollama

# Run the benchmark (skip GPU-heavy dims)
python run_benchmark.py --model qwen2.5-7b-win-ollama \
    --skip stability,embedding,rerank,asr
# → output/reports/qwen2.5-7b-win-ollama_<ts>.json + .html
```

### Provider configurations

The harness talks HTTP to any OpenAI-compatible endpoint. Supported `provider:` values:

| Provider | `provider:` value | Notes |
|----------|-------------------|-------|
| Local vLLM | `local_vllm` | Default. `port: 8000` |
| **llama.cpp server** | `llama_cpp` | `llama-server --port 8080 --model model.gguf` |
| Ollama | `ollama` | Works on Linux + macOS + **Windows** |
| OpenAI cloud | `openai` | `api_key_env: OPENAI_API_KEY`; 429 auto-retried |
| DashScope | `dashscope` | `api_key_env: DASHSCOPE_API_KEY` |
| DeepSeek | `deepseek` | `api_key_env: DEEPSEEK_API_KEY`; 429 auto-retried |

### 3-step deploy

```bash
# 1. On a machine with internet — download all artifacts
git clone https://github.com/qiurui144/local-ai-bench.git
cd local-ai-bench
MODEL_SET=standard bash scripts/prepare_offline.sh
# MODEL_SET options:
#   minimal  (~16 GB) — VLM primary only
#   standard (~80 GB) — VLM ×2 + LLM-30B  [recommended]
#   full    (~320 GB) — all 4 models including 235B

# 2. (Optional) bundle for offline transfer to an air-gapped GPU host
tar czf local-ai-bench-bundle.tar.gz local-ai-bench/
scp local-ai-bench-bundle.tar.gz dgx:/data/

# 3. On the GPU host
cd /path/to/local-ai-bench
sudo bash scripts/bootstrap.sh   # installs vLLM, links models to HF cache
bash run.sh                      # default: VLM primary, skips 30-min stability
```

### Targeted runs

```bash
# Replace baseline with candidate — the core "can X replace Y?" question
bash vllm_configs/start_all.sh   # uncomment baseline in start_all.sh first
python run_benchmark.py --model qwen2.5-vl-7b-fp16  --skip stability
python run_benchmark.py --model qwen3-vl-8b-instruct --skip stability
cat output/reports/matrix_*.md

# LLM concurrency sweep only
python run_benchmark.py --model qwen3-30b-a3b-instruct-2507-fp8 \
    --skip accuracy,ttft,throughput,stability

# Flagship 235B smoke test (needs 8×H100)
python run_benchmark.py --model qwen3-235b-a22b-instruct-2507-fp8 \
    --skip concurrency,stability
```

---

## Bring your own data

The repo ships **no fixture images** by design — VLM benchmarks need real-world screenshots / scans / photos that often contain PII. See [`fixtures/README.md`](fixtures/README.md) for guidance on:

- What images go where (one per `golden/expectations.json::cases[].image`)
- How to author your own golden-set entries (`must_identify_entities`, `must_identify_facts`, **`must_not_say`**)
- Why `.gitignore` excludes binary fixtures by default

The shipped `golden/expectations.json` is a synthetic 9-case demo. Replace it with your own ground truth to evaluate against your domain.

---

## Repository layout

```
local-ai-bench/
├── run.sh                    # one-liner entry point
├── run_benchmark.py          # main scheduler
├── models.yaml               # model matrix (edit this to add/remove models)
├── common.py                 # vLLM client + shared utilities
├── requirements.txt          # httpx / pyyaml / Pillow / pynvml / pydantic / loguru / numpy / scipy / sacrebleu / pytest …
├── benchmark/
│   ├── accuracy.py           # golden-set driven accuracy
│   ├── performance.py        # TTFT / throughput / concurrency / stability / PP-TG split
│   ├── translation/          # zh<->en MT: SacreBLEU/chrF/COMET + latency (L1/L2/L3)
│   ├── embedding/            # retrieval recall@k/MRR/nDCG + latency/RSS + numerical validation
│   ├── rerank/               # standalone reranker nDCG/MRR + per-pair latency
│   ├── asr/                  # Chinese CER/WER/RTF (ONNX backend, graceful BLOCKED)
│   ├── scenarios/            # real-scenario dimension (wechat_intent / case_logic / article_knowledge)
│   ├── general_ability/      # gsm8k / mmlu / hellaswag via llama_benchmark adapter (pinned datasets)
│   ├── conditioned/          # context-ladder + needle + prefix-cache cold/warm condition curves
│   ├── registry.py           # DimensionSpec registry + shared verdict semantics (single source)
│   ├── report/               # per-dimension Markdown render hooks
│   │   ├── html_report.py            # self-contained HTML report with Chart.js radar/bar/drift
│   │   └── sections.py               # per-dimension Markdown render hooks
│   ├── compare.py            # --compare replaceability verdict (2σ discipline, exit 0/1/2)
│   ├── rigor/                # statistical rigor library (multi-seed, effect sizes, calibration, …)
│   ├── rag/                  # RAG validation framework (12 chapters + labs/rubrics/schemas)
│   ├── llama_benchmark/      # absorbed legacy harness (library; CLI currently unusable — see RELEASE.md)
│   ├── llama_configs/        # llama_benchmark config (models.yaml / benchmarks.yaml / devices)
│   └── llama_baselines/      # llama_benchmark measured baselines (K1-SpacemiT runs + trend.md)
├── vllm_configs/
│   ├── launch_helpers.sh     # vllm serve helper functions
│   └── start_all.sh          # batch model startup (default: VLM primary only)
├── scripts/
│   ├── prepare_offline.sh    # internet host: pull wheels + models
│   ├── bootstrap.sh          # GPU host: install vLLM, link models
│   ├── probe_provider.py         # endpoint smoke test (reachability / JSON mode / seed / VLM)
│   ├── verify_benchmark.py       # dataset integrity verification (auto-detects scenarios via registry)
│   ├── render_wechat_case.py # synthetic WeChat-screenshot renderer (scenario S1 fixtures)
│   ├── curate_scenario_case.py  # ingest reviewed real-world scenario cases (curated/dataset track)
│   ├── derive_cail_cases.py  # dataset-track S2 cases derived from CAIL2018 (HF revision-pinned)
│   ├── derive_cail_dialogs.py   # dataset-track S1 dialogs derived from CAIL corpora
│   ├── check_no_real_images.sh  # PII control: fixture-image provenance whitelist
│   └── setup_zerotier.sh     # OPTIONAL: ZeroTier VPN for remote deploy
├── datasets/
│   ├── translation/          # zh<->en parallel corpora (custom JSONL; Flores at runtime)
│   ├── retrieval/            # embedding/rerank retrieval set (custom JSONL; builtin fallback)
│   ├── asr/                  # ASR manifest template (audio + reference transcript)
│   ├── scenarios/            # real-scenario cases (wechat_intent / case_logic / article_knowledge)
│   └── conditioned/          # needle probes for the conditioned context-ladder dimension
├── fixtures/
│   └── README.md             # bring-your-own-data guide
├── golden/
│   └── expectations.json     # acceptance criteria (per dimension) + demo cases
├── docs/                     # deep dives: rigor / baselines / reproducibility / case-studies / specs
├── tests/                    # offline test suite (no GPU needed) — see tests/TESTING.md
└── .github/workflows/ci.yml  # lint / syntax / shellcheck + full offline pytest suite
```

---

## Optional: deploying to a remote air-gapped GPU host via ZeroTier

If you want to ship to a remote DGX through a flat L2 VPN, the bundled `scripts/setup_zerotier.sh` automates ZeroTier install and joining a network you've created at [my.zerotier.com](https://my.zerotier.com):

```bash
ZEROTIER_NETWORK_ID=<your-16-hex-id> sudo -E bash scripts/setup_zerotier.sh
# Then approve the new node at https://my.zerotier.com/network/<your-id>
```

This is entirely optional — direct SSH or `scp` works just as well.

---

## FAQ

**Q: Why both VLM and LLM in one repo?**
A: Many teams replace both at the same major model upgrade (e.g. Qwen2.5 → Qwen3). Keeping them in one harness lets you compare against a shared baseline and run cross-modal accept criteria.

**Q: Why vLLM specifically?**
A: It's the de-facto OpenAI-compatible serving stack with strong continuous-batching, paged-attention, and FP8 / AWQ support. The harness itself only talks HTTP, so you can point it at any compatible endpoint — but the launch scripts assume vLLM.

**Q: Can I run on consumer GPUs (4090 / 3090 / 7900 XT)?**
A: The 235B flagship — no. The 30B-A3B FP8 — barely (RTX 4090 is 24 GB, FP8 wants ~35). The 8B VLMs — yes, with care (bf16 → fp16). Adjust `models.yaml::quantization` and `dtype` accordingly.

**Q: My model is OpenAI-compatible but not on HuggingFace.**
A: Set `hf_repo: null` in `models.yaml` and skip `prepare_offline.sh` — point `start_all.sh` directly at your endpoint URL.

**Q: How do I run this on Windows without a GPU server?**
A: Use Ollama for Windows (`winget install Ollama.Ollama`), set `provider: ollama` with `base_url_override` pointing at the Windows machine IP. The harness only needs HTTP access to the endpoint. For remote management, enable OpenSSH (`Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0`). See "Quick start on Windows" above.

**Q: How do I use llama.cpp instead of vLLM?**
A: Start `llama-server --port 8080 --model model.gguf` (it serves an OpenAI-compatible endpoint), set `provider: llama_cpp` and `port: 8080` in models.yaml. Run `python3 scripts/probe_provider.py --model <name>` to verify. The harness's 429 retry only activates for cloud providers; local llama.cpp doesn't need it.

**Q: The HTML report shows my quality scores as 0 for several dimensions.**
A: Dimensions that are `SKIPPED` (model doesn't have the capability) or `BLOCKED` (prerequisites missing — e.g. datasets unreachable) show as 0 in the radar. Run `python3 scripts/verify_benchmark.py` to check dataset integrity, or `--skip <dim>` to omit dimensions you haven't set up yet.

**Q: `--compare` returns INCONCLUSIVE even though both models ran fine.**
A: INCONCLUSIVE is returned when: (a) either report used only 1 seed — re-run with `--seeds 3`; (b) hardware profiles differ between reports — performance side is forced INCONCLUSIVE but quality still compared; (c) reports come from different harness versions — rebuild both reports on the same version.

**Q: How do I add a new benchmark dimension?**
A: New package under `benchmark/<dim>/`, add a `DimensionSpec` entry to `run_benchmark.py::DIMENSIONS` (run + capability gate + render hook in `benchmark/report/sections.py`), add thresholds to `models.yaml::benchmarks`. See `docs/CONTRIBUTING.md` and `DEVELOP.md`.

---

## Contributing

PRs welcome — see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). Interactions in this project are governed by the [Code of Conduct](docs/CODE_OF_CONDUCT.md).

The maintainers prefer small, focused PRs over sweeping refactors. New model adapters, hardware configs, and benchmark dimensions are especially welcome. **Never commit real PII to fixtures/.**

## License

[Apache License 2.0](LICENSE)

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm) — the serving stack that makes this all reasonable
- [Qwen](https://github.com/QwenLM/Qwen3) — the reference model family used in the default matrix
- [HuggingFace Hub](https://huggingface.co) — model distribution
