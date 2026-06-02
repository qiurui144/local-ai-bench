# vlm-llm-benchmark

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/qiurui144/vlm-llm-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/qiurui144/vlm-llm-benchmark/actions/workflows/ci.yml)

A small, reproducible benchmark harness for evaluating **VLM** (vision-language) and **LLM** (text) models served via **vLLM** — across accuracy, latency, throughput, concurrency, stability, token-budget and **translation** dimensions on a single high-end GPU node — plus a **complete RAG / LLM validation framework** for production deployments.

Built for the question: *"Can model X replace model Y in production without quality regression?"* — and now: *"Is my RAG pipeline ready to ship?"*

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
- **Appendices**: 3 JSON schemas, 5 YAML rubrics, 8 runnable labs,
  6 production case studies, 120-question interview bank, and a
  capstone system-design document — all in `docs/` and
  `benchmark/rag/{labs,rubrics,schemas,case_studies}/`.

Companion docs:

- `docs/ACADEMIC-RIGOR.md` — the 12 principles the framework enforces.
- `docs/BASELINES.md` — reference baselines and threshold defaults.
- `docs/REPRODUCIBILITY.md` — pinning policy and snapshot format.
- `docs/CITATION.md` — bibtex for citing this work and underlying methods.
- `docs/CONTRIBUTING.md` — methodological guidelines for contributors.
- `docs/CROSS-BENCH-MAPPING.md` — how to combine with attune-bench Rust criterion suite.
- `docs/capstone-system-design.md` — reference end-to-end architecture.

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

This validation framework is the **single source of truth** for
RAG / LLM evaluation methodology across attune, attune-pro, cloud,
and the RV toolchain projects. The legacy `algo-base/llama-benchmark`
has been absorbed under `benchmark/llama_benchmark/`.

---

## What it measures

| Dimension | What | Why it matters |
|---|---|---|
| **Accuracy** | Classification precision, entity recall, fact recall, **must-not-say violations** against a golden set | Catches digit-shift errors (e.g. ¥120 vs ¥1200) that pure perplexity misses |
| **TTFT** | First-token latency P50 / P95 (streaming) | UX baseline — anything > 2s feels broken |
| **Throughput** | Aggregate tokens-per-second under sustained load | Capacity planning |
| **Concurrency** | Success rate + P50/P95 across 1 / 5 / 10 / 30 / 50 concurrent requests | Production load shape |
| **Stability** | 30-min sustained run; latency drift between first 5 min and last 5 min | Memory leaks, KV-cache thrashing |
| **Token budget** | Input/output token distribution + truncation rate | Cost monitoring + silent truncation detection |
| **PP / TG split** | Prefill (PP) vs decode (TG) tokens-per-second measured separately (llama-bench style) | Aggregate throughput hides two different hardware regimes — prefill is compute-bound, decode is bandwidth-bound |
| **Translation** | zh↔en MT quality (SacreBLEU / chrF / COMET) + per-language-pair latency, across 3 task levels | Validates a model's bilingual deployment readiness |
| **Embedding** | Retrieval recall@k / MRR / nDCG@10 + single-query latency P50 (resident) + RSS dual distinction + numerical validation | The core AI-box RAG retriever — quality, real chat-query latency/memory, and a zero/NaN-vector gate |
| **Rerank** | Standalone reranker nDCG@10 / MRR + per-pair latency (distinct from the RAG-internal reranker) | Second-stage re-rank quality vs its latency cost (real-time vs offline) |
| **ASR** | Chinese CER / WER / RTF over an audio manifest (ONNX backend) | Speech transcription accuracy + real-time capability |

Pass/Warn/Fail is determined by thresholds in `golden/expectations.json::*_acceptance_criteria` and `models.yaml::benchmarks.*.thresholds` — exit code `0` PASS / `1` WARN / `2` FAIL, ready for CI consumption.

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

- **Flores-200** zh↔en (`devtest` split, 100-sentence subset by default) — pulled at runtime from the HF `facebook/flores` dataset. Offline / air-gapped hosts fall back to a small built-in synthetic set automatically (set `TRANSLATION_OFFLINE=1` to force it).
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

The harness is provider-agnostic — anything serving an **OpenAI-compatible** endpoint works (vLLM, sglang, lmdeploy, llama.cpp server, Ollama 0.21+, …). Out of the box we ship a 4-model reference matrix:

| Role | Model | Quant | Port | VRAM | Min HW |
|---|---|---|---|---|---|
| 🌟 VLM primary | Qwen3-VL-8B-Instruct | BF16 | 8001 | 20 GB | A100-40G |
| 📍 VLM baseline | Qwen2.5-VL-7B-Instruct | BF16 | 8002 | 18 GB | A100-40G |
| 🌟 LLM primary | Qwen3-30B-A3B-Instruct-2507-FP8 (MoE) | FP8 | 9001 | 35 GB | H100-80G |
| 🌟🌟🌟 LLM flagship | Qwen3-235B-A22B-Instruct-2507-FP8 (MoE) | FP8 | 9002 | 240 GB | 8×H100-80G |

**No-downgrade design**: if you have DGX-class hardware, run real models. The flagship MoE entries activate only ~3B / ~22B params per forward pass, so they're competitive with much smaller dense models on latency while preserving quality.

Drop in your own models by appending to `models.yaml` — only `(name, hf_repo, port, role)` are required; other fields are documentation hints.

---

## Quick start

### Prerequisites

- Linux (Ubuntu 22.04 or 24.04 tested) with **CUDA-capable GPU**
- Python 3.10+
- ~50 GB free disk for the default 80-GB-of-models matrix (or 16 GB for the minimal set)

### 3-step deploy

```bash
# 1. On a machine with internet — download all artifacts
git clone https://github.com/qiurui144/vlm-llm-benchmark.git
cd vlm-llm-benchmark
MODEL_SET=standard bash scripts/prepare_offline.sh
# MODEL_SET options:
#   minimal  (~16 GB) — VLM primary only
#   standard (~80 GB) — VLM ×2 + LLM-30B  [recommended]
#   full    (~320 GB) — all 4 models including 235B

# 2. (Optional) bundle for offline transfer to an air-gapped GPU host
tar czf vlm-llm-benchmark-bundle.tar.gz vlm-llm-benchmark/
scp vlm-llm-benchmark-bundle.tar.gz dgx:/data/

# 3. On the GPU host
cd /path/to/vlm-llm-benchmark
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
vlm-llm-benchmark/
├── run.sh                    # one-liner entry point
├── run_benchmark.py          # main scheduler
├── models.yaml               # model matrix (edit this to add/remove models)
├── common.py                 # vLLM client + shared utilities
├── requirements.txt          # httpx / pyyaml / Pillow / pynvml
├── benchmark/
│   ├── accuracy.py           # golden-set driven accuracy
│   ├── performance.py        # TTFT / throughput / concurrency / stability / PP-TG split
│   ├── translation/          # zh<->en MT: SacreBLEU/chrF/COMET + latency (L1/L2/L3)
│   ├── embedding/            # retrieval recall@k/MRR/nDCG + latency/RSS + numerical validation
│   ├── rerank/               # standalone reranker nDCG/MRR + per-pair latency
│   └── asr/                  # Chinese CER/WER/RTF (ONNX backend, graceful BLOCKED)
├── vllm_configs/
│   ├── launch_helpers.sh     # vllm serve helper functions
│   └── start_all.sh          # batch model startup (default: VLM primary only)
├── scripts/
│   ├── prepare_offline.sh    # internet host: pull wheels + models
│   ├── bootstrap.sh          # GPU host: install vLLM, link models
│   └── setup_zerotier.sh     # OPTIONAL: ZeroTier VPN for remote deploy
├── datasets/
│   ├── translation/          # zh<->en parallel corpora (custom JSONL; Flores at runtime)
│   ├── retrieval/            # embedding/rerank retrieval set (custom JSONL; builtin fallback)
│   └── asr/                  # ASR manifest template (audio + reference transcript)
├── fixtures/
│   └── README.md             # bring-your-own-data guide
├── golden/
│   └── expectations.json     # acceptance criteria (per dimension) + demo cases
└── .github/workflows/ci.yml  # lint / syntax / shellcheck
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

**Q: How do I add a new benchmark dimension?**
A: New file under `benchmark/`, register in `run_benchmark.py::BENCHMARKS`, add thresholds to `models.yaml::benchmarks`. See `CONTRIBUTING.md`.

---

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Interactions in this project are governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

The maintainers prefer small, focused PRs over sweeping refactors. New model adapters, hardware configs, and benchmark dimensions are especially welcome. **Never commit real PII to fixtures/.**

## License

[Apache License 2.0](LICENSE)

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm) — the serving stack that makes this all reasonable
- [Qwen](https://github.com/QwenLM/Qwen3) — the reference model family used in the default matrix
- [HuggingFace Hub](https://huggingface.co) — model distribution
