# llama_benchmark — absorbed legacy benchmark suite

This subtree is the legacy `algo-base/llama-benchmark` project (~13.5k
lines of Python), absorbed into this repository as
`benchmark/llama_benchmark/`. It is a self-contained, multi-modal model
acceptance-testing framework (`llama-bench` CLI) targeting local
inference backends — llama.cpp, Ollama, OpenAI-compatible HTTP, ONNX
(SenseVoice / Whisper), sentence-transformers, pyannote and more
(see `backends/` — 19 backend adapters).

## Task families

Enumerated from `benchmarks/` (with their dataset loaders in `datasets/`):

| Family | Dir | What it measures | Datasets |
|---|---|---|---|
| **LLM** | `benchmarks/llm/` | GSM8K / MMLU / HellaSwag accuracy, context scaling, prefill/decode performance | `gsm8k_dataset.py`, `mmlu_dataset.py`, `hellaswag_dataset.py` |
| **Whisper** | `benchmarks/whisper/` | Transcription WER / CER | `librispeech_dataset.py` |
| **ASR** | `benchmarks/asr/` | Real-time factor (RTF) for streaming-style transcription | `librispeech_dataset.py` |
| **Embedding** | `benchmarks/embedding/` | Similarity + retrieval quality | `beir_dataset.py` |
| **Rerank** | `benchmarks/rerank/` | Reranker ranking quality (nDCG etc.) | `beir_dataset.py` |
| **OCR** | `benchmarks/ocr/` | OCR accuracy + throughput | built-in samples |
| **Doc parse** | `benchmarks/docling/` | Document-parsing accuracy + throughput (docling / marker / MinerU / PyMuPDF / unstructured backends) | `docling_dataset.py` |
| **Speaker** | `benchmarks/speaker/` | Speaker **verification** and **diarization** (DER) | `ami_dataset.py`, `aishell4_dataset.py`, `callhome_dataset.py` |

Supporting layers: `core/` (config models, registry, `TaskResult`
contract), `metrics/` (NLP / ranking / speaker / performance),
`reporters/` (JSON / HTML / Markdown / compare / recommendation),
`utils/`.

## Configs and baselines

- **Configs**: `benchmark/llama_configs/` — `models.yaml`,
  `benchmarks.yaml`, `devices/k1.yaml`.
- **Measured baselines**: `benchmark/llama_baselines/k1-spacemit/` —
  three real runs on K1 (SpacemiT) plus `trend.md` trend table.
- **Tests**: `tests/llama_benchmark/` (see
  [tests/TESTING.md](../../tests/TESTING.md)) — dataset supply-chain
  contracts, diarization `TaskResult` contract, synthetic-fallback flag.

## Relationship to the main harness

Currently **zero imports in either direction**: the main
`run_benchmark.py` harness (11 dimensions, OpenAI-compatible endpoints)
and this subtree share no code paths. Reuse of this subtree's pinned
dataset loaders by the main harness is planned per the platform
positioning spec
(`docs/superpowers/specs/2026-06-11-platform-positioning.md`).

## STATUS — read before using

**The typer CLI is usable** (fixed 2026-06-11; previously tracked as a
known issue in the repo-root `RELEASE.md`):

- `typer` / `rich` are declared in the repo-root `requirements.txt`
  (alongside `pydantic` / `loguru` for the subtree's import chain).
- The CLI defaults now resolve to the bundled configs at
  `benchmark/llama_configs/{models,benchmarks}.yaml` (path computed from
  the package location, independent of CWD). A missing / wrong
  `--models` / `--benchmarks` path fails with an actionable error
  naming the bundled config locations (exit 2, no traceback).

  ```bash
  python -m benchmark.llama_benchmark.cli --help
  python -m benchmark.llama_benchmark.cli validate-config   # bundled defaults
  python -m benchmark.llama_benchmark.cli run --dry-run
  # Other commands: list-models / list-tasks / compare
  ```

  Smoke tests: `tests/llama_benchmark/test_cli.py`.

**Dead-code pruning (2026-06-11)**: the four zero-consumer modules
flagged by the coverage audit were removed — `analysis/`
(bottleneck_classifier), `utils/bandwidth_analyzer.py`,
`utils/baseline_tracker.py`, `utils/system_profiler.py` (~486 stmts;
consumer grep re-verified post-v0.3.0 before deletion).

**Dataset supply-chain status** (2026-06-11 probe, `datasets==4.5.0`;
all loaders pin `revision` to a full commit SHA and never pass
`trust_remote_code`):

| Dataset | Status |
|---|---|
| GSM8K (`openai/gsm8k`), MMLU (`cais/mmlu`), HellaSwag (`Rowan/hellaswag`), LibriSpeech | Pins **verified** — plain parquet, anonymous load works. |
| CallHome (`diarizers-community/callhome`) | **PENDING-VERIFY** — gated (HF token + terms required); pin set from API probe only, anonymous load not testable. Unauthorized access fails loudly and falls back to builtin synthetic samples with a WARNING. |
| AMI (`Edinburgh/ami`), AISHELL-4 (`speechio/aishell4`) | **Dead upstream** (HF 401 / removed). `_load_hf` raises loudly; use a local copy via `dataset_path` (expected directory layouts documented in each loader's docstring). |

When real data is unavailable every loader falls back to builtin
synthetic samples **with an explicit WARNING and a synthetic flag** — it
never silently poses as real benchmark data.
