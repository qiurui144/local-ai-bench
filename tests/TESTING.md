# Testing Guide

The repository ships an offline suite (`pytest --collect-only -q` reports the
current count). All tests run without network, without GPUs, and without any
model server — CI runs the full suite on every push.

## How to run

```bash
# Full suite (from the repo root; conftest.py adds the root to sys.path)
python -m pytest tests/ -q

# Exactly what CI runs (.github/workflows/ci.yml): offline translation mode
TRANSLATION_OFFLINE=1 python -m pytest tests/ -q

# One sub-suite / one file / one test
python -m pytest tests/rigor/ -q
python -m pytest tests/translation/test_flores_loading.py -q
python -m pytest tests/scenarios/test_judge.py::TestJudge -q

# Coverage (pytest-cov is optional, not in requirements.txt)
pip install pytest-cov
python -m pytest tests/ -q --cov=benchmark --cov=run_benchmark --cov=common
```

`TRANSLATION_OFFLINE=1` forces the translation dimension onto its
built-in fallback sentence set so no HuggingFace download is attempted.
Set it for any environment without (or where you don't want) network.

## Suite map

| Path | Covers |
|---|---|
| `tests/test_run_benchmark.py` | Orchestrator (`run_benchmark.py`): exit-code contract, `--skip` / `--seeds` semantics, worst-of-N verdicts, per-dimension wiring. |
| `tests/test_common_http.py` | Shared OpenAI-compatible async HTTP client in `common.py` (uses `pytest-asyncio`). |
| `tests/scenarios/` | Real-scenario dimension: case schema/logic, registry, runner (judge ≠ tested-model rule), L2 judge, seed datasets, wechat-intent and article-knowledge scenarios. |
| `tests/translation/` | Translation metrics (BLEU/chrF/term-match thresholds) plus Flores-200 loader contracts: revision pin, offline fallback, env overrides. |
| `tests/embedding/` | Embedding dimension: cosine/retrieval accuracy, RSS dual-source handling, zero-vector FAIL. |
| `tests/rerank/` | Rerank dimension: ranking metrics and the "per-pair latency reported but not gated" contract. |
| `tests/asr/` | ASR dimension: transcription scoring and BLOCKED→SKIP behavior when optional deps are absent. |
| `tests/rag/` | 12-chapter RAG validation framework: retrieval metrics, groundedness, answer relevance, judge prompts/calibration/attacks, reranker, drift detection, canary, offline/online alignment, regression CI. |
| `tests/rigor/` | Statistical-rigor modules: statistical tests, effect size, power analysis, multi-seed runner, cross-validation, calibration, inter-rater, OOD assessment, ablation, reproducibility snapshot. |
| `tests/performance/` | Prefill/decode (PP/TG) split and throughput statistics. |
| `tests/llama_benchmark/` | Absorbed llama_benchmark subtree: dataset supply-chain contracts (no `trust_remote_code`, pinned `revision` SHAs, loud fallback), diarization `TaskResult` contract, synthetic-fallback flag. |

## Conventions

- **Offline-only / no network in tests.** No test may hit HuggingFace,
  a model endpoint, or any URL. Network-touching code is exercised via
  fakes; CI has no credentials and must stay green.
- **Fake `datasets` module idiom.** Loader tests never import the real
  `datasets` package. They install a stub capturing the call:
  `monkeypatch.setitem(sys.modules, "datasets", fake_mod)` — then assert
  on call shape (pinned `revision=`, no `trust_remote_code`). See
  `tests/translation/test_flores_loading.py` and
  `tests/llama_benchmark/test_dataset_loading.py`.
- **Monkeypatch module-ref idiom.** Patch the attribute on the module
  that *uses* it, not the definition site:
  `monkeypatch.setattr(accuracy, "infer_embedding", fake)` — so the
  patched reference is the one the code under test resolves.
- **TDD policy.** Bug fixes land with a regression test that reproduces
  the bug first (red), then the fix (green). New features ship their
  tests in the same PR; per `docs/contributing.md`, every new metric
  reproduces a published reference value where one exists.
- **Synthetic data only.** Fixtures are synthetic and PII-free
  (`scripts/check_no_real_images.sh` guards images).

## Adding tests for a new dimension / scenario

New benchmark dimension (`benchmark/<dim>/`):

1. Create `tests/<dim>/` with an `__init__.py`, mirroring the existing
   per-dimension layout (e.g. `tests/embedding/`).
2. Test the metric math on synthetic inputs with known expected values;
   fake all HTTP via the monkeypatch module-ref idiom — no live server.
3. Cover the verdict edges: PASS/WARN/FAIL thresholds, BLOCKED/SKIP
   when optional deps or data are missing, and any provenance capping.
4. Register the dimension in `run_benchmark.py::BENCHMARKS` and extend
   `tests/test_run_benchmark.py` so `--skip <dim>` and the exit-code
   contract cover it.

New scenario (scenarios dimension):

1. Add the case set under `datasets/scenarios/` (synthetic or curated
   via `scripts/curate_scenario_case.py`; honest `provenance` field —
   synthetic caps at WARN).
2. Add `tests/scenarios/test_<scenario>.py` covering case-schema
   validity, registry pickup, and scoring logic against fixed fixtures.
3. Run `python -m pytest tests/scenarios/ -q` plus the full suite before
   pushing.
