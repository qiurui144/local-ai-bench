# Contributing to local-ai-bench

Thanks for your interest in improving this benchmark. The project is two
things at once: a small, reproducible **benchmark harness** (the 11
dimensions driven by `run_benchmark.py`) and an **academic-grade
validation framework** (`benchmark/rigor/`, `benchmark/rag/`) meant to
support citable benchmark claims. Contributions to the harness should
keep it small and focused; contributions to the validation framework
need to clear a higher rigor bar than typical OSS feature requests.

This file is the single contributing guide for the repository.

## What we want

Harness:

- **New model adapters** in `models.yaml` (any OpenAI-compatible endpoint
  works out of the box).
- **New benchmark dimensions** beyond the 11 currently shipped
  (`accuracy, ttft, throughput, prefill_decode, concurrency, stability,
  translation, embedding, rerank, asr, scenarios`) — e.g.
  cost-per-1k-tokens or energy efficiency.
- **More golden-set patterns** in `golden/expectations.json` — we ship a
  small synthetic demo; reference patterns for new domains (medical,
  financial, legal, retail) are valuable.
- **Bug fixes** with a regression test reproducing the bug first.
- **Hardware-specific configs** in `vllm_configs/` (currently A100/H100
  focused — Ada / MI300X / Habana welcome).

Validation framework:

- New metrics with a literature citation and a unit-test fixture against
  a published reference value (or a derivation in the PR).
- New ablation knobs to existing modules.
- New labs that exercise existing modules on new synthetic data.
- New case studies documenting a real or representative incident.

## What we don't want

- **Real PII in `fixtures/`**. Never commit real chat screenshots, ID
  photos, contracts, or any image with identifiable people. The repo's
  `.gitignore` excludes binary fixtures by default — keep it that way.
- **Vendor-locked code**. The harness talks OpenAI-compatible HTTP.
  Don't add hard dependencies on a specific provider's SDK in the core
  path.
- **Dependency creep**. `requirements.txt` is deliberately kept small
  (currently 11 pinned entries — httpx, pyyaml, Pillow, pydantic,
  loguru, pynvml, numpy, scipy, pytest, pytest-asyncio, sacrebleu — plus
  documented optional extras). New dependencies need justification.
- "Drop-in better-than-everything" metrics with no peer-reviewed basis.
- Replacing well-cited methods with toy implementations.
- Lowering existing thresholds without a data-driven justification.
- Removing tests because "they are flaky" (write the deflake first).

## Workflow

1. Open an issue first for non-trivial changes — a 5-line discussion can
   save a 500-line PR. Structural changes (new top-level subpackages)
   should be sketched and reviewed before implementation.
2. Fork, create a feature branch (`feat/short-description`), do your work.
3. Run `ruff check .`, `python -m py_compile $(git ls-files '*.py')`, and
   `python -m pytest tests/ -q` locally before pushing (see
   [tests/TESTING.md](../tests/TESTING.md)).
4. Open a PR. Describe **what** changed, **why**, and **how you tested** it.
5. Be patient — this is a side project for the maintainers; reviews may
   take a few days.

## PR checklist

- [ ] Each new module has a docstring naming the algorithm, the
  literature reference, and the expected use case.
- [ ] Each new metric has at least one unit test that reproduces a
  known reference output (e.g. NDCG on the textbook example).
- [ ] No file uses arbitrary code execution (`ast.literal_eval`
  or `json.loads` for parsing structured data; no `exec`). No
  filename or variable contains the standalone word that means
  "code-execution" — prefer `evaluation` / `evaluator` / `assessor`.
- [ ] All public symbols have type annotations.
- [ ] Linting is clean (ruff or flake8 default rules).
- [ ] If you added a metric: a sentence in `BASELINES.md` recording
  any default threshold you suggested.
- [ ] If your change can affect numbers: a note in
  [`RELEASE.md`](../RELEASE.md) (repo root) describing the expected
  delta.

## Methodological guidelines

- **No single-seed claims.** If your contribution argues "this
  metric is better," demonstrate it with >=3 seeds and report
  mean +- std.
- **Effect sizes alongside p-values.** Significant-but-tiny is not
  a contribution.
- **Bucketed reporting.** If you introduce a metric, show how
  `bucketed_metrics` would surface it per-domain.
- **Reproducibility.** Any benchmark snippet in the PR description
  must include a snapshot path.

## Coding style

- Python: 4-space indent, `ruff` for lint, no unused imports, type hints
  on public functions. Pure Python where possible; depend on numpy/scipy
  where needed.
- Avoid heavy ML framework dependencies in `benchmark/rigor/`;
  keep that subpackage lean enough to import on CI containers
  without GPU.
- Shell: `set -euo pipefail`, prefer `shellcheck`-clean scripts.
- YAML: 2-space indent.
- Comments in English. Minimal — explain *why*, not *what* (the code
  already says what). Module docstrings explain *why* the module exists;
  aim for the level of explanation a competent ML engineer needs to
  choose between options.

## Tests

- `tests/` holds a 436-test offline suite (no network, runs in CI with
  `TRANSLATION_OFFLINE=1`). See [tests/TESTING.md](../tests/TESTING.md)
  for the suite map, conventions, and how to run it.
- Every new benchmark dimension must ship synthetic-data unit tests in
  `tests/<dimension>/`, mirroring the existing per-dimension layout.
- For golden-set patterns, ship a synthetic example users can reproduce
  without your private data.
- Bug fixes come with a regression test that reproduces the bug first.

## Reviewing other contributions

- Be specific: "The Cohen's d implementation does not use Welch's
  variance" beats "this is wrong."
- Suggest tests when you suspect a bug; don't just file an issue.
- When the literature is ambiguous, link the paper; do not assume
  shared context.

## DCO / Sign-off

Not required. A clear PR description and clean commits are enough.

## Security

If you find a security issue (e.g. a way the harness could leak
credentials or write outside its working dir), please email the
maintainer privately rather than opening a public issue. Contact info is
in the repo profile.

## License

By contributing, you agree your contributions are licensed under the
Apache License 2.0 (same as the project).

## Getting help

Open a discussion before a sizeable PR. Interactions in this project are
governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
