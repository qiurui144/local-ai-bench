# Contributing

Thanks for considering a contribution. This framework is meant to
support academic-grade benchmark claims, so contributions need to
clear a higher rigor bar than typical OSS feature requests.

## What we accept

- New metrics with a literature citation and a unit-test fixture
  against a published reference value (or a derivation in the PR).
- New ablation knobs to existing modules.
- New labs that exercise existing modules on new synthetic data.
- New case studies documenting a real or representative incident.
- Bug fixes that come with a regression test reproducing the bug
  first.

## What we do not accept

- "Drop-in better-than-everything" metrics with no peer-reviewed
  basis.
- Replacing well-cited methods with toy implementations.
- Lowering existing thresholds without a data-driven justification.
- Removing tests because "they are flaky" (write the deflake first).

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
- [ ] If your change can affect numbers: a note in `RELEASE.md`
  describing the expected delta.

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

- Pure Python where possible; depend on numpy/scipy where needed.
- Avoid heavy ML framework dependencies in `benchmark/rigor/`;
  keep that subpackage lean enough to import on CI containers
  without GPU.
- Module docstrings explain *why* the module exists, not just *what*
  it does. Aim for the level of explanation a competent ML
  engineer needs to choose between options.

## Reviewing other contributions

- Be specific: "The Cohen's d implementation does not use Welch's
  variance" beats "this is wrong."
- Suggest tests when you suspect a bug; don't just file an issue.
- When the literature is ambiguous, link the paper; do not assume
  shared context.

## Getting help

Open a discussion before a sizeable PR; structural changes (new
top-level subpackages) should be sketched and reviewed before
implementation.
