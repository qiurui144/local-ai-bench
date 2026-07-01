# Documentation Index

This directory is for stable project documentation. Generated run logs,
scratch notes, and date-stamped audit records are local artifacts and must not
be promoted here unless they are rewritten as maintained documentation.

## Naming Rules

- Root-level conventional files keep their standard uppercase names:
  `README.md`, `DEVELOP.md`, `RELEASE.md`, `SECURITY.md`, `LICENSE`.
- Files under `docs/` use lowercase kebab-case, for example
  `academic-rigor.md` and `deploy-targets.md`.
- Case studies live under `docs/case-studies/` and use
  `caseNN-short-name.md`.
- Public platform reports live under `reports/` and use fixed lowercase names
  such as `amd-windows.en.md` or `model-matrix.en.md`.
- Local-only records stay under ignored paths: `output/`,
  `reports/YYYY-*`, `reports/YYYY/`, `reports/runs/`, and `.remember/`.
- Tracked Markdown must be UTF-8 compatible. ASCII is fine because it is a
  UTF-8 subset.

## Stable Documents

| File | Purpose |
|---|---|
| [academic-rigor.md](academic-rigor.md) | Statistical rigor requirements for benchmark claims |
| [amd-intel-linux-test-plan.md](amd-intel-linux-test-plan.md) | AMD/Intel Linux sequencing after Windows runs |
| [baselines.md](baselines.md) | Baseline and threshold policy |
| [citation.md](citation.md) | Citation guidance |
| [code-of-conduct.md](code-of-conduct.md) | Code of conduct |
| [contributing.md](contributing.md) | Contribution workflow and methodology rules |
| [cross-platform-compare.md](cross-platform-compare.md) | Cross-platform comparison semantics |
| [deploy-targets.md](deploy-targets.md) | Multi-platform deployment SOP |
| [reproducibility.md](reproducibility.md) | Reproducibility policy |

## Reports

The public `reports/` directory intentionally contains only curated fixed-name
reports. Date-stamped reports and raw run records are ignored because they can
contain local machine details, transient logs, or unreviewed benchmark output.
