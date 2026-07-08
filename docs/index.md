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
- Curated report entry points live under `reports/index.en.md`,
  `reports/index.zh.md`, and `reports/selection/`.
- Platform reports live under `reports/platforms/<platform>/` and keep
  English/Chinese pairs such as `index.en.md` and `index.zh.md`.
- CPU, GPU/iGPU, NPU, runtime, and workflow-risk paths stay split when the
  platform has separate execution paths.
- Legacy root-level reports such as `amd-windows.en.md` remain as evidence and
  compatibility links, but new readers should start from `reports/index.*.md`.
- Local-only records stay under ignored paths: `output/`,
  `reports/YYYY-*`, `reports/YYYY/`, `reports/runs/`, and `.remember/`.
- Tracked Markdown must be UTF-8 compatible. ASCII is fine because it is a
  UTF-8 subset.
- Hostnames, IP addresses, account names, passwords, and reusable connection
  strings must not appear in tracked docs or reports.

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
| [k3-realistic-stress-plan.md](k3-realistic-stress-plan.md) | K3 product-like mixed-traffic stress test contract |
| [k3-realistic-stress-plan.zh.md](k3-realistic-stress-plan.zh.md) | K3 product-like mixed-traffic stress test contract, Chinese version |
| [k3-source-runtime-and-long-context.md](k3-source-runtime-and-long-context.md) | K3 source runtime comparison and aviation-manual long-context gating |
| [rockchip-rknn3-model-cache.md](rockchip-rknn3-model-cache.md) | RK3588/RK182X RKNN3 model cache and sync workflow |
| [spacemit-model-zoo.md](spacemit-model-zoo.md) | SpacemiT model_zoo data acquisition, local cache layout, and K3 invocation map |

## Reports

The public `reports/` directory contains curated fixed-name reports and
bilingual platform report pairs.

| Entry point | Purpose |
|---|---|
| [../reports/index.en.md](../reports/index.en.md) | English reports index |
| [../reports/index.zh.md](../reports/index.zh.md) | Chinese reports index |
| [../reports/selection/model-selection.en.md](../reports/selection/model-selection.en.md) | English cross-platform model-selection entry point |
| [../reports/selection/model-selection.zh.md](../reports/selection/model-selection.zh.md) | Chinese cross-platform model-selection entry point |
| [../reports/evidence/k3-riscv-32g.evidence.en.md](../reports/evidence/k3-riscv-32g.evidence.en.md) | K3 evidence map and run-log provenance |
| [../reports/platforms/k3-riscv-32g/index.en.md](../reports/platforms/k3-riscv-32g/index.en.md) | K3 RISC-V 32G platform report |
| [../reports/platforms/amd-windows/index.en.md](../reports/platforms/amd-windows/index.en.md) | AMD Windows CPU/iGPU/NPU report |
| [../reports/platforms/intel-windows/index.en.md](../reports/platforms/intel-windows/index.en.md) | Intel Windows CPU/iGPU/NPU report |
| [../reports/platforms/rk3588/index.en.md](../reports/platforms/rk3588/index.en.md) | RK3588/RK1828 platform report |

Date-stamped reports and raw run records are ignored because they can contain
local machine details, transient logs, or unreviewed benchmark output.
