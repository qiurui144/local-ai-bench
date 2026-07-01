# Academic Rigor Principles

This benchmark publishes claims about model and pipeline quality. Those
claims have to be defensible to a critical reviewer. The principles
below are not aspirational; they are the explicit contract the rest of
the framework enforces.

## 1. No claim without a hypothesis test

Every "A beats B" claim must be paired with:

- a test statistic and p-value from `benchmark.rigor.statistical_tests`,
- an effect size from `benchmark.rigor.effect_size`,
- and, when relevant, a multiple-comparison correction
  (Bonferroni / Holm / BH).

Reference: Demsar, J. (2006). *Statistical Comparisons of Classifiers
over Multiple Data Sets*. JMLR 7. We follow the Wilcoxon-over-paired-t
recommendation for retrieval and quality metrics, which are usually
non-normal.

## 2. No effect size without a magnitude judgment

A statistically significant gap of d=0.05 with n=10,000 is not the same
finding as a gap of d=0.6 with n=30. Effect-size magnitude bands per
Cohen (1988) are reported alongside p-values; whoever consumes the
result can decide whether to act.

## 3. No single-seed numbers in the headline

Per the CLAUDE.md "调研/算法项目工作纪律" §7 rule, every claim of
"X improves over baseline" must come from >=3 seeds:

- report mean +- std (not the best of three),
- declare improvement only if mean gap >= 2 * typical_std,
- detect and surface rank flips via
  `benchmark.rigor.multi_seed_runner.detect_rank_flips`.

A single-seed comparison may appear as a smoke-test row only when
labeled as such.

## 4. No comparison without confidence intervals

Point estimates without CIs are advisory. Use either:

- t-based CI when sample size and approximate normality justify it,
- bootstrap CI (`bootstrap_ci`, `paired_bootstrap_ci`) for any metric
  with no closed-form distribution (NDCG, F1, MRR aggregated).

## 5. No subjective rating without inter-rater reliability

Any benchmark that uses human judges or LLM judges has to publish
Cohen / Fleiss / Krippendorff statistics:

- pairwise Cohen's kappa for two raters,
- Fleiss / Krippendorff for >=3 raters,
- Gwet's AC1 when one category dominates (kappa paradoxes).

Raw agreement is reported alongside chance-corrected statistics as a
sanity check.

## 6. No probabilistic judge without calibration

Judges that emit confidence numbers are calibrated and the diagnostic
(ECE, Brier, reliability curve) published. Mis-calibrated probabilities
are worse than no probabilities because they encourage over-trust.

## 7. No "no effect found" without a power analysis

Null results require a power analysis showing the design could have
detected an operationally important effect. Use
`benchmark.rigor.power_analysis` for prospective and post-hoc reports.

## 8. No ablation without one-variable-at-a-time isolation

`benchmark.rigor.ablation` provides the canonical OAT and factorial
designs. Each ablation report includes a baseline row plus per-knob
deltas; we never publish a ranked list of "best configurations" without
the OAT decomposition behind it.

## 9. No production claim without offline-online alignment

`benchmark.rag.offline_online_alignment.AlignmentChecker` runs on
every release; alignment less than 0.7 Spearman or PSI > 0.25 is a
flag. We don't trust offline numbers in production unless alignment
was verified.

## 10. No release without reproducibility snapshot

Every shipped benchmark run is paired with
`benchmark.rigor.reproducibility.ReproducibilitySnapshot.capture()`
output. Code SHA, pip freeze, hardware spec, and data SHA256s are
mandatory so that a future investigator can recreate the run on the
same versions.

## 11. Bucketed reporting is non-negotiable

Aggregates lie; subgroups break. Every metric on every release must be
reported by domain / difficulty / language as well as overall.
`benchmark.rag.retrieval_metrics.bucketed_metrics` and
`benchmark.rigor.ood_assessment.subgroup_audit` are the entry points.

## 12. Adversarial robustness is a published number

LLM judges are attackable. The perturbation suite
(`benchmark.rag.judge_attacks.adversarial_perturbation_suite`) is run
on every judge release; stability rates are reported.

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
  Sciences*, 2nd ed.
- Demsar, J. (2006). Statistical Comparisons of Classifiers over
  Multiple Data Sets. JMLR.
- Efron, B. (1979). Bootstrap Methods. Annals of Statistics.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D.,
  Meger, D. (2018). Deep Reinforcement Learning that Matters. AAAI.
- Bouthillier, X. et al. (2021). Accounting for Variance in Machine
  Learning Benchmarks. MLSys.
- Pineau, J. et al. (2021). Improving Reproducibility in Machine
  Learning Research. JMLR.
- Naeini, M. P., Cooper, G. F., Hauskrecht, M. (2015). Obtaining Well
  Calibrated Probabilities Using Bayesian Binning. AAAI.
- Krippendorff, K. (2004). Content Analysis. Sage.
- Buckley, C. & Voorhees, E. M. (2004). Retrieval Evaluation with
  Incomplete Information. SIGIR.
- Chapelle, O. et al. (2009). Expected Reciprocal Rank for Graded
  Relevance. CIKM.
- Moffat, A. & Zobel, J. (2008). Rank-Biased Precision. TOIS.
- Cormack, G. V. et al. (2009). Reciprocal Rank Fusion. SIGIR.
- Es, S. et al. (2023). RAGAs. EACL Demos.
- Liu, Y. et al. (2023). G-Eval. EMNLP.
- Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench. NeurIPS.
- Bohnet, B. et al. (2022). Attributed Question Answering.
