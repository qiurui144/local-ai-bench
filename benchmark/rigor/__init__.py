"""Academic-rigor foundation modules for the validation framework.

This package implements the statistical and methodological tooling required
to make benchmark claims defensible:

- statistical_tests: t-test / Mann-Whitney / Wilcoxon / KS / bootstrap CIs.
- effect_size: Cohen's d / Hedges' g / Glass's Delta / Cliff's delta.
- multi_seed_runner: seeded N-runs aggregation; rank-stability checks.
- reproducibility: seed pin, dependency snapshot, hardware capture.
- calibration: ECE, Brier score, reliability diagrams.
- inter_rater: Cohen / Fleiss kappa, Krippendorff's alpha.
- ablation: factorial design, one-variable-at-a-time orchestration.
- cross_validation: k-fold, stratified, leave-one-out, nested CV.
- power_analysis: sample size estimation, post-hoc power.
- ood_assessment: domain shift, temporal drift, OOD detection helpers.

All modules are pure Python (numpy/scipy where used) with no external service
dependencies; they are designed to be run inside the bench harness and inside
CI pipelines without GPU.

Key references baked into module docstrings:
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods.
- Lin, L. (1989). A Concordance Correlation Coefficient.
- Naeini et al. (2015). Obtaining Well-Calibrated Probabilities Using Bayesian Binning.
- Krippendorff, K. (2004). Content Analysis.
- Efron, B. (1979). Bootstrap Methods.
- Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Datasets.
"""

__all__ = [
    "statistical_tests",
    "effect_size",
    "multi_seed_runner",
    "reproducibility",
    "calibration",
    "inter_rater",
    "ablation",
    "cross_validation",
    "power_analysis",
    "ood_assessment",
]
