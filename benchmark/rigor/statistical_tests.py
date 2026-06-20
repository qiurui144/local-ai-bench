"""Statistical hypothesis tests for benchmark comparisons.

Why a dedicated module rather than just calling scipy ad-hoc:
- A benchmark publishes claims like "model B beats model A on metric M".
  Such claims need a hypothesis test attached, not just point estimates,
  otherwise reviewers cannot tell whether the gap is signal or noise.
- We also need bootstrap CIs around aggregate metrics (NDCG, F1) that have
  no closed-form distribution, and Demsar (2006) Wilcoxon comparisons across
  multiple datasets.

Tests provided
--------------
- Welch's two-sample t-test (unequal variance, default for benchmark scores)
- Paired t-test (same items scored by two systems)
- Wilcoxon signed-rank (paired, non-parametric)
- Mann-Whitney U (independent, non-parametric)
- Kolmogorov-Smirnov two-sample (distribution shift detection)
- Paired bootstrap CI for arbitrary metric callable
- Permutation test (exact, for small samples)
- McNemar's test (paired binary outcomes)

References
----------
- Welch, B. L. (1947). The Generalization of Student's Problem.
- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. Biometrics.
- Mann, H. B. & Whitney, D. R. (1947). On a Test of Whether one of Two Random
  Variables is Stochastically Larger than the Other. Ann. Math. Stat.
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.
- Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple
  Data Sets. JMLR 7.
- McNemar, Q. (1947). Note on the Sampling Error of the Difference Between
  Correlated Proportions or Percentages. Psychometrika.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """Outcome of a hypothesis test.

    Attributes
    ----------
    statistic
        Test statistic (interpretation depends on the test).
    p_value
        Two-sided p-value unless `alternative` says otherwise.
    df
        Degrees of freedom where applicable (Welch / paired t).
    alternative
        "two-sided" | "less" | "greater".
    test_name
        Human-friendly label for reports / logs.
    n1, n2
        Sample sizes that produced the test.
    """

    statistic: float
    p_value: float
    df: Optional[float]
    alternative: str
    test_name: str
    n1: int
    n2: int

    def reject(self, alpha: float = 0.05) -> bool:
        """True iff p < alpha. We never make 'accept H0' claims (impossible)."""
        return self.p_value < alpha


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval around a point estimate."""

    estimate: float
    lower: float
    upper: float
    level: float
    n_resamples: int
    method: str  # "percentile" | "bca" | "paired-percentile"

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


# ---------------------------------------------------------------------------
# Parametric tests
# ---------------------------------------------------------------------------


def welch_t_test(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> TestResult:
    """Welch's two-sample t-test for unequal variances.

    Default for comparing two independent benchmark runs (different seeds /
    different items) where we should not assume equal variance.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size < 2 or b_arr.size < 2:
        raise ValueError("welch_t_test requires n>=2 in both groups")
    stat, p = stats.ttest_ind(a_arr, b_arr, equal_var=False, alternative=alternative)
    # Welch-Satterthwaite df
    va, vb = a_arr.var(ddof=1), b_arr.var(ddof=1)
    na, nb = a_arr.size, b_arr.size
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / df_den if df_den > 0 else float("nan")
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        df=float(df),
        alternative=alternative,
        test_name="welch_t",
        n1=na,
        n2=nb,
    )


def paired_t_test(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> TestResult:
    """Paired t-test for two systems scored on the same items.

    Use this when the same query / sample is scored by both system A and
    system B (the standard within-subjects benchmark layout).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size != b_arr.size:
        raise ValueError("paired_t_test requires equal-length samples")
    if a_arr.size < 2:
        raise ValueError("paired_t_test requires n>=2")
    stat, p = stats.ttest_rel(a_arr, b_arr, alternative=alternative)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        df=float(a_arr.size - 1),
        alternative=alternative,
        test_name="paired_t",
        n1=a_arr.size,
        n2=b_arr.size,
    )


# ---------------------------------------------------------------------------
# Non-parametric tests
# ---------------------------------------------------------------------------


def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> TestResult:
    """Wilcoxon signed-rank test (paired, non-parametric).

    Preferred over paired_t when scores are bounded / ordinal / heavy-tailed,
    which describes most retrieval metrics (NDCG, MRR) on small query sets.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size != b_arr.size:
        raise ValueError("wilcoxon requires equal-length samples")
    if a_arr.size < 1:
        raise ValueError("wilcoxon requires n>=1")
    # If all diffs are zero, scipy raises; handle gracefully.
    diffs = a_arr - b_arr
    if np.allclose(diffs, 0):
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            df=None,
            alternative=alternative,
            test_name="wilcoxon",
            n1=a_arr.size,
            n2=b_arr.size,
        )
    stat, p = stats.wilcoxon(a_arr, b_arr, alternative=alternative, zero_method="wilcox")
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        df=None,
        alternative=alternative,
        test_name="wilcoxon",
        n1=a_arr.size,
        n2=b_arr.size,
    )


def mann_whitney_u(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> TestResult:
    """Mann-Whitney U for two independent samples."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size < 1 or b_arr.size < 1:
        raise ValueError("mann_whitney requires n>=1 in both groups")
    stat, p = stats.mannwhitneyu(a_arr, b_arr, alternative=alternative)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        df=None,
        alternative=alternative,
        test_name="mann_whitney_u",
        n1=a_arr.size,
        n2=b_arr.size,
    )


def ks_two_sample(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> TestResult:
    """Kolmogorov-Smirnov two-sample test.

    Detects whether two distributions differ in *shape*, not just mean.
    Useful for catching tail-shifts in latency distributions that a mean
    test would miss.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size < 1 or b_arr.size < 1:
        raise ValueError("ks_two_sample requires n>=1 in both groups")
    stat, p = stats.ks_2samp(a_arr, b_arr, alternative=alternative)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        df=None,
        alternative=alternative,
        test_name="ks_2samp",
        n1=a_arr.size,
        n2=b_arr.size,
    )


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 9999,
    alternative: str = "two-sided",
    random_state: Optional[int] = 0,
) -> TestResult:
    """Exact-ish permutation test for arbitrary statistic.

    Shuffles group labels `n_resamples` times to build a null distribution
    of the statistic. Defaults to two-sided p computation. Use this when
    no closed-form null is available and sample sizes are small.
    """
    rng = np.random.default_rng(random_state)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    observed = float(statistic_fn(a_arr, b_arr))
    pool = np.concatenate([a_arr, b_arr])
    na = a_arr.size
    null_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        rng.shuffle(pool)
        null_stats[i] = statistic_fn(pool[:na], pool[na:])
    if alternative == "two-sided":
        p = (np.sum(np.abs(null_stats) >= abs(observed)) + 1) / (n_resamples + 1)
    elif alternative == "greater":
        p = (np.sum(null_stats >= observed) + 1) / (n_resamples + 1)
    elif alternative == "less":
        p = (np.sum(null_stats <= observed) + 1) / (n_resamples + 1)
    else:
        raise ValueError(f"unknown alternative: {alternative}")
    return TestResult(
        statistic=observed,
        p_value=float(p),
        df=None,
        alternative=alternative,
        test_name="permutation",
        n1=a_arr.size,
        n2=b_arr.size,
    )


def mcnemar_test(b_only: int, a_only: int, exact: bool = True) -> TestResult:
    """McNemar's test for paired binary outcomes.

    Inputs are the off-diagonal counts of a 2x2 contingency:
    - `b_only`: items where system B got it right, A wrong
    - `a_only`: items where A got it right, B wrong

    Standard for comparing two classifiers on the same test set when the
    metric is accuracy or any binary correctness indicator.
    """
    if b_only < 0 or a_only < 0:
        raise ValueError("mcnemar counts must be non-negative")
    n_disc = b_only + a_only
    if n_disc == 0:
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            df=None,
            alternative="two-sided",
            test_name="mcnemar",
            n1=0,
            n2=0,
        )
    if exact:
        # Exact binomial: under H0 each discordant pair is 50/50.
        k = min(b_only, a_only)
        p = 2 * stats.binom.cdf(k, n_disc, 0.5)
        p = min(1.0, p)
        stat = float(b_only - a_only)
    else:
        stat = (abs(b_only - a_only) - 1) ** 2 / n_disc
        p = 1 - stats.chi2.cdf(stat, df=1)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        df=None if exact else 1.0,
        alternative="two-sided",
        test_name="mcnemar",
        n1=n_disc,
        n2=n_disc,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: Sequence[float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    level: float = 0.95,
    n_resamples: int = 2000,
    method: str = "percentile",
    random_state: Optional[int] = 0,
) -> BootstrapCI:
    """Bootstrap CI for any metric callable.

    For "BCa" (bias-corrected accelerated) we delegate to scipy if available
    since the manual implementation is finicky. Default is percentile, which
    is robust and easy to reason about.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size < 2:
        raise ValueError("bootstrap_ci requires n>=2")
    rng = np.random.default_rng(random_state)
    point = float(metric_fn(arr))
    if method == "bca":
        res = stats.bootstrap(
            (arr,),
            statistic=metric_fn,
            n_resamples=n_resamples,
            confidence_level=level,
            method="BCa",
            random_state=rng,
        )
        return BootstrapCI(
            estimate=point,
            lower=float(res.confidence_interval.low),
            upper=float(res.confidence_interval.high),
            level=level,
            n_resamples=n_resamples,
            method="bca",
        )
    # percentile
    resamples = np.empty(n_resamples, dtype=float)
    n = arr.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resamples[i] = metric_fn(arr[idx])
    alpha = (1 - level) / 2
    lower = float(np.quantile(resamples, alpha))
    upper = float(np.quantile(resamples, 1 - alpha))
    return BootstrapCI(
        estimate=point,
        lower=lower,
        upper=upper,
        level=level,
        n_resamples=n_resamples,
        method="percentile",
    )


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    level: float = 0.95,
    n_resamples: int = 2000,
    random_state: Optional[int] = 0,
) -> BootstrapCI:
    """Paired bootstrap for difference metrics.

    Useful when `metric_fn` is e.g. `lambda x,y: np.mean(x) - np.mean(y)`
    and we want a CI on the gap, keeping the pairing intact.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size != b_arr.size:
        raise ValueError("paired_bootstrap requires equal-length samples")
    if a_arr.size < 2:
        raise ValueError("paired_bootstrap requires n>=2")
    rng = np.random.default_rng(random_state)
    point = float(metric_fn(a_arr, b_arr))
    resamples = np.empty(n_resamples, dtype=float)
    n = a_arr.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resamples[i] = metric_fn(a_arr[idx], b_arr[idx])
    alpha = (1 - level) / 2
    return BootstrapCI(
        estimate=point,
        lower=float(np.quantile(resamples, alpha)),
        upper=float(np.quantile(resamples, 1 - alpha)),
        level=level,
        n_resamples=n_resamples,
        method="paired-percentile",
    )


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------


def bonferroni(p_values: Sequence[float]) -> List[float]:
    """Bonferroni correction. Conservative but trivially defensible."""
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down. Strictly more powerful than Bonferroni
    while still controlling FWER. Recommended default for `k` paired
    benchmark tests."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = p_values[idx]
        candidate = (m - rank) * p
        running_max = max(running_max, min(1.0, candidate))
        adjusted[idx] = running_max
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction. Use when controlling FDR rather
    than FWER, i.e. you accept some false positives in exchange for power.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        p = p_values[idx]
        candidate = min(1.0, p * m / (rank + 1))
        running_min = min(running_min, candidate)
        adjusted[idx] = running_min
    return adjusted
