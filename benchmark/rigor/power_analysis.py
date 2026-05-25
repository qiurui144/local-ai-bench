"""Statistical power and sample size estimation.

You cannot defend a "no effect found" conclusion without quantifying the
power your study had to find an effect of operationally relevant size.
This module provides the prospective sample-size formulae and the
post-hoc power calculations a reviewer expects to see.

Provided
--------
- sample_size_two_means: how many per group to detect Cohen's d at given
  alpha and power.
- sample_size_two_proportions: same for binary outcomes.
- sample_size_paired: same for paired-t designs.
- post_hoc_power: given observed effect/N, what power did the test have?
- mde: minimum detectable effect given fixed N, alpha, power.

References
----------
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- Chow, S. C., Shao, J. & Wang, H. (2008). Sample Size Calculations in
  Clinical Research.
- G*Power user manual (Faul, F. et al. 2007).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SampleSizeResult:
    n_per_group: int
    total_n: int
    alpha: float
    power: float
    effect_size: float
    estimator: str
    note: str = ""


@dataclass(frozen=True)
class PowerResult:
    power: float
    alpha: float
    effect_size: float
    n_per_group: int
    estimator: str


# ---------------------------------------------------------------------------
# Two independent means (Welch / classical t)
# ---------------------------------------------------------------------------


def sample_size_two_means(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> SampleSizeResult:
    """Per-group N for detecting Cohen's d at given alpha and power.

    Uses the normal-approximation closed form (Cohen 1988 eq 2.4.1):
        n = 2 * ((z_alpha + z_beta) / d) ** 2

    For exact noncentral-t solution see the iterative `..._exact` variant
    below (slower but accurate at small N).
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie in (0, 1)")
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    n_per = int(math.ceil(n))
    return SampleSizeResult(
        n_per_group=n_per,
        total_n=2 * n_per,
        alpha=alpha,
        power=power,
        effect_size=effect_size,
        estimator="two_means_normal_approx",
        note="normal approximation; for n<20 prefer _exact variant",
    )


def sample_size_two_means_exact(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    max_iter: int = 64,
) -> SampleSizeResult:
    """Iterative noncentral-t sample size (more accurate at small N).

    Binary-search the smallest n such that the achieved power >= target.
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    lo, hi = 2, 100000
    iters = 0
    while lo < hi and iters < max_iter:
        mid = (lo + hi) // 2
        achieved = _achieved_power_two_means(effect_size, mid, alpha, two_sided)
        if achieved >= power:
            hi = mid
        else:
            lo = mid + 1
        iters += 1
    return SampleSizeResult(
        n_per_group=lo,
        total_n=2 * lo,
        alpha=alpha,
        power=power,
        effect_size=effect_size,
        estimator="two_means_noncentral_t",
    )


def _achieved_power_two_means(
    d: float, n_per: int, alpha: float, two_sided: bool
) -> float:
    """Power for two-sample t-test under Cohen's d. Uses noncentral-t.

    Falls back to a normal approximation when the noncentrality parameter
    is large enough that `nct.cdf` returns NaN (numerical underflow on
    huge n / huge nc).
    """
    if n_per < 2:
        return 0.0
    df = 2 * (n_per - 1)
    nc = d * math.sqrt(n_per / 2)  # noncentrality parameter
    if two_sided:
        crit = stats.t.ppf(1 - alpha / 2, df)
        power = 1 - stats.nct.cdf(crit, df, nc) + stats.nct.cdf(-crit, df, nc)
    else:
        crit = stats.t.ppf(1 - alpha, df)
        power = 1 - stats.nct.cdf(crit, df, nc)
    power = float(power)
    if math.isnan(power) or power < 0 or power > 1:
        # Normal-approximation fallback (Cohen 1988 eq 2.4.1 in reverse).
        z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
        power = 1 - stats.norm.cdf(z_alpha - nc)
        if two_sided:
            power += stats.norm.cdf(-z_alpha - nc)
        power = float(max(0.0, min(1.0, power)))
    return power


# ---------------------------------------------------------------------------
# Two proportions
# ---------------------------------------------------------------------------


def sample_size_two_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> SampleSizeResult:
    """Per-group N for two-proportion z-test detecting p1 vs p2.

    Use Cohen's h via arcsine transform.
    """
    if not 0 < p1 < 1 or not 0 < p2 < 1:
        raise ValueError("p1 and p2 must be in (0, 1)")
    if p1 == p2:
        raise ValueError("cannot detect zero effect")
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    h = abs(phi1 - phi2)
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / h) ** 2
    n_per = int(math.ceil(n))
    return SampleSizeResult(
        n_per_group=n_per,
        total_n=2 * n_per,
        alpha=alpha,
        power=power,
        effect_size=h,
        estimator="two_proportions_cohens_h",
    )


# ---------------------------------------------------------------------------
# Paired t
# ---------------------------------------------------------------------------


def sample_size_paired(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> SampleSizeResult:
    """Pairs needed for paired-t to detect Cohen's d_z.

    n = ((z_alpha + z_beta) / d_z) ** 2 + 1  (half the variance of independent)
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2 + 1
    n_pairs = int(math.ceil(n))
    return SampleSizeResult(
        n_per_group=n_pairs,
        total_n=n_pairs,
        alpha=alpha,
        power=power,
        effect_size=effect_size,
        estimator="paired_t_normal_approx",
    )


# ---------------------------------------------------------------------------
# Post-hoc power
# ---------------------------------------------------------------------------


def post_hoc_power_two_means(
    observed_d: float,
    n_per_group: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> PowerResult:
    """Given an observed Cohen's d and N, what power did we have?

    Use with care: post-hoc power is widely criticized for being a direct
    function of the p-value. Best practice is to report it alongside the
    test, not as a salvage for a null result.
    """
    power = _achieved_power_two_means(observed_d, n_per_group, alpha, two_sided)
    return PowerResult(
        power=power,
        alpha=alpha,
        effect_size=observed_d,
        n_per_group=n_per_group,
        estimator="two_means_noncentral_t",
    )


# ---------------------------------------------------------------------------
# Minimum detectable effect
# ---------------------------------------------------------------------------


def minimum_detectable_effect_two_means(
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """Smallest Cohen's d the design can detect at the requested power.

    Binary-searches d such that achieved power equals target.
    """
    lo, hi = 0.001, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        achieved = _achieved_power_two_means(mid, n_per_group, alpha, two_sided)
        if achieved < power:
            lo = mid
        else:
            hi = mid
    return float(hi)
