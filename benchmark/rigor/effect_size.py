"""Effect size estimators.

p-values alone tell you *whether* a gap exists; effect sizes tell you
*how big* it is. A benchmark claim like "model B improves over A
(p=0.001)" is only meaningful with an attached effect: an improvement
of 0.005 NDCG with d=0.05 is statistical noise made significant by
large N; an improvement of 0.05 NDCG with d=0.6 is operationally
important.

Estimators provided
-------------------
- Cohen's d        - standardized mean difference (pooled SD)
- Hedges' g        - Cohen's d with small-sample bias correction
- Glass's Delta    - uses only control group SD (useful when treatment
                     inflates variance)
- Cliff's delta    - non-parametric, ordinal-scale safe
- Odds ratio       - binary outcomes
- Rank-biserial    - companion to Mann-Whitney U
- Common Language Effect Size (CLES) - probability that a random
  draw from A exceeds a random draw from B; intuitive for non-stat readers.

References
----------
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences,
  2nd ed. Lawrence Erlbaum.
- Hedges, L. V. (1981). Distribution Theory for Glass's Estimator.
- Glass, G. V. (1976). Primary, Secondary, and Meta-Analysis of Research.
- Cliff, N. (1993). Dominance statistics.
- Sullivan, G. M. & Feinn, R. (2012). Using Effect Size - or Why the P
  Value Is Not Enough. J Grad Med Educ.
- McGraw, K. O. & Wong, S. P. (1992). A common language effect size
  statistic. Psychological Bulletin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Magnitude classification per Cohen (1988); standard pedagogy.
# Note: small/medium/large thresholds are conventions, not laws of nature.
# Reviewers may demand domain-specific calibration; we surface both the raw
# value and the conventional label.
# ---------------------------------------------------------------------------

COHENS_D_BANDS = (
    (0.2, "small"),
    (0.5, "medium"),
    (0.8, "large"),
)


@dataclass(frozen=True)
class EffectSize:
    value: float
    estimator: str
    magnitude: str  # "negligible" | "small" | "medium" | "large"


def _classify_d(d: float) -> str:
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    for threshold, label in COHENS_D_BANDS[::-1]:
        if abs_d >= threshold:
            return label
    return "negligible"


# ---------------------------------------------------------------------------
# Parametric (continuous-scale) effect sizes
# ---------------------------------------------------------------------------


def cohens_d(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Cohen's d using pooled standard deviation.

    d = (mean_a - mean_b) / s_pooled
    where s_pooled = sqrt(((na-1) s_a^2 + (nb-1) s_b^2) / (na + nb - 2))
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    na, nb = a_arr.size, b_arr.size
    if na < 2 or nb < 2:
        raise ValueError("cohens_d requires n>=2 in both groups")
    sa, sb = a_arr.var(ddof=1), b_arr.var(ddof=1)
    pooled = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    if pooled == 0:
        return EffectSize(value=0.0, estimator="cohens_d", magnitude="negligible")
    d = (a_arr.mean() - b_arr.mean()) / pooled
    return EffectSize(value=float(d), estimator="cohens_d", magnitude=_classify_d(d))


def hedges_g(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Hedges' g: Cohen's d with small-sample bias correction.

    Multiply Cohen's d by J = 1 - 3/(4*(na+nb)-9). Prefer this when
    n_total < 50, which is common with curated benchmark golden sets.
    """
    d = cohens_d(a, b).value
    na, nb = len(a), len(b)
    j = 1 - 3 / (4 * (na + nb) - 9)
    g = d * j
    return EffectSize(value=float(g), estimator="hedges_g", magnitude=_classify_d(g))


def glass_delta(treatment: Sequence[float], control: Sequence[float]) -> EffectSize:
    """Glass's Delta uses only the control SD.

    delta = (mean_treatment - mean_control) / sd_control

    Use when the treatment is suspected to change variance (e.g. a model
    that is sometimes wildly off-baseline), so pooling SDs is unfair.
    """
    t_arr = np.asarray(treatment, dtype=float)
    c_arr = np.asarray(control, dtype=float)
    if c_arr.size < 2:
        raise ValueError("glass_delta requires control n>=2")
    sd_c = c_arr.std(ddof=1)
    if sd_c == 0:
        return EffectSize(value=0.0, estimator="glass_delta", magnitude="negligible")
    delta = (t_arr.mean() - c_arr.mean()) / sd_c
    return EffectSize(
        value=float(delta),
        estimator="glass_delta",
        magnitude=_classify_d(delta),
    )


# ---------------------------------------------------------------------------
# Non-parametric / ordinal-scale effect sizes
# ---------------------------------------------------------------------------


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Cliff's delta: probability that a > b minus probability that a < b.

    Range [-1, 1]. Non-parametric, robust to outliers, and applicable to
    ordinal metrics (judge labels 1..5, rank positions). We compute the
    O(n*m) pairwise version for clarity; for n*m > 1e7 prefer the
    rank-based formulation.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("cliffs_delta requires non-empty samples")
    # Vectorized pairwise comparison.
    diff = a_arr[:, None] - b_arr[None, :]
    n_gt = float(np.sum(diff > 0))
    n_lt = float(np.sum(diff < 0))
    total = a_arr.size * b_arr.size
    delta = (n_gt - n_lt) / total
    # Romano et al. magnitude bands.
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "negligible"
    elif abs_d < 0.33:
        mag = "small"
    elif abs_d < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return EffectSize(value=float(delta), estimator="cliffs_delta", magnitude=mag)


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Rank-biserial correlation; the companion effect size to Mann-Whitney U.

    r_b = 1 - 2*U / (n1 * n2)
    """
    from scipy import stats as _stats  # local import to keep top minimal

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("rank_biserial requires non-empty samples")
    u, _ = _stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
    rb = 1 - (2 * u) / (a_arr.size * b_arr.size)
    return EffectSize(
        value=float(rb),
        estimator="rank_biserial",
        magnitude=_classify_d(rb),
    )


# ---------------------------------------------------------------------------
# Binary-outcome effect sizes
# ---------------------------------------------------------------------------


def odds_ratio(
    a_success: int,
    a_total: int,
    b_success: int,
    b_total: int,
    haldane_correction: bool = True,
) -> EffectSize:
    """Odds ratio for two binary cohorts.

    OR = (a_success * b_failure) / (a_failure * b_success)

    Haldane-Anscombe (+0.5) correction prevents division-by-zero when any
    cell is empty; standard in epidemiology.
    """
    if a_total < a_success or b_total < b_success:
        raise ValueError("success count exceeds total")
    a_fail = a_total - a_success
    b_fail = b_total - b_success
    if haldane_correction:
        a_s, a_f, b_s, b_f = (
            a_success + 0.5,
            a_fail + 0.5,
            b_success + 0.5,
            b_fail + 0.5,
        )
    else:
        a_s, a_f, b_s, b_f = a_success, a_fail, b_success, b_fail
    if a_f == 0 or b_s == 0:
        return EffectSize(value=float("inf"), estimator="odds_ratio", magnitude="large")
    or_val = (a_s * b_f) / (a_f * b_s)
    # Magnitude bands for OR per Chen et al. 2010 small=1.5, medium=3.5, large=9.
    if or_val < 1.5 and or_val > 1 / 1.5:
        mag = "negligible"
    elif or_val < 3.5 and or_val > 1 / 3.5:
        mag = "small"
    elif or_val < 9 and or_val > 1 / 9:
        mag = "medium"
    else:
        mag = "large"
    return EffectSize(value=float(or_val), estimator="odds_ratio", magnitude=mag)


def risk_difference(
    a_success: int,
    a_total: int,
    b_success: int,
    b_total: int,
) -> EffectSize:
    """Absolute risk difference: P(A success) - P(B success). Range [-1, 1]."""
    if a_total == 0 or b_total == 0:
        raise ValueError("risk_difference requires non-zero totals")
    p_a = a_success / a_total
    p_b = b_success / b_total
    rd = p_a - p_b
    abs_rd = abs(rd)
    if abs_rd < 0.05:
        mag = "negligible"
    elif abs_rd < 0.10:
        mag = "small"
    elif abs_rd < 0.20:
        mag = "medium"
    else:
        mag = "large"
    return EffectSize(value=float(rd), estimator="risk_difference", magnitude=mag)


# ---------------------------------------------------------------------------
# Intuitive / communication-friendly effect sizes
# ---------------------------------------------------------------------------


def cles(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Common Language Effect Size: P(random a > random b).

    Easier to communicate to non-stat audiences than Cohen's d. CLES=0.5
    means coin-flip; CLES>0.5 means A tends to score higher.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("cles requires non-empty samples")
    diff = a_arr[:, None] - b_arr[None, :]
    wins = float(np.sum(diff > 0))
    ties = float(np.sum(diff == 0))
    total = a_arr.size * b_arr.size
    p = (wins + 0.5 * ties) / total
    deviation = abs(p - 0.5)
    if deviation < 0.05:
        mag = "negligible"
    elif deviation < 0.1:
        mag = "small"
    elif deviation < 0.2:
        mag = "medium"
    else:
        mag = "large"
    return EffectSize(value=float(p), estimator="cles", magnitude=mag)


# ---------------------------------------------------------------------------
# Combined reporting helper
# ---------------------------------------------------------------------------


def effect_size_report(a: Sequence[float], b: Sequence[float]) -> dict:
    """One-shot summary of multiple effect sizes for a publication table.

    Returns dict with keys: cohens_d, hedges_g, glass_delta_a, glass_delta_b,
    cliffs_delta, rank_biserial, cles. Use this when you want all the
    common estimators side-by-side so readers can pick.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return {
        "cohens_d": cohens_d(a_arr, b_arr),
        "hedges_g": hedges_g(a_arr, b_arr),
        "glass_delta_treatment_a": glass_delta(a_arr, b_arr),
        "glass_delta_treatment_b": glass_delta(b_arr, a_arr),
        "cliffs_delta": cliffs_delta(a_arr, b_arr),
        "rank_biserial": rank_biserial(a_arr, b_arr),
        "cles": cles(a_arr, b_arr),
        "n_a": int(a_arr.size),
        "n_b": int(b_arr.size),
        "mean_a": float(a_arr.mean()),
        "mean_b": float(b_arr.mean()),
    }
