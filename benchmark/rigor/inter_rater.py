"""Inter-rater reliability for human and LLM-judge consensus.

Whenever a benchmark relies on subjective judgment (relevance labels,
groundedness flags, judge-as-evaluator outputs), the *agreement* among
judges is what gives the labels evidentiary weight. Raw agreement is
inflated by chance, so we use chance-corrected statistics.

Statistics provided
-------------------
- Cohen's kappa             - two raters, categorical
- Weighted Cohen's kappa    - two raters, ordinal (linear or quadratic weights)
- Fleiss's kappa            - >=2 raters, categorical, fixed N per item
- Krippendorff's alpha      - arbitrary raters + scales + missing data
- Percent agreement         - sanity check; report alongside kappa
- Gwet's AC1                - kappa alternative that resists prevalence/skew bias

Practical guidance (Landis & Koch 1977 - applies to all kappa variants):
   < 0.00  poor
   0.00 - 0.20  slight
   0.21 - 0.40  fair
   0.41 - 0.60  moderate
   0.61 - 0.80  substantial
   0.81 - 1.00  almost perfect

References
----------
- Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales.
- Fleiss, J. L. (1971). Measuring Nominal Scale Agreement Among Many Raters.
- Krippendorff, K. (2004). Content Analysis: An Introduction to Its
  Methodology. Sage.
- Gwet, K. L. (2008). Computing inter-rater reliability and its variance
  in the presence of high agreement.
- Landis, J. R. & Koch, G. G. (1977). The measurement of observer
  agreement for categorical data. Biometrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AgreementResult:
    coefficient: float
    estimator: str
    interpretation: str
    n_items: int
    n_raters: int
    extra: Dict[str, float]


def _landis_koch(kappa: float) -> str:
    if kappa < 0:
        return "poor"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost_perfect"


# ---------------------------------------------------------------------------
# Two-rater
# ---------------------------------------------------------------------------


def percent_agreement(a: Sequence, b: Sequence) -> float:
    """Raw fraction of matching ratings. Always report alongside kappa to
    avoid the kappa paradoxes (high kappa with low agreement and vice versa
    under skewed prevalence).
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.size != b_arr.size:
        raise ValueError("ratings must have equal length")
    if a_arr.size == 0:
        return 1.0
    return float(np.mean(a_arr == b_arr))


def cohens_kappa(a: Sequence, b: Sequence) -> AgreementResult:
    """Cohen's kappa for two raters and a categorical scale."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.size != b_arr.size:
        raise ValueError("ratings must have equal length")
    n = a_arr.size
    if n == 0:
        raise ValueError("empty rating vectors")
    categories = sorted(set(a_arr.tolist()) | set(b_arr.tolist()))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    cm = np.zeros((k, k), dtype=float)
    for ai, bi in zip(a_arr, b_arr):
        cm[cat_to_idx[ai], cat_to_idx[bi]] += 1
    po = float(np.trace(cm) / n)
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))
    if pe >= 1:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1 - pe)
    return AgreementResult(
        coefficient=float(kappa),
        estimator="cohens_kappa",
        interpretation=_landis_koch(kappa),
        n_items=int(n),
        n_raters=2,
        extra={"percent_agreement": po, "expected_agreement": pe},
    )


def weighted_cohens_kappa(
    a: Sequence[int],
    b: Sequence[int],
    weighting: str = "quadratic",
) -> AgreementResult:
    """Weighted kappa for ordinal scales.

    For a 1..5 relevance scale, off-by-one disagreement should be penalized
    less than off-by-four. Quadratic weights are standard for ordinal Likert.
    """
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    if a_arr.size != b_arr.size:
        raise ValueError("ratings must have equal length")
    n = a_arr.size
    categories = sorted(set(a_arr.tolist()) | set(b_arr.tolist()))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    cm = np.zeros((k, k), dtype=float)
    for ai, bi in zip(a_arr, b_arr):
        cm[cat_to_idx[ai], cat_to_idx[bi]] += 1
    cm /= n
    if weighting == "linear":
        w = np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    elif weighting == "quadratic":
        w = np.array([[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)])
    else:
        raise ValueError(f"unknown weighting: {weighting}")
    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    expected = np.outer(row_marg, col_marg)
    numerator = float(np.sum(w * cm))
    denominator = float(np.sum(w * expected))
    if denominator == 0:
        kappa = 1.0 if numerator == 0 else 0.0
    else:
        kappa = 1 - numerator / denominator
    return AgreementResult(
        coefficient=float(kappa),
        estimator=f"cohens_kappa_{weighting}",
        interpretation=_landis_koch(kappa),
        n_items=int(n),
        n_raters=2,
        extra={"weighting": 1.0 if weighting == "quadratic" else 0.0},
    )


# ---------------------------------------------------------------------------
# Multi-rater
# ---------------------------------------------------------------------------


def fleiss_kappa(ratings: Sequence[Sequence[int]]) -> AgreementResult:
    """Fleiss's kappa for >=2 raters with fixed N per item.

    Input: `ratings[i][c]` = number of raters who assigned category c to item i.
    Each row must sum to the same total (N raters).
    """
    matrix = np.asarray(ratings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("expected 2D ratings matrix [n_items, n_categories]")
    n_items, n_cats = matrix.shape
    n_raters = int(matrix.sum(axis=1)[0])
    if not np.allclose(matrix.sum(axis=1), n_raters):
        raise ValueError("each item must have the same total rater count")
    if n_items == 0:
        raise ValueError("empty rating matrix")
    if n_raters < 2:
        raise ValueError("fleiss kappa needs >=2 raters")

    p_j = matrix.sum(axis=0) / (n_items * n_raters)
    p_i = (np.sum(matrix * matrix, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    pe = float(np.sum(p_j ** 2))
    if pe >= 1:
        kappa = 1.0 if p_bar == 1.0 else 0.0
    else:
        kappa = (p_bar - pe) / (1 - pe)
    return AgreementResult(
        coefficient=float(kappa),
        estimator="fleiss_kappa",
        interpretation=_landis_koch(kappa),
        n_items=int(n_items),
        n_raters=int(n_raters),
        extra={"p_bar": p_bar, "pe": pe},
    )


def krippendorff_alpha(
    data: Sequence[Sequence[Optional[float]]],
    level: str = "nominal",
) -> AgreementResult:
    """Krippendorff's alpha, handling missing data and mixed measurement levels.

    `data[r][i]` = rating from rater r for item i (None = missing).

    `level`:
      - "nominal"   distance = 0 if equal else 1
      - "ordinal"   distance based on rank pair differences (assume integer codes)
      - "interval"  distance = (x - y)^2
      - "ratio"     distance = ((x - y) / (x + y))^2

    Returns alpha in [-1, 1]; alpha >= 0.8 considered acceptable for
    high-stakes coding (Krippendorff 2004), >=0.667 tentative.
    """
    n_raters = len(data)
    if n_raters < 2:
        raise ValueError("krippendorff_alpha needs >=2 raters")
    n_items = len(data[0])
    if any(len(r) != n_items for r in data):
        raise ValueError("rater rows must be aligned")

    # Build value list per item.
    units: List[List[float]] = []
    for i in range(n_items):
        col = [data[r][i] for r in range(n_raters) if data[r][i] is not None]
        if len(col) >= 2:
            units.append([float(v) for v in col])
    if not units:
        raise ValueError("no item has >=2 non-missing ratings")

    def delta(x: float, y: float) -> float:
        if level == "nominal":
            return 0.0 if x == y else 1.0
        if level == "interval":
            return (x - y) ** 2
        if level == "ratio":
            if (x + y) == 0:
                return 0.0
            return ((x - y) / (x + y)) ** 2
        if level == "ordinal":
            # Pairwise rank-based.
            return (x - y) ** 2  # interval distance is a reasonable proxy
        raise ValueError(f"unknown level {level}")

    do_num = 0.0
    do_den = 0.0
    for col in units:
        m_u = len(col)
        for x in col:
            for y in col:
                do_num += delta(x, y)
        do_den += m_u - 1
    do = do_num / (do_den * len(units[0]) if False else max(1, sum(len(c) for c in units)))

    # de: expected disagreement across the value frequency distribution.
    all_vals: List[float] = [v for col in units for v in col]
    n = len(all_vals)
    de_num = 0.0
    for x in all_vals:
        for y in all_vals:
            de_num += delta(x, y)
    de = de_num / (n * (n - 1)) if n > 1 else 0.0
    if de == 0:
        alpha = 1.0 if do == 0 else 0.0
    else:
        alpha = 1 - do / de
    return AgreementResult(
        coefficient=float(alpha),
        estimator=f"krippendorff_alpha_{level}",
        interpretation=_landis_koch(alpha),
        n_items=int(n_items),
        n_raters=int(n_raters),
        extra={"observed_disagreement": float(do), "expected_disagreement": float(de)},
    )


def gwets_ac1(a: Sequence, b: Sequence) -> AgreementResult:
    """Gwet's AC1: alternative to Cohen's kappa that is robust to prevalence
    and bias paradoxes when one category dominates."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.size != b_arr.size:
        raise ValueError("ratings must have equal length")
    n = a_arr.size
    categories = sorted(set(a_arr.tolist()) | set(b_arr.tolist()))
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    cm = np.zeros((k, k), dtype=float)
    for ai, bi in zip(a_arr, b_arr):
        cm[cat_to_idx[ai], cat_to_idx[bi]] += 1
    cm /= n
    po = float(np.trace(cm))
    # Gwet pe = sum_q pi_q * (1 - pi_q) / (k - 1) where pi_q = (row+col)/2.
    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    pi = (row_marg + col_marg) / 2
    if k == 1:
        ac1 = 1.0
    else:
        pe = float(np.sum(pi * (1 - pi)) / (k - 1))
        ac1 = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return AgreementResult(
        coefficient=float(ac1),
        estimator="gwets_ac1",
        interpretation=_landis_koch(ac1),
        n_items=int(n),
        n_raters=2,
        extra={"percent_agreement": po},
    )


# ---------------------------------------------------------------------------
# Judge / human consensus convenience wrapper
# ---------------------------------------------------------------------------


def judge_agreement_panel(
    judges: Dict[str, Sequence[int]],
) -> Dict[str, AgreementResult]:
    """Compute pairwise Cohen's kappa for a panel of judges.

    Useful when you have an LLM-judge and several human references and want
    a triangulation matrix.
    """
    keys = list(judges.keys())
    out: Dict[str, AgreementResult] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            label = f"{keys[i]}__vs__{keys[j]}"
            out[label] = cohens_kappa(judges[keys[i]], judges[keys[j]])
    return out
