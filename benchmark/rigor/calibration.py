"""Probability calibration metrics.

A classifier or judge that outputs probabilities should be *calibrated*:
when it says 70%, it should be right ~70% of the time. Calibration is
independent of accuracy: a 70% accurate model can be perfectly
calibrated (consistently reports its confidence) or wildly overconfident.

Metrics provided
----------------
- ECE        Expected Calibration Error (bin-based estimator)
- MCE        Maximum Calibration Error (worst-bin deviation)
- Brier      Squared error between predicted prob and 0/1 outcome
- Brier skill score vs reference (e.g. base rate)
- Reliability curve (predicted vs observed) for diagram rendering
- ACE        Adaptive Calibration Error (equal-mass bins instead of equal-width)
- Platt and isotonic re-calibration helpers

When to reach for which
-----------------------
- Use ECE for headline reporting on dashboards (familiar, intuitive).
- Use Brier when comparing models that span different accuracy levels
  (decomposes into reliability + resolution + uncertainty).
- Use ACE when class probabilities cluster near 0/1 (binned ECE underestimates
  miscalibration in sparse bins).
- Re-calibrate via Platt for binary; isotonic for multi-class one-vs-rest.

References
----------
- Naeini, M. P., Cooper, G. F., Hauskrecht, M. (2015). Obtaining Well
  Calibrated Probabilities Using Bayesian Binning. AAAI.
- Brier, G. W. (1950). Verification of Forecasts Expressed in Terms of
  Probability.
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Nixon, J. et al. (2019). Measuring Calibration in Deep Learning. CVPR
  Workshops. (adaptive binning)
- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CalibrationCurvePoint:
    bin_low: float
    bin_high: float
    mean_predicted: float
    fraction_positive: float
    n_samples: int


@dataclass(frozen=True)
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    n_bins: int
    n_samples: int
    curve: List[CalibrationCurvePoint]
    binning: str  # "equal_width" | "equal_mass"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _validate(probs: np.ndarray, labels: np.ndarray) -> None:
    if probs.shape != labels.shape:
        raise ValueError(
            f"probs and labels must have the same shape; got {probs.shape} vs {labels.shape}"
        )
    if probs.size == 0:
        raise ValueError("calibration metrics require non-empty inputs")
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    uniq = np.unique(labels)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError("labels must be binary 0/1")


def _equal_width_bins(probs: np.ndarray, n_bins: int) -> np.ndarray:
    """Return bin index per sample; bins are [0, 1/n), [1/n, 2/n), ..., [1-1/n, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `digitize` returns 1..n_bins+1; subtract 1 and clip top edge.
    idx = np.digitize(probs, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    return idx


def _equal_mass_bins(probs: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based binning so each bin holds ~n/k samples."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(probs, quantiles)
    # Ensure monotone (degenerates near constant predictions).
    edges[0] = 0.0
    edges[-1] = 1.0
    idx = np.digitize(probs, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    return idx


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
    binning: str = "equal_width",
) -> CalibrationReport:
    """Bin-based ECE estimator.

    ECE = sum_b |B_b|/N * |acc(B_b) - conf(B_b)|

    Where acc is the fraction of positives in the bin and conf is the
    mean predicted probability of the bin.
    """
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    _validate(probs_arr, labels_arr)

    if binning == "equal_width":
        bin_idx = _equal_width_bins(probs_arr, n_bins)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif binning == "equal_mass":
        bin_idx = _equal_mass_bins(probs_arr, n_bins)
        edges = np.quantile(probs_arr, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        raise ValueError(f"unknown binning: {binning}")

    n = probs_arr.size
    ece = 0.0
    mce = 0.0
    curve: List[CalibrationCurvePoint] = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            curve.append(
                CalibrationCurvePoint(
                    bin_low=float(edges[b]),
                    bin_high=float(edges[b + 1]),
                    mean_predicted=float("nan"),
                    fraction_positive=float("nan"),
                    n_samples=0,
                )
            )
            continue
        conf = float(probs_arr[mask].mean())
        acc = float(labels_arr[mask].mean())
        gap = abs(conf - acc)
        ece += n_b / n * gap
        mce = max(mce, gap)
        curve.append(
            CalibrationCurvePoint(
                bin_low=float(edges[b]),
                bin_high=float(edges[b + 1]),
                mean_predicted=conf,
                fraction_positive=acc,
                n_samples=n_b,
            )
        )
    brier = float(np.mean((probs_arr - labels_arr) ** 2))
    return CalibrationReport(
        ece=float(ece),
        mce=float(mce),
        brier=brier,
        n_bins=n_bins,
        n_samples=n,
        curve=curve,
        binning=binning,
    )


def adaptive_calibration_error(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> CalibrationReport:
    """ACE alias for ECE with equal-mass binning."""
    return expected_calibration_error(probs, labels, n_bins=n_bins, binning="equal_mass")


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Brier score = mean (p - y)^2."""
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    _validate(probs_arr, labels_arr)
    return float(np.mean((probs_arr - labels_arr) ** 2))


def brier_skill_score(
    probs: Sequence[float],
    labels: Sequence[int],
    reference_probs: Optional[Sequence[float]] = None,
) -> float:
    """Brier Skill Score relative to a reference.

    BSS = 1 - BS_model / BS_reference

    Default reference is the base-rate predictor (same probability for
    everyone equal to the observed positive rate).
    """
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    _validate(probs_arr, labels_arr)
    if reference_probs is None:
        base = float(labels_arr.mean())
        ref = np.full_like(probs_arr, fill_value=base)
    else:
        ref = np.asarray(reference_probs, dtype=float)
        if ref.shape != probs_arr.shape:
            raise ValueError("reference shape mismatch")
    bs_model = float(np.mean((probs_arr - labels_arr) ** 2))
    bs_ref = float(np.mean((ref - labels_arr) ** 2))
    if bs_ref == 0:
        return 0.0 if bs_model == 0 else float("-inf")
    return 1 - bs_model / bs_ref


# ---------------------------------------------------------------------------
# Reliability curve (for plotting); returns x/y arrays for matplotlib if any.
# ---------------------------------------------------------------------------


def reliability_curve(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
    binning: str = "equal_width",
) -> Tuple[List[float], List[float], List[int]]:
    """Return mean_predicted, fraction_positive, n_per_bin lists for plotting.

    Empty bins are skipped from the returned arrays.
    """
    rpt = expected_calibration_error(probs, labels, n_bins=n_bins, binning=binning)
    xs, ys, ns = [], [], []
    for pt in rpt.curve:
        if pt.n_samples == 0:
            continue
        xs.append(pt.mean_predicted)
        ys.append(pt.fraction_positive)
        ns.append(pt.n_samples)
    return xs, ys, ns


# ---------------------------------------------------------------------------
# Re-calibration helpers
# ---------------------------------------------------------------------------


def platt_recalibrate(
    probs_train: Sequence[float],
    labels_train: Sequence[int],
) -> "PlattScaler":
    """Fit Platt scaling on a held-out calibration set."""
    return PlattScaler.fit(probs_train, labels_train)


class PlattScaler:
    """Platt scaling: logistic regression on a single feature (the prob).

    p_cal = 1 / (1 + exp(A * p + B))
    Standard remedy for binary classifier overconfidence.
    """

    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b

    @classmethod
    def fit(cls, probs: Sequence[float], labels: Sequence[int]) -> "PlattScaler":
        from scipy.optimize import minimize

        p = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
        y = np.asarray(labels, dtype=float)
        _validate(p, y.astype(int))

        def nll(params: np.ndarray) -> float:
            a, b = float(params[0]), float(params[1])
            z = a * p + b
            # Numerically stable log-sigmoid (Mächler 2012). The probability
            # of the positive class under Platt's formulation is
            # sigmoid(-z) = 1 / (1 + exp(z)). We compute log_pos = log(sigmoid(-z))
            # and log_neg = log(1 - sigmoid(-z)) = log(sigmoid(z)) without
            # overflowing on either side of zero.
            log_pos = -np.logaddexp(0.0, z)        # log(1/(1+exp(z)))
            log_neg = -np.logaddexp(0.0, -z)       # log(1/(1+exp(-z)))
            return -float(np.sum(y * log_pos + (1 - y) * log_neg))

        res = minimize(nll, x0=np.array([-1.0, 0.0]), method="Nelder-Mead")
        a, b = float(res.x[0]), float(res.x[1])
        return cls(a=a, b=b)

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
        return 1.0 / (1.0 + np.exp(self.a * p + self.b))


def isotonic_recalibrate(
    probs_train: Sequence[float],
    labels_train: Sequence[int],
) -> "IsotonicScaler":
    """Pool-adjacent-violators isotonic regression for monotone re-calibration."""
    return IsotonicScaler.fit(probs_train, labels_train)


class IsotonicScaler:
    """Isotonic regression-based calibrator using pool-adjacent-violators.

    More flexible than Platt; can model arbitrary monotone miscalibration
    shapes, at the cost of more parameters (one knot per training sample).
    """

    def __init__(self, x_knots: np.ndarray, y_knots: np.ndarray) -> None:
        self.x = x_knots
        self.y = y_knots

    @classmethod
    def fit(cls, probs: Sequence[float], labels: Sequence[int]) -> "IsotonicScaler":
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=float)
        _validate(p, y.astype(int))
        order = np.argsort(p)
        x_sorted = p[order]
        y_sorted = y[order]
        # PAVA: repeatedly merge violating adjacent blocks.
        weights = np.ones_like(y_sorted)
        values = y_sorted.copy()
        i = 0
        while i < len(values) - 1:
            if values[i] > values[i + 1]:
                new_w = weights[i] + weights[i + 1]
                new_v = (weights[i] * values[i] + weights[i + 1] * values[i + 1]) / new_w
                values = np.concatenate([values[:i], [new_v], values[i + 2 :]])
                weights = np.concatenate([weights[:i], [new_w], weights[i + 2 :]])
                # x_sorted is collapsed by averaging the merged x range too,
                # but for piecewise-constant lookup we keep the left endpoint.
                x_sorted = np.concatenate(
                    [x_sorted[:i], [x_sorted[i]], x_sorted[i + 2 :]]
                )
                if i > 0:
                    i -= 1
            else:
                i += 1
        return cls(x_knots=x_sorted, y_knots=values)

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        out = np.interp(p, self.x, self.y, left=self.y[0], right=self.y[-1])
        return out
