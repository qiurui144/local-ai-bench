"""Out-of-distribution and drift assessment helpers.

A model that scores well on a fixed golden set may collapse on the real
production query distribution. This module measures the gap.

Provided
--------
- domain_shift_score: KS/Wasserstein distance between train and prod feature/score distributions.
- temporal_drift: rolling cohort agreement; flags drift onset.
- ood_detector: simple density-based scorer using kNN distance in feature space.
- subgroup_audit: per-bucket performance with multi-test correction.

Why not just "test set accuracy":
- Production data shifts (new query intents, new document corpora).
- A bench that only measures one fixed set is brittle to drift.
- Reporting per-subgroup numbers prevents the famous "94% accurate but 0%
  on group X" pathology (Buolamwini & Gebru 2018 motivation).

References
----------
- Lipton, Z. C. et al. (2018). Detecting and Correcting for Label Shift
  with Black Box Predictors. ICML.
- Rabanser, S. et al. (2019). Failing Loudly: An Empirical Study of
  Methods for Detecting Dataset Shift. NeurIPS.
- Sun, Y. et al. (2022). Out-of-Distribution Detection with Deep Nearest
  Neighbors. ICML.
- Buolamwini, J. & Gebru, T. (2018). Gender Shades. FAT*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from .statistical_tests import ks_two_sample


@dataclass(frozen=True)
class DistShiftReport:
    statistic: float
    p_value: float
    distance: float
    metric: str
    n_reference: int
    n_observed: int
    interpretation: str


@dataclass(frozen=True)
class SubgroupResult:
    name: str
    metric_value: float
    n: int


@dataclass(frozen=True)
class SubgroupAudit:
    metric: str
    overall: float
    subgroups: List[SubgroupResult]
    worst_subgroup_gap: float
    flagged_subgroups: List[str]  # subgroups exceeding gap threshold


# ---------------------------------------------------------------------------
# Distribution shift
# ---------------------------------------------------------------------------


def domain_shift_score(
    reference: Sequence[float],
    observed: Sequence[float],
    metric: str = "ks",
) -> DistShiftReport:
    """Quantify shift between a reference (e.g. training) distribution and
    a new observed distribution (e.g. production queries this week).

    `metric`:
      - "ks":         Kolmogorov-Smirnov two-sample statistic
      - "wasserstein": 1D Earth Mover's distance (better at tail-shift)
      - "jensen_shannon": symmetric KL on histograms
    """
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if metric == "ks":
        res = ks_two_sample(ref, obs)
        interp = "no_shift" if res.p_value > 0.05 else "shift"
        return DistShiftReport(
            statistic=res.statistic,
            p_value=res.p_value,
            distance=res.statistic,
            metric="ks",
            n_reference=ref.size,
            n_observed=obs.size,
            interpretation=interp,
        )
    if metric == "wasserstein":
        dist = float(stats.wasserstein_distance(ref, obs))
        # Heuristic interpretation: > std(reference) is a meaningful tail shift.
        thr = float(ref.std()) if ref.size > 1 else 1.0
        interp = "no_shift" if dist < 0.1 * thr else ("shift" if dist > thr else "mild_shift")
        return DistShiftReport(
            statistic=dist,
            p_value=float("nan"),
            distance=dist,
            metric="wasserstein",
            n_reference=ref.size,
            n_observed=obs.size,
            interpretation=interp,
        )
    if metric == "jensen_shannon":
        # Build common bins (Freedman-Diaconis number of bins).
        all_vals = np.concatenate([ref, obs])
        n_bins = min(max(10, int(np.sqrt(all_vals.size))), 100)
        edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
        p_hist, _ = np.histogram(ref, bins=edges, density=True)
        q_hist, _ = np.histogram(obs, bins=edges, density=True)
        # Normalize to probabilities.
        p = p_hist / max(p_hist.sum(), 1e-12)
        q = q_hist / max(q_hist.sum(), 1e-12)
        m = 0.5 * (p + q)

        def _kl(p_: np.ndarray, q_: np.ndarray) -> float:
            mask = (p_ > 0) & (q_ > 0)
            return float(np.sum(p_[mask] * np.log(p_[mask] / q_[mask])))

        js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        interp = "no_shift" if js < 0.05 else ("shift" if js > 0.2 else "mild_shift")
        return DistShiftReport(
            statistic=float(js),
            p_value=float("nan"),
            distance=float(js),
            metric="jensen_shannon",
            n_reference=ref.size,
            n_observed=obs.size,
            interpretation=interp,
        )
    raise ValueError(f"unknown metric: {metric}")


def psi(
    reference: Sequence[float],
    observed: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Population Stability Index, standard credit-scoring drift metric.

    PSI = sum_b (p_obs_b - p_ref_b) * ln(p_obs_b / p_ref_b)

    Rules of thumb (industry, not academic):
       PSI < 0.10  : no significant change
       0.10 - 0.25 : moderate change, monitor
       PSI > 0.25  : major change, retrain
    """
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(observed, dtype=float)
    # Bin on reference quantiles so empty bins are rare.
    edges = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    p_ref, _ = np.histogram(ref, bins=edges)
    p_obs, _ = np.histogram(obs, bins=edges)
    p_ref = p_ref.astype(float) / max(p_ref.sum(), 1)
    p_obs = p_obs.astype(float) / max(p_obs.sum(), 1)
    # Apply Laplace smoothing to avoid log(0).
    p_ref = np.where(p_ref == 0, 1e-4, p_ref)
    p_obs = np.where(p_obs == 0, 1e-4, p_obs)
    return float(np.sum((p_obs - p_ref) * np.log(p_obs / p_ref)))


# ---------------------------------------------------------------------------
# Temporal drift
# ---------------------------------------------------------------------------


def temporal_drift(
    values: Sequence[float],
    timestamps: Sequence[float],
    window: int = 50,
    step: int = 25,
) -> List[Dict[str, float]]:
    """Sliding-window mean and KS drift against the first window.

    Returns a list of dicts describing each window:
       window_start_ts, window_end_ts, mean, std, ks_stat, ks_p
    """
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must align")
    arr = np.asarray(values, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    order = np.argsort(ts)
    arr = arr[order]
    ts = ts[order]
    n = arr.size
    if n < window * 2:
        raise ValueError("need at least 2 * window samples")
    baseline = arr[:window]
    out: List[Dict[str, float]] = []
    start = 0
    while start + window <= n:
        chunk = arr[start : start + window]
        ks_stat, ks_p = stats.ks_2samp(baseline, chunk)
        out.append(
            {
                "window_start_ts": float(ts[start]),
                "window_end_ts": float(ts[start + window - 1]),
                "mean": float(chunk.mean()),
                "std": float(chunk.std(ddof=1)) if chunk.size > 1 else 0.0,
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
            }
        )
        start += step
    return out


# ---------------------------------------------------------------------------
# kNN-based OOD detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OODResult:
    is_ood: bool
    score: float  # distance / negative log density
    threshold: float


class KNNOODDetector:
    """Simple OOD detector: distance to k-th nearest training neighbor.

    Fits a vector store of in-distribution embeddings. At inference, an
    embedding whose k-NN distance exceeds the calibration threshold is
    flagged OOD.

    This is the "Deep Nearest Neighbors" approach (Sun et al. 2022).
    Embeddings should already be L2-normalized for cosine-like behavior.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._train: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None

    def fit(
        self,
        training_embeddings: np.ndarray,
        calibration_quantile: float = 0.95,
    ) -> None:
        if training_embeddings.ndim != 2:
            raise ValueError("expected 2D embeddings matrix")
        if training_embeddings.shape[0] <= self.k:
            raise ValueError("need more training samples than k")
        self._train = training_embeddings
        # Compute kNN distance for each training point (leave-one-out)
        # to calibrate the OOD threshold.
        dists = self._knn_distance(training_embeddings, leave_one_out=True)
        self._threshold = float(np.quantile(dists, calibration_quantile))

    def score(self, query_embeddings: np.ndarray) -> List[OODResult]:
        if self._train is None or self._threshold is None:
            raise RuntimeError("call fit() first")
        scores = self._knn_distance(query_embeddings, leave_one_out=False)
        return [
            OODResult(is_ood=float(s) > self._threshold, score=float(s), threshold=self._threshold)
            for s in scores
        ]

    def _knn_distance(
        self, query: np.ndarray, leave_one_out: bool
    ) -> np.ndarray:
        # Pairwise cosine distance against training.
        assert self._train is not None
        a = query
        b = self._train
        a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
        cosine_sim = (a / a_norm) @ (b / b_norm).T
        dist = 1 - cosine_sim
        if leave_one_out and a.shape == b.shape:
            np.fill_diagonal(dist, np.inf)
        # k-th smallest distance per row.
        k = min(self.k, dist.shape[1] - 1) if leave_one_out else self.k
        partitioned = np.partition(dist, k, axis=1)[:, :k]
        return partitioned.mean(axis=1)


# ---------------------------------------------------------------------------
# Subgroup audit
# ---------------------------------------------------------------------------


def subgroup_audit(
    scores: Sequence[float],
    subgroup_labels: Sequence[str],
    metric_fn: Callable[[Sequence[float]], float] = np.mean,
    gap_threshold: float = 0.05,
) -> SubgroupAudit:
    """Compute per-subgroup metric and flag any subgroup whose gap from
    overall exceeds `gap_threshold`.

    This is the foundational fairness/robustness audit: a benchmark that
    only reports averages can hide catastrophic failures on rare groups.
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(subgroup_labels)
    if scores_arr.size != labels_arr.size:
        raise ValueError("scores and labels must align")
    overall = float(metric_fn(scores_arr))
    results: List[SubgroupResult] = []
    flagged: List[str] = []
    worst_gap = 0.0
    for name in sorted(set(labels_arr.tolist())):
        mask = labels_arr == name
        sub = scores_arr[mask]
        if sub.size == 0:
            continue
        val = float(metric_fn(sub))
        gap = abs(val - overall)
        if gap > worst_gap:
            worst_gap = gap
        if gap > gap_threshold:
            flagged.append(name)
        results.append(SubgroupResult(name=name, metric_value=val, n=int(sub.size)))
    return SubgroupAudit(
        metric=getattr(metric_fn, "__name__", "metric_fn"),
        overall=overall,
        subgroups=results,
        worst_subgroup_gap=worst_gap,
        flagged_subgroups=flagged,
    )
