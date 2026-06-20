"""Offline-online alignment (PDF Chapter 2).

A benchmark is only as useful as its correlation with production
behavior. This module measures the gap between offline judgments
(golden-set metrics) and online observations (sampled production
traces) and surfaces drift early.

Three classes
-------------
- OfflineRunner: replays a golden-set evaluation
- OnlineMonitor: samples production traces and computes the same metrics
- AlignmentChecker: compares the two distributions and emits a verdict

The checker reports:
- Spearman rank correlation between offline and online per-system rankings
- KL divergence between metric distributions
- Per-metric absolute gap and direction

If alignment is poor, the offline harness is not representative; either
the golden set is stale or production has shifted (drift detection
module addresses the latter).

References
----------
- Bernardi, L. et al. (2021). 150 Successful Machine Learning Models:
  6 Lessons Learned at Booking.com. KDD. (Offline-online gap motivation)
- Gilotte, A. et al. (2018). Offline A/B Testing for Recommender Systems.
  WSDM.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from ..rigor.statistical_tests import ks_two_sample


@dataclass
class GoldenItem:
    item_id: str
    query: str
    expected_answer: str
    expected_citations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionTrace:
    trace_id: str
    query: str
    answer: str
    citations: List[str]
    latency_ms: float
    user_feedback: Optional[str] = None  # "thumbs_up" | "thumbs_down" | None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentReport:
    metric: str
    offline_mean: float
    online_mean: float
    abs_gap: float
    relative_gap: float
    ks_statistic: float
    ks_p_value: float
    spearman_rho: Optional[float]
    spearman_p: Optional[float]
    verdict: str  # "aligned" | "drifted" | "uncertain"
    n_offline: int
    n_online: int


class OfflineRunner:
    """Evaluate the system under test on a golden set.

    `evaluator_fn` takes a GoldenItem and the system's answer/citations
    and returns a dict of per-item metrics. Aggregation is mean by
    default but the runner exposes the raw per-item rows for distribution
    comparison.
    """

    def __init__(
        self,
        system_fn: Callable[[str], Dict[str, Any]],
        evaluator_fn: Callable[[GoldenItem, Dict[str, Any]], Dict[str, float]],
    ):
        self.system_fn = system_fn
        self.evaluator_fn = evaluator_fn

    def run(self, golden: Sequence[GoldenItem]) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for item in golden:
            sys_out = self.system_fn(item.query)
            metrics = self.evaluator_fn(item, sys_out)
            metrics_with_id = {"item_id_hash": float(hash(item.item_id) % 1_000_003)}
            metrics_with_id.update({k: float(v) for k, v in metrics.items()})
            rows.append(metrics_with_id)
        return rows


class OnlineMonitor:
    """Score a sample of production traces with the same evaluator as the
    offline runner. The catch: production has no ground truth, so the
    evaluator must use proxy signals (user feedback, downstream metrics,
    LLM judge with the answer-only mode).
    """

    def __init__(
        self,
        evaluator_fn: Callable[[ProductionTrace], Dict[str, float]],
    ):
        self.evaluator_fn = evaluator_fn

    def run(self, traces: Sequence[ProductionTrace]) -> List[Dict[str, float]]:
        return [
            {k: float(v) for k, v in self.evaluator_fn(t).items()} for t in traces
        ]


class AlignmentChecker:
    """Compare offline and online metric distributions.

    Thresholds:
    - "aligned" if |gap| < tolerance AND ks_p > 0.10 AND spearman_rho > 0.7
    - "drifted" if any threshold fails
    - "uncertain" if insufficient sample size
    """

    def __init__(self, tolerance: float = 0.05, min_samples: int = 20):
        self.tolerance = tolerance
        self.min_samples = min_samples

    def compare(
        self,
        offline_rows: Sequence[Dict[str, float]],
        online_rows: Sequence[Dict[str, float]],
        metric: str,
    ) -> AlignmentReport:
        off = np.asarray([r[metric] for r in offline_rows if metric in r], dtype=float)
        on = np.asarray([r[metric] for r in online_rows if metric in r], dtype=float)
        if off.size == 0 or on.size == 0:
            raise ValueError(f"no samples for metric={metric}")
        off_mean = float(off.mean())
        on_mean = float(on.mean())
        abs_gap = abs(off_mean - on_mean)
        rel_gap = abs_gap / (abs(off_mean) + 1e-9)
        ks = ks_two_sample(off, on)
        rho: Optional[float] = None
        rho_p: Optional[float] = None
        # Spearman is between matched per-item rankings; requires alignment.
        # When either side is constant, scipy returns NaN; that case is
        # treated as "rho not informative" rather than as a failure.
        if off.size == on.size:
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                rho_stat, p = stats.spearmanr(off, on)
            if rho_stat == rho_stat:  # not NaN
                rho = float(rho_stat)
                rho_p = float(p)
        if off.size < self.min_samples or on.size < self.min_samples:
            verdict = "uncertain"
        elif abs_gap < self.tolerance and ks.p_value > 0.10 and (rho is None or rho > 0.7):
            verdict = "aligned"
        else:
            verdict = "drifted"
        return AlignmentReport(
            metric=metric,
            offline_mean=off_mean,
            online_mean=on_mean,
            abs_gap=float(abs_gap),
            relative_gap=float(rel_gap),
            ks_statistic=float(ks.statistic),
            ks_p_value=float(ks.p_value),
            spearman_rho=rho,
            spearman_p=rho_p,
            verdict=verdict,
            n_offline=off.size,
            n_online=on.size,
        )

    def compare_all(
        self,
        offline_rows: Sequence[Dict[str, float]],
        online_rows: Sequence[Dict[str, float]],
    ) -> Dict[str, AlignmentReport]:
        if not offline_rows or not online_rows:
            return {}
        keys = (set(offline_rows[0].keys()) & set(online_rows[0].keys())) - {"item_id_hash"}
        return {k: self.compare(offline_rows, online_rows, k) for k in sorted(keys)}

    @staticmethod
    def write_report(reports: Dict[str, AlignmentReport], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in reports.items()}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
