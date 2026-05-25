"""Drift detection + auto-curation (PDF Chapter 12).

Three kinds of drift to monitor:

1. *Query distribution drift*: today's queries look unlike last week's.
   Detect via KS / PSI on query embedding centroid / length distribution.
2. *Embedding drift*: the embedding model output distribution itself
   shifted (silent dependency upgrade).
3. *Temporal performance drift*: per-week cohort metrics degrade even if
   the inputs look stationary.

We re-use ../rigor/ood_assessment.py for the distance computations and
wrap them in production-friendly entry points.

Plus auto-curation: identify low-confidence/edge production cases and
queue them for golden-set inclusion. Closes the loop between online
monitoring and offline benchmark.

References
----------
- Lipton, Z. C. et al. (2018). Detecting and Correcting for Label Shift.
- Yu, J. et al. (2020). MLPerf: An Industry Standard Benchmark Suite for
  Machine Learning Performance. (drift over time concept)
- Klaise, J. et al. (2021). Monitoring and explainability of models in
  production. (auto-curation patterns)
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..rigor.ood_assessment import domain_shift_score, psi, temporal_drift


@dataclass(frozen=True)
class DriftSignal:
    metric: str
    value: float
    threshold: float
    triggered: bool
    direction: str  # "increase" | "decrease" | "either"


@dataclass(frozen=True)
class DriftReport:
    timestamp: float
    n_reference: int
    n_observed: int
    signals: List[DriftSignal]
    recommendation: str  # "no_action" | "monitor" | "retrain"


# ---------------------------------------------------------------------------
# Query distribution drift
# ---------------------------------------------------------------------------


def query_length_drift(
    reference_queries: Sequence[str],
    observed_queries: Sequence[str],
    ks_p_threshold: float = 0.01,
) -> DriftSignal:
    ref_lens = np.asarray([len(q) for q in reference_queries], dtype=float)
    obs_lens = np.asarray([len(q) for q in observed_queries], dtype=float)
    rep = domain_shift_score(ref_lens, obs_lens, metric="ks")
    return DriftSignal(
        metric="query_length_ks",
        value=rep.p_value,
        threshold=ks_p_threshold,
        triggered=rep.p_value < ks_p_threshold,
        direction="decrease",
    )


def query_embedding_drift(
    reference_embeddings: np.ndarray,
    observed_embeddings: np.ndarray,
    centroid_shift_threshold: float = 0.10,
) -> DriftSignal:
    """Cosine distance between query embedding centroids.

    A sudden centroid shift means production is asking about a different
    topic distribution than the training/reference corpus.
    """
    if reference_embeddings.ndim != 2 or observed_embeddings.ndim != 2:
        raise ValueError("expected 2D embedding matrices")
    ref_centroid = reference_embeddings.mean(axis=0)
    obs_centroid = observed_embeddings.mean(axis=0)
    rn = np.linalg.norm(ref_centroid)
    on = np.linalg.norm(obs_centroid)
    if rn == 0 or on == 0:
        cos = 0.0
    else:
        cos = float(np.dot(ref_centroid, obs_centroid) / (rn * on))
    distance = 1 - cos
    return DriftSignal(
        metric="centroid_cosine_distance",
        value=float(distance),
        threshold=centroid_shift_threshold,
        triggered=distance > centroid_shift_threshold,
        direction="increase",
    )


def query_distribution_psi(
    reference_scores: Sequence[float],
    observed_scores: Sequence[float],
    psi_threshold: float = 0.25,
) -> DriftSignal:
    val = psi(reference_scores, observed_scores, n_bins=10)
    return DriftSignal(
        metric="psi",
        value=float(val),
        threshold=psi_threshold,
        triggered=val > psi_threshold,
        direction="increase",
    )


# ---------------------------------------------------------------------------
# Performance drift over time
# ---------------------------------------------------------------------------


def per_week_performance(
    timestamps: Sequence[float],
    values: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Bucket values by ISO week and return per-week summary statistics."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for ts, v in zip(timestamps, values):
        week_key = time.strftime("%Y-W%U", time.gmtime(ts))
        buckets[week_key].append(float(v))
    out: Dict[str, Dict[str, float]] = {}
    for week, vals in sorted(buckets.items()):
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[week] = {"n": float(len(vals)), "mean": mean, "std": std}
    return out


def temporal_performance_drift(
    timestamps: Sequence[float],
    metric_values: Sequence[float],
    window: int = 100,
    step: int = 50,
    drop_threshold: float = 0.05,
) -> List[DriftSignal]:
    """Sliding-window mean; alert when a window mean drops by more than
    `drop_threshold` vs the first window."""
    windows = temporal_drift(metric_values, timestamps, window=window, step=step)
    if not windows:
        return []
    baseline_mean = windows[0]["mean"]
    signals: List[DriftSignal] = []
    for w in windows[1:]:
        delta = baseline_mean - w["mean"]
        signals.append(
            DriftSignal(
                metric=f"window_mean_drop@{w['window_start_ts']:.0f}",
                value=float(delta),
                threshold=drop_threshold,
                triggered=delta > drop_threshold,
                direction="increase",
            )
        )
    return signals


# ---------------------------------------------------------------------------
# Top-level drift assessor
# ---------------------------------------------------------------------------


def assess_drift(
    reference: Dict[str, Any],
    observed: Dict[str, Any],
) -> DriftReport:
    """One-stop helper.

    `reference`, `observed` dicts may include:
      - "queries": List[str]
      - "embeddings": np.ndarray (n, d)
      - "scores": Sequence[float]
      - "timestamps" + "metric_values" for temporal mode
    Missing keys are skipped silently; we run whatever drift checks the
    data supports.
    """
    signals: List[DriftSignal] = []
    if "queries" in reference and "queries" in observed:
        signals.append(query_length_drift(reference["queries"], observed["queries"]))
    if "embeddings" in reference and "embeddings" in observed:
        signals.append(
            query_embedding_drift(reference["embeddings"], observed["embeddings"])
        )
    if "scores" in reference and "scores" in observed:
        signals.append(query_distribution_psi(reference["scores"], observed["scores"]))
    if "timestamps" in observed and "metric_values" in observed:
        signals.extend(
            temporal_performance_drift(
                observed["timestamps"], observed["metric_values"]
            )
        )
    n_triggered = sum(1 for s in signals if s.triggered)
    if n_triggered == 0:
        rec = "no_action"
    elif n_triggered <= 1:
        rec = "monitor"
    else:
        rec = "retrain"
    return DriftReport(
        timestamp=time.time(),
        n_reference=len(reference.get("queries", []) or reference.get("embeddings", [])),
        n_observed=len(observed.get("queries", []) or observed.get("embeddings", [])),
        signals=signals,
        recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Auto-curation: mine low-confidence production cases for the golden set
# ---------------------------------------------------------------------------


@dataclass
class CurationCandidate:
    item_id: str
    query: str
    confidence: float
    reason: str
    extra: Dict[str, Any] = field(default_factory=dict)


def mine_low_confidence_candidates(
    items: Sequence[Dict[str, Any]],
    confidence_field: str = "confidence",
    threshold: float = 0.5,
    limit: int = 50,
) -> List[CurationCandidate]:
    """Pick items whose confidence is below `threshold`, sorted by ascending
    confidence (lowest first)."""
    flagged = [
        i for i in items if isinstance(i.get(confidence_field), (int, float))
        and i[confidence_field] < threshold
    ]
    flagged.sort(key=lambda i: i[confidence_field])
    out: List[CurationCandidate] = []
    for it in flagged[:limit]:
        out.append(
            CurationCandidate(
                item_id=str(it.get("item_id", "")),
                query=str(it.get("query", "")),
                confidence=float(it[confidence_field]),
                reason="below_confidence_threshold",
                extra={k: v for k, v in it.items() if k not in {"item_id", "query", confidence_field}},
            )
        )
    return out


def mine_high_disagreement_candidates(
    items: Sequence[Dict[str, Any]],
    judge_outputs_field: str = "judge_outputs",
    limit: int = 50,
) -> List[CurationCandidate]:
    """Pick items where multiple judges disagree the most.

    `items[i][judge_outputs_field]` is expected to be a list of judge run
    dicts each with a `verdict` field. Disagreement is measured as 1 minus
    the modal-class fraction.
    """
    flagged: List[CurationCandidate] = []
    for it in items:
        outs = it.get(judge_outputs_field) or []
        verdicts = [o.get("verdict") for o in outs if o.get("verdict")]
        if len(verdicts) < 2:
            continue
        modal = max(verdicts, key=verdicts.count)
        agree = verdicts.count(modal) / len(verdicts)
        disagree = 1 - agree
        if disagree > 0:
            flagged.append(
                CurationCandidate(
                    item_id=str(it.get("item_id", "")),
                    query=str(it.get("query", "")),
                    confidence=float(agree),
                    reason=f"judge_disagreement={disagree:.2f}",
                    extra={"verdicts": verdicts},
                )
            )
    flagged.sort(key=lambda c: c.confidence)
    return flagged[:limit]
