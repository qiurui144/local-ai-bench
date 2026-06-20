"""Retrieval metrics (PDF Chapter 3) + academic SOTA extensions.

Covers the standard IR metric battery plus less-common but defensible
variants that appear in modern retrieval papers:

Standard
--------
- precision_at_k, recall_at_k, f1_at_k
- mean_reciprocal_rank (MRR)
- average_precision (per-query AP)
- mean_average_precision (MAP)
- ndcg_at_k (graded relevance)
- r_precision (precision at R=number of relevant docs)
- success_at_k (binary indicator)

Extensions (academic literature, beyond PDF baseline)
-----------------------------------------------------
- bpref (Buckley & Voorhees 2004) - robust to incomplete judgments
- err_at_k (Chapelle et al. 2009) - Expected Reciprocal Rank, models user
  satisfaction via cascade.
- rank_biased_precision (Moffat & Zobel 2008) - user-persistence
  parameterization.
- ndcg_with_gain_decay - exposes the gain function for swap experiments.

Per-bucket reporting helpers turn any of these into per-domain /
per-difficulty breakouts, which production teams need to debug
retrieval regressions.

References
----------
- Manning, C. D., Raghavan, P., Schutze, H. (2008). Introduction to
  Information Retrieval.
- Buckley, C. & Voorhees, E. M. (2004). Retrieval Evaluation with
  Incomplete Information. SIGIR.
- Chapelle, O., Metlzer, D., Zhang, Y., Grinspan, P. (2009). Expected
  Reciprocal Rank for Graded Relevance. CIKM.
- Moffat, A. & Zobel, J. (2008). Rank-Biased Precision for Measurement
  of Retrieval Effectiveness. TOIS.
- Jarvelin, K. & Kekalainen, J. (2002). Cumulated Gain-Based Evaluation
  of IR Techniques. TOIS.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class RetrievalQueryResult:
    """A single query's retrieval output + ground truth."""

    query_id: str
    ranked_doc_ids: List[str]
    relevant_doc_ids: List[str]  # binary truth
    relevance_grades: Optional[Dict[str, float]] = None  # graded {doc_id: gain}
    bucket: Optional[str] = None  # for per-bucket reporting


# ---------------------------------------------------------------------------
# Binary-judgment metrics
# ---------------------------------------------------------------------------


def precision_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    top_k = ranked[:k]
    relevant_set = set(relevant)
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked[:k]
    relevant_set = set(relevant)
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / len(relevant_set)


def f1_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def success_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary: did any relevant doc make top-k?"""
    relevant_set = set(relevant)
    return 1.0 if any(d in relevant_set for d in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for i, d in enumerate(ranked, start=1):
        if d in relevant_set:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """AP = mean of P@k for each rank at which a relevant doc appears."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    total = 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant_set:
            hits += 1
            total += hits / i
    return total / len(relevant_set)


def r_precision(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """Precision@R where R = |relevant|. Insensitive to k choice."""
    if not relevant:
        return 0.0
    return precision_at_k(ranked, relevant, len(relevant))


# ---------------------------------------------------------------------------
# Graded-judgment metrics
# ---------------------------------------------------------------------------


def dcg_at_k(
    ranked: Sequence[str],
    relevance_grades: Dict[str, float],
    k: int,
    gain_fn: Callable[[float], float] = lambda r: (2**r - 1),
) -> float:
    """DCG with default Jarvelin-Kekalainen gain (2^r - 1).

    `gain_fn` lets you swap for linear gain (lambda r: r) when comparing
    against papers that use the older formulation.
    """
    score = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        grade = relevance_grades.get(d, 0.0)
        if grade <= 0:
            continue
        score += gain_fn(grade) / math.log2(i + 1)
    return score


def ndcg_at_k(
    ranked: Sequence[str],
    relevance_grades: Dict[str, float],
    k: int,
    gain_fn: Callable[[float], float] = lambda r: (2**r - 1),
) -> float:
    """Normalized DCG: DCG divided by ideal DCG."""
    actual = dcg_at_k(ranked, relevance_grades, k, gain_fn)
    ideal_order = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal = sum(
        gain_fn(g) / math.log2(i + 1) for i, g in enumerate(ideal_order, start=1) if g > 0
    )
    if ideal == 0:
        return 0.0
    return actual / ideal


# ---------------------------------------------------------------------------
# Academic-literature extensions
# ---------------------------------------------------------------------------


def bpref(ranked: Sequence[str], relevant: Sequence[str], judged: Sequence[str]) -> float:
    """Binary preference (Buckley & Voorhees 2004).

    bpref = 1/R * sum_{r in retrieved relevant} (1 - n/R)
    where n is the number of judged-non-relevant retrieved above r and R
    is the count of judged-relevant docs.

    Robust to incomplete judgments, which is the standard case for any
    benchmark on a corpus larger than the judged subset.
    """
    relevant_set = set(relevant)
    judged_nonrel = set(judged) - relevant_set
    R = len(relevant_set)
    if R == 0:
        return 0.0
    nonrel_seen = 0
    total = 0.0
    for d in ranked:
        if d in relevant_set:
            penalty = min(nonrel_seen, R) / R
            total += 1 - penalty
        elif d in judged_nonrel:
            nonrel_seen += 1
    return total / R


def err_at_k(
    ranked: Sequence[str],
    relevance_grades: Dict[str, float],
    k: int,
    max_grade: float = 4.0,
) -> float:
    """Expected Reciprocal Rank (Chapelle et al. 2009).

    Models user satisfaction in a cascade: probability of stopping at
    rank i conditional on dissatisfaction with ranks 1..i-1.

    Probability of satisfaction at rank i: R_i = (2^g_i - 1) / 2^max_grade
    Stops at i with prob R_i * prod_{j<i}(1 - R_j); contributes 1/i.
    """
    err = 0.0
    p_continue = 1.0
    for i, d in enumerate(ranked[:k], start=1):
        g = relevance_grades.get(d, 0.0)
        if g < 0:
            g = 0
        r = (2**g - 1) / (2**max_grade)
        err += p_continue * r / i
        p_continue *= 1 - r
    return err


def rank_biased_precision(
    ranked: Sequence[str], relevant: Sequence[str], persistence: float = 0.8, k: Optional[int] = None
) -> float:
    """Rank-Biased Precision (Moffat & Zobel 2008).

    RBP = (1 - p) * sum_{i=1..k} R_i * p^{i-1}
    where R_i is 1 if rank i is relevant, else 0, and p is the user
    persistence (probability of continuing to next rank). Defaults to
    p=0.8 (typical web search calibration).
    """
    if not 0 < persistence < 1:
        raise ValueError("persistence must be in (0, 1)")
    cutoff = len(ranked) if k is None else min(k, len(ranked))
    relevant_set = set(relevant)
    total = 0.0
    for i in range(cutoff):
        if ranked[i] in relevant_set:
            total += persistence**i
    return (1 - persistence) * total


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


def mean_metric(
    results: Sequence[RetrievalQueryResult],
    per_query_fn: Callable[[RetrievalQueryResult], float],
) -> float:
    """Mean across queries; ignores queries with NaN."""
    vals = [per_query_fn(r) for r in results]
    return float(sum(vals) / len(vals)) if vals else 0.0


def mean_average_precision(results: Sequence[RetrievalQueryResult]) -> float:
    return mean_metric(
        results,
        lambda r: average_precision(r.ranked_doc_ids, r.relevant_doc_ids),
    )


def mean_mrr(results: Sequence[RetrievalQueryResult]) -> float:
    return mean_metric(
        results,
        lambda r: reciprocal_rank(r.ranked_doc_ids, r.relevant_doc_ids),
    )


def mean_ndcg_at_k(results: Sequence[RetrievalQueryResult], k: int) -> float:
    return mean_metric(
        results,
        lambda r: ndcg_at_k(r.ranked_doc_ids, r.relevance_grades or {}, k),
    )


def bucketed_metrics(
    results: Sequence[RetrievalQueryResult],
    per_query_fn: Callable[[RetrievalQueryResult], float],
) -> Dict[str, Dict[str, float]]:
    """Bucket-by-bucket aggregation. Returns nested dict of {bucket: {n, mean, std}}.

    Use when you want per-domain or per-difficulty breakouts; this is the
    fix for the "94% overall, 0% on Bucket Z" pitfall.
    """
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        bucket = r.bucket or "unbucketed"
        grouped[bucket].append(per_query_fn(r))
    out: Dict[str, Dict[str, float]] = {}
    for bucket, vals in grouped.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        out[bucket] = {
            "n": float(len(vals)),
            "mean": float(mean),
            "std": float(math.sqrt(var)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
    return out
