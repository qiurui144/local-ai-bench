"""Reranker assessment + fusion (PDF Chapter 4).

A reranker takes the top-N retrieval candidates and reorders them by a
finer-grained scoring model. Two questions matter:

1. Does the reranker actually move relevant docs up? (win-rate, NDCG lift)
2. Does it fit within the latency budget? (P50/P95)

Beyond the PDF, this module includes classic IR fusion methods
(Borda, RRF, CombSUM, CombMNZ) because in production we often blend
multiple retrievers rather than rely on a single reranker.

References
----------
- Cormack, G. V., Clarke, C. L. A., Buettcher, S. (2009). Reciprocal
  Rank Fusion outperforms Condorcet and individual Rank Learning
  Methods. SIGIR.
- Fox, E. A. & Shaw, J. A. (1994). Combination of Multiple Searches.
  TREC. (CombSUM/CombMNZ)
- van Erp, M. & Schomaker, L. (2000). Variants of the Borda count
  method for combining ranked classifier hypotheses.
- Nogueira, R. & Cho, K. (2019). Passage Re-ranking with BERT. (modern
  cross-encoder rerankers)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .retrieval_metrics import (
    RetrievalQueryResult,
    average_precision,
    ndcg_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# Win-rate / lift evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankWinRateReport:
    n_queries: int
    wins: int
    ties: int
    losses: int
    win_rate: float  # wins / total
    mean_metric_delta: float
    p95_latency_overhead_ms: float


def rerank_win_rate(
    baseline_results: Sequence[RetrievalQueryResult],
    reranked_results: Sequence[RetrievalQueryResult],
    metric: str = "ndcg",
    k: int = 10,
    latency_overhead_ms: Optional[Sequence[float]] = None,
    tie_eps: float = 1e-6,
) -> RerankWinRateReport:
    """Compare two sets of ranked lists on a chosen metric.

    `baseline_results[i]` and `reranked_results[i]` must align by
    `query_id`. We compute the per-query metric for each, count wins
    (reranked > baseline by more than tie_eps), and report aggregate
    win rate + mean delta.

    `latency_overhead_ms` should be a per-query list of measured rerank
    latency overhead (model forward cost). We report P95 alongside the
    quality lift so callers can do quality-vs-latency tradeoff.
    """
    if len(baseline_results) != len(reranked_results):
        raise ValueError("baseline/reranked must align")
    base_map = {r.query_id: r for r in baseline_results}
    rerank_map = {r.query_id: r for r in reranked_results}
    if set(base_map) != set(rerank_map):
        raise ValueError("query_id sets diverge")
    wins = ties = losses = 0
    deltas: List[float] = []
    for qid in base_map:
        a = _per_query_metric(base_map[qid], metric, k)
        b = _per_query_metric(rerank_map[qid], metric, k)
        delta = b - a
        deltas.append(delta)
        if delta > tie_eps:
            wins += 1
        elif delta < -tie_eps:
            losses += 1
        else:
            ties += 1
    n = len(base_map)
    mean_delta = sum(deltas) / n if n else 0.0
    p95_lat = 0.0
    if latency_overhead_ms:
        sorted_lat = sorted(latency_overhead_ms)
        idx = max(0, int(0.95 * len(sorted_lat)) - 1)
        p95_lat = float(sorted_lat[idx])
    return RerankWinRateReport(
        n_queries=n,
        wins=wins,
        ties=ties,
        losses=losses,
        win_rate=float(wins / n) if n else 0.0,
        mean_metric_delta=float(mean_delta),
        p95_latency_overhead_ms=p95_lat,
    )


def _per_query_metric(r: RetrievalQueryResult, metric: str, k: int) -> float:
    if metric == "ndcg":
        return ndcg_at_k(r.ranked_doc_ids, r.relevance_grades or {}, k)
    if metric == "ap":
        return average_precision(r.ranked_doc_ids, r.relevant_doc_ids)
    if metric == "mrr":
        return reciprocal_rank(r.ranked_doc_ids, r.relevant_doc_ids)
    raise ValueError(f"unknown metric {metric}")


# ---------------------------------------------------------------------------
# Rank fusion methods
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    rrf_k: int = 60,
) -> List[str]:
    """RRF (Cormack et al. 2009): doc score = sum_r 1 / (rrf_k + rank_r(d)).

    `rrf_k=60` is the canonical default from the original paper, found
    empirically to dampen the influence of top ranks without erasing them.
    """
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (rrf_k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def borda_count(rankings: Sequence[Sequence[str]]) -> List[str]:
    """Borda: each ranker gives N-rank+1 points; sum across rankers."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        n = len(ranking)
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += n - rank + 1
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def comb_sum(
    rankings_with_scores: Sequence[Sequence[Tuple[str, float]]],
    normalize: bool = True,
) -> List[str]:
    """CombSUM (Fox & Shaw 1994): sum normalized scores across rankers."""
    accum: Dict[str, float] = defaultdict(float)
    for ranking in rankings_with_scores:
        if not ranking:
            continue
        scores = [s for _, s in ranking]
        if normalize:
            mn, mx = min(scores), max(scores)
            rng = mx - mn or 1.0
            for doc, s in ranking:
                accum[doc] += (s - mn) / rng
        else:
            for doc, s in ranking:
                accum[doc] += s
    return [d for d, _ in sorted(accum.items(), key=lambda x: x[1], reverse=True)]


def comb_mnz(
    rankings_with_scores: Sequence[Sequence[Tuple[str, float]]],
    normalize: bool = True,
) -> List[str]:
    """CombMNZ (Fox & Shaw 1994): sum * count_of_rankers_seen.

    Rewards docs that appear in many rankings; reduces influence of
    single-ranker outliers.
    """
    accum: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for ranking in rankings_with_scores:
        seen_here = set()
        if not ranking:
            continue
        scores = [s for _, s in ranking]
        if normalize:
            mn, mx = min(scores), max(scores)
            rng = mx - mn or 1.0
            for doc, s in ranking:
                accum[doc] += (s - mn) / rng
                seen_here.add(doc)
        else:
            for doc, s in ranking:
                accum[doc] += s
                seen_here.add(doc)
        for doc in seen_here:
            counts[doc] += 1
    final = {doc: accum[doc] * counts[doc] for doc in accum}
    return [d for d, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)]


# ---------------------------------------------------------------------------
# Latency budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyBudgetReport:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    n_samples: int
    fits_budget: bool
    budget_ms: float


def latency_budget(
    latencies_ms: Sequence[float],
    p50_budget: float = 50,
    p95_budget: float = 200,
) -> LatencyBudgetReport:
    """Standard rerank latency budgets per the PDF: P50<50ms, P95<200ms."""
    if not latencies_ms:
        return LatencyBudgetReport(0, 0, 0, 0, 0, True, p95_budget)
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    p50 = sorted_lat[max(0, int(0.50 * n) - 1)]
    p95 = sorted_lat[max(0, int(0.95 * n) - 1)]
    p99 = sorted_lat[max(0, int(0.99 * n) - 1)]
    max_lat = sorted_lat[-1]
    fits = float(p50) <= p50_budget and float(p95) <= p95_budget
    return LatencyBudgetReport(
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        max_ms=float(max_lat),
        n_samples=n,
        fits_budget=fits,
        budget_ms=p95_budget,
    )
