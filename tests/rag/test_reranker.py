"""Tests for benchmark.rag.reranker."""
from __future__ import annotations

import pytest

from benchmark.rag.reranker import (
    borda_count,
    comb_mnz,
    comb_sum,
    latency_budget,
    reciprocal_rank_fusion,
    rerank_win_rate,
)
from benchmark.rag.retrieval_metrics import RetrievalQueryResult


def test_rrf_orders_consistent_docs_first():
    rankings = [["a", "b", "c"], ["a", "c", "b"]]
    result = reciprocal_rank_fusion(rankings)
    assert result[0] == "a"


def test_borda_count_ranks_by_total_points():
    rankings = [["a", "b"], ["b", "a"]]
    result = borda_count(rankings)
    # Both have same total Borda points; order is implementation-defined,
    # but both should be present.
    assert set(result) == {"a", "b"}


def test_comb_sum_normalized():
    rankings = [[("a", 1.0), ("b", 0.0)], [("a", 0.9), ("b", 0.1)]]
    result = comb_sum(rankings, normalize=True)
    assert result[0] == "a"


def test_comb_mnz_rewards_doc_in_more_rankers():
    rankings = [[("a", 0.9), ("b", 0.8)], [("a", 0.9)]]
    result = comb_mnz(rankings, normalize=False)
    assert result[0] == "a"


def test_latency_budget_fits():
    lat = [50, 60, 70, 80]
    rep = latency_budget(lat, p50_budget=100, p95_budget=200)
    assert rep.fits_budget
    assert rep.n_samples == 4


def test_latency_budget_exceeds():
    lat = [500, 600, 700, 800]
    rep = latency_budget(lat, p50_budget=100, p95_budget=200)
    assert not rep.fits_budget


def _q(qid: str, ranked, rel, grades):
    return RetrievalQueryResult(
        query_id=qid,
        ranked_doc_ids=ranked,
        relevant_doc_ids=rel,
        relevance_grades=grades,
    )


def test_rerank_win_rate_obvious_winner():
    base = [_q("q1", ["x", "a"], ["a"], {"a": 3.0, "x": 0})]
    rer = [_q("q1", ["a", "x"], ["a"], {"a": 3.0, "x": 0})]
    rpt = rerank_win_rate(base, rer, metric="ndcg", k=2)
    assert rpt.wins == 1
    assert rpt.win_rate == 1.0


def test_rerank_win_rate_aligned_query_ids():
    base = [_q("q1", ["a"], ["a"], {"a": 3.0})]
    rer = [_q("q2", ["a"], ["a"], {"a": 3.0})]
    with pytest.raises(ValueError):
        rerank_win_rate(base, rer, metric="ndcg", k=1)


def test_rerank_win_rate_uses_latency_overhead():
    base = [_q("q1", ["a"], ["a"], {"a": 3.0})]
    rer = [_q("q1", ["a"], ["a"], {"a": 3.0})]
    rpt = rerank_win_rate(base, rer, metric="ndcg", k=1, latency_overhead_ms=[100, 200, 300])
    assert rpt.p95_latency_overhead_ms > 0


def test_rerank_invalid_metric():
    base = [_q("q1", ["a"], ["a"], {"a": 3.0})]
    rer = [_q("q1", ["a"], ["a"], {"a": 3.0})]
    with pytest.raises(ValueError):
        rerank_win_rate(base, rer, metric="not_a_metric", k=1)
