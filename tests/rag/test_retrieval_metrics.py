"""Tests for benchmark.rag.retrieval_metrics."""
from __future__ import annotations

import math

import pytest

from benchmark.rag.retrieval_metrics import (
    RetrievalQueryResult,
    average_precision,
    bpref,
    bucketed_metrics,
    dcg_at_k,
    err_at_k,
    f1_at_k,
    mean_average_precision,
    mean_mrr,
    mean_ndcg_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    rank_biased_precision,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
)


def test_precision_at_k_basic():
    assert precision_at_k(["a", "b", "c"], ["a", "c"], k=3) == pytest.approx(2 / 3)


def test_recall_at_k_basic():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0


def test_f1_at_k_zero_when_no_overlap():
    assert f1_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_success_at_k_binary():
    assert success_at_k(["a", "b"], ["b"], k=2) == 1.0
    assert success_at_k(["a", "b"], ["c"], k=2) == 0.0


def test_mrr_known_position():
    rr = reciprocal_rank(["x", "y", "a", "z"], ["a"])
    assert rr == pytest.approx(1 / 3)


def test_mrr_no_relevant_zero():
    assert reciprocal_rank(["a", "b"], ["c"]) == 0.0


def test_average_precision_textbook():
    # ranked: [r, _, r, _, r] with 3 relevant docs
    ap = average_precision(["r1", "x", "r2", "y", "r3"], ["r1", "r2", "r3"])
    expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
    assert ap == pytest.approx(expected)


def test_r_precision_basic():
    # R = |relevant| = 2; top-2 = [a, b]; both relevant; precision = 1.0
    rp_perfect = r_precision(["a", "b", "c", "d"], ["a", "b"])
    assert rp_perfect == pytest.approx(1.0)
    # Mixed: top-2 = [a, c]; only one relevant; precision = 0.5
    rp_partial = r_precision(["a", "c", "b", "d"], ["a", "b"])
    assert rp_partial == pytest.approx(0.5)


def test_dcg_and_ndcg_perfect_ranking():
    ranked = ["a", "b", "c"]
    grades = {"a": 3.0, "b": 2.0, "c": 1.0}
    nd = ndcg_at_k(ranked, grades, 3)
    assert nd == pytest.approx(1.0)


def test_ndcg_drop_on_swap():
    grades = {"a": 3.0, "b": 0, "c": 1.0}
    perfect = ndcg_at_k(["a", "c", "b"], grades, 3)
    swapped = ndcg_at_k(["b", "c", "a"], grades, 3)
    assert swapped < perfect


def test_bpref_basic():
    ranked = ["r1", "n1", "r2", "n2"]
    rel = ["r1", "r2"]
    judged = ranked
    bp = bpref(ranked, rel, judged)
    # r1 at rank 1: no nonrel above; r2 at rank 3: 1 nonrel above.
    expected = (1.0 + (1 - 1/2)) / 2
    assert bp == pytest.approx(expected, rel=0.01)


def test_err_at_k_decay():
    grades = {"a": 4.0, "b": 0.0}
    err = err_at_k(["a", "b"], grades, 2, max_grade=4.0)
    assert 0.5 < err <= 1.0


def test_rank_biased_precision_in_unit_range():
    rel = ["a"]
    ranked = ["x", "a", "y"]
    low = rank_biased_precision(ranked, rel, persistence=0.5)
    high = rank_biased_precision(ranked, rel, persistence=0.9)
    # Both are positive and bounded by 1; the relationship between high and
    # low persistence depends on rank of relevant doc (deeper rank + high
    # persistence -> more credit; shallow + high persistence -> less due to
    # the (1-p) prefactor). We only assert range bounds here.
    assert 0 < low <= 1
    assert 0 < high <= 1


def test_mean_metrics_aggregate_over_queries():
    queries = [
        RetrievalQueryResult(
            query_id=f"q{i}",
            ranked_doc_ids=["a", "b"],
            relevant_doc_ids=["a"],
            relevance_grades={"a": 3.0, "b": 0.0},
        )
        for i in range(5)
    ]
    assert mean_mrr(queries) == pytest.approx(1.0)
    assert mean_average_precision(queries) == pytest.approx(1.0)
    assert mean_ndcg_at_k(queries, 2) == pytest.approx(1.0)


def test_bucketed_metrics_per_bucket():
    queries = [
        RetrievalQueryResult(
            query_id=f"q{i}",
            ranked_doc_ids=["a"],
            relevant_doc_ids=["a"],
            relevance_grades={"a": 3.0},
            bucket="easy" if i < 3 else "hard",
        )
        for i in range(5)
    ]
    buckets = bucketed_metrics(
        queries, lambda r: float(reciprocal_rank(r.ranked_doc_ids, r.relevant_doc_ids))
    )
    assert "easy" in buckets and "hard" in buckets
    assert buckets["easy"]["mean"] == pytest.approx(1.0)
