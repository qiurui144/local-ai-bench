"""Tests for benchmark.embedding.

All tests run on CPU with no vLLM / GPU dependency:
- recall@k / MRR / nDCG@10 on hand-constructed rankings with known answers.
- cosine top-k ranking from raw vectors.
- numerical validation (NaN / Inf / zero-vector / dim drift → FAIL).
- dataset loaders (built-in synthetic fallback + custom JSONL).
- run_embedding orchestrator via an injected fake embedder (no endpoint).
"""
from __future__ import annotations

import json
import math

import pytest

from benchmark.embedding import metrics
from benchmark.embedding.datasets import (
    RetrievalQuery,
    load_builtin_retrieval,
    load_retrieval,
    load_retrieval_jsonl,
)


# --------------------------------------------------------------------------- #
# Ranking metrics — known inputs, known outputs
# --------------------------------------------------------------------------- #
def test_recall_at_k_single_relevant():
    ranked = [3, 1, 0, 2]
    assert metrics.recall_at_k(ranked, {3}, 1) == 1.0      # hit at rank 1
    assert metrics.recall_at_k(ranked, {1}, 1) == 0.0      # miss at rank 1
    assert metrics.recall_at_k(ranked, {1}, 2) == 1.0      # hit within top-2


def test_recall_at_k_multi_relevant():
    ranked = [0, 1, 2, 3]
    # 2 relevant docs, both in top-3 → recall 1.0; only one in top-1 → 0.5
    assert metrics.recall_at_k(ranked, {0, 2}, 3) == 1.0
    assert metrics.recall_at_k(ranked, {0, 2}, 1) == 0.5


def test_reciprocal_rank():
    assert metrics.reciprocal_rank([5, 2, 7], {2}) == pytest.approx(0.5)
    assert metrics.reciprocal_rank([5, 2, 7], {5}) == 1.0
    assert metrics.reciprocal_rank([5, 2, 7], {9}) == 0.0


def test_ndcg_perfect_and_zero():
    # Relevant doc first → perfect nDCG = 1.0
    assert metrics.ndcg_at_k([0, 1, 2], {0}, 10) == pytest.approx(1.0)
    # No relevant retrieved → 0.0
    assert metrics.ndcg_at_k([1, 2, 3], {9}, 10) == 0.0


def test_ndcg_known_value():
    # relevant={1}: hit at rank 2 → DCG = 1/log2(3); ideal = 1/log2(2)=1
    val = metrics.ndcg_at_k([0, 1, 2], {1}, 10)
    assert val == pytest.approx(1.0 / math.log2(3))


def test_cosine_topk_orders_by_similarity():
    query = [1.0, 0.0]
    docs = [[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]]   # idx1 identical, idx2 close, idx0 orthogonal
    ranked = metrics.cosine_topk(query, docs)
    assert ranked[0] == 1
    assert ranked[1] == 2
    assert ranked[2] == 0


def test_cosine_topk_zero_vector_ranks_last():
    query = [1.0, 0.0]
    docs = [[0.0, 0.0], [1.0, 0.0]]   # zero doc must not NaN; ranks below the real hit
    ranked = metrics.cosine_topk(query, docs)
    assert ranked[0] == 1


def test_aggregate_retrieval():
    rankings = [[0, 1, 2], [2, 0, 1]]
    relevants = [{0}, {0}]
    agg = metrics.aggregate_retrieval(rankings, relevants)
    assert agg["num_queries"] == 2
    assert agg["recall@1"] == pytest.approx(0.5)   # first query hits @1, second @2
    assert agg["hit@1"] == pytest.approx(0.5)
    assert agg["hit@5"] == pytest.approx(1.0)
    assert agg["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert 0.0 <= agg["ndcg@10"] <= 1.0


def test_multi_relevant_hit_at_1_does_not_cap_like_recall_at_1():
    rankings = [[0, 1, 2], [2, 1, 0]]
    relevants = [{0, 1}, {1, 2}]
    agg = metrics.aggregate_retrieval(rankings, relevants)
    assert agg["recall@1"] == pytest.approx(0.5)
    assert agg["hit@1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Numerical validation — the "fast but wrong" gate
# --------------------------------------------------------------------------- #
def test_validate_embeddings_ok():
    v = metrics.validate_embeddings([[1.0, 2.0], [3.0, 4.0]])
    assert v["ok"] is True
    assert v["dim"] == 2


def test_validate_embeddings_zero_vector_fails():
    v = metrics.validate_embeddings([[1.0, 2.0], [0.0, 0.0]])
    assert v["ok"] is False
    assert v["zero_vectors"] == 1


def test_validate_embeddings_nan_inf_fail():
    v = metrics.validate_embeddings([[float("nan"), 1.0], [float("inf"), 2.0]])
    assert v["ok"] is False
    assert v["nan_vectors"] == 1
    assert v["inf_vectors"] == 1


def test_validate_embeddings_dim_mismatch_fails():
    v = metrics.validate_embeddings([[1.0, 2.0], [1.0, 2.0, 3.0]])
    assert v["ok"] is False
    assert v["dim_mismatch"] == 1


def test_validate_embeddings_empty_batch_fails():
    assert metrics.validate_embeddings([])["ok"] is False


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def test_builtin_retrieval_fallback():
    qs = load_builtin_retrieval(num_samples=3)
    assert len(qs) == 3
    assert all(isinstance(q, RetrievalQuery) for q in qs)
    assert all(q.source == "builtin" for q in qs)
    assert all(q.candidates and q.relevant for q in qs)


def test_load_retrieval_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [
        {"query": "q1", "candidates": ["a", "b", "c"], "relevant": [0, 2], "qid": "x1"},
        {"query": "q2", "candidates": ["d", "e"], "relevant": []},  # skipped: no gold
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    qs = load_retrieval_jsonl(p)
    assert len(qs) == 1                       # malformed row dropped
    assert qs[0].source == "custom"
    assert qs[0].relevant == {0, 2}


def test_load_retrieval_missing_file_falls_back(tmp_path):
    qs = load_retrieval(tmp_path / "does_not_exist.jsonl")
    assert qs and qs[0].source == "builtin"


# --------------------------------------------------------------------------- #
# Orchestrator with an injected fake embedder (no endpoint)
# --------------------------------------------------------------------------- #
class _FakeModel:
    name = "fake-embed"
    hf_repo = "fake/embed"


def test_run_embedding_perfect_via_fake_embedder(monkeypatch):
    """A deterministic embedder that maps each unique text to a basis-ish vector.

    Query text equals candidate[relevant] verbatim in our fixture, so cosine
    ranks the gold first → recall@1 should be 1.0.
    """
    from benchmark.embedding import accuracy
    from common import EmbedResult

    vocab: dict[str, list[float]] = {}

    def _vec(text: str) -> list[float]:
        if text not in vocab:
            i = len(vocab)
            v = [0.0] * 16
            v[i % 16] = 1.0
            v[(i * 3 + 1) % 16] += 0.5
            vocab[text] = v
        return list(vocab[text])

    def fake_infer_embedding(model_cfg, inputs, **kw):
        if isinstance(inputs, str):
            inputs = [inputs]
        return EmbedResult(model="fake", ok=True, embeddings=[_vec(t) for t in inputs])

    monkeypatch.setattr(accuracy, "infer_embedding", fake_infer_embedding)

    # Build queries where the query string IS the gold candidate (exact-vector match).
    queries = [
        RetrievalQuery(query="alpha", candidates=["alpha", "beta", "gamma"],
                       relevant={0}, qid="q1", source="builtin"),
        RetrievalQuery(query="delta", candidates=["epsilon", "delta", "zeta"],
                       relevant={1}, qid="q2", source="builtin"),
    ]
    res = accuracy.run_embedding(_FakeModel(), queries,
                                thresholds={"hit_at_1_min": 0.9, "recall_at_10_min": 0.9,
                                            "mrr_min": 0.9, "ndcg_at_10_min": 0.9})
    assert res["verdict"] == "PASS"
    assert res["aggregate"]["recall@1"] == pytest.approx(1.0)
    assert res["aggregate"]["hit@1"] == pytest.approx(1.0)
    assert res["aggregate"]["validation"]["ok"] is True


def test_run_embedding_zero_vector_fails(monkeypatch):
    from benchmark.embedding import accuracy
    from common import EmbedResult

    def fake_infer_embedding(model_cfg, inputs, **kw):
        if isinstance(inputs, str):
            inputs = [inputs]
        # Every vector is zero → numerical validation must FAIL.
        return EmbedResult(model="fake", ok=True, embeddings=[[0.0, 0.0, 0.0] for _ in inputs])

    monkeypatch.setattr(accuracy, "infer_embedding", fake_infer_embedding)
    queries = [RetrievalQuery(query="a", candidates=["a", "b"], relevant={0}, source="builtin")]
    res = accuracy.run_embedding(_FakeModel(), queries)
    assert res["verdict"] == "FAIL"
    assert any("numerical validation" in r for r in res["verdict_reasons"])


def test_run_embedding_endpoint_down_fails(monkeypatch):
    from benchmark.embedding import accuracy
    from common import EmbedResult

    def fake_infer_embedding(model_cfg, inputs, **kw):
        return EmbedResult(model="fake", ok=False, error="ConnectError")

    monkeypatch.setattr(accuracy, "infer_embedding", fake_infer_embedding)
    queries = [RetrievalQuery(query="a", candidates=["a", "b"], relevant={0}, source="builtin")]
    res = accuracy.run_embedding(_FakeModel(), queries)
    assert res["verdict"] == "FAIL"


def test_run_embedding_exception_returns_blocked(monkeypatch):
    """If _embed_texts raises (e.g. network crash), run_embedding must return BLOCKED."""
    from benchmark.embedding import accuracy

    def fake_infer_embedding(model_cfg, inputs, **kw):
        raise ConnectionError("endpoint refused connection")

    monkeypatch.setattr(accuracy, "infer_embedding", fake_infer_embedding)
    queries = [RetrievalQuery(query="a", candidates=["a", "b"], relevant={0}, source="builtin")]
    res = accuracy.run_embedding(_FakeModel(), queries)
    assert res["verdict"] == "BLOCKED"
    assert "embedding endpoint failed" in res["reason"]
