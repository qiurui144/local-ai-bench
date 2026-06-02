"""Tests for benchmark.rerank.

CPU-only, no endpoint:
- yes/no relevance parsing (English + Chinese, terse + verbose).
- native /v1/rerank response parsing + score routing (mocked httpx).
- run_rerank orchestrator via a monkeypatched scorer (deterministic scores),
  both generative-proxy and native-batch paths.
"""
from __future__ import annotations

import pytest

import common
from benchmark.embedding.datasets import RetrievalQuery
from benchmark.rerank import scorer


# --------------------------------------------------------------------------- #
# Relevance parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("yes", 1.0),
    ("Yes.", 1.0),
    ("no", 0.0),
    ("No, irrelevant", 0.0),
    ("相关", 1.0),
    ("不相关", 0.0),
    ("是", 1.0),
    ("否", 0.0),
    ("maybe", 0.5),
    ("", 0.5),
])
def test_parse_relevance(text, expected):
    assert scorer.parse_relevance(text) == expected


def test_rerank_prompt_contains_query_and_doc():
    p = scorer.rerank_prompt("如何重置密码", "在设置页修改密码")
    assert "如何重置密码" in p and "在设置页修改密码" in p
    assert "yes or no" in p.lower()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class _FakeModel:
    name = "fake-rerank"
    hf_repo = "fake/rerank"


def test_run_rerank_perfect(monkeypatch):
    """score_pair returns 1.0 for gold docs, 0.0 otherwise → perfect ranking."""
    from benchmark.rerank import accuracy

    def fake_score_pair(model_cfg, query, doc, **kw):
        # In the fixture, gold docs literally contain the marker "GOLD".
        return (1.0 if "GOLD" in doc else 0.0), True, 12.0

    monkeypatch.setattr(accuracy, "score_pair", fake_score_pair)

    queries = [
        RetrievalQuery(query="q1", candidates=["noise", "GOLD doc", "noise2"],
                       relevant={1}, qid="q1", source="builtin"),
        RetrievalQuery(query="q2", candidates=["GOLD a", "noise", "noise"],
                       relevant={0}, qid="q2", source="builtin"),
    ]
    res = accuracy.run_rerank(_FakeModel(), queries,
                             thresholds={"ndcg_at_10_min": 0.9, "mrr_min": 0.9})
    assert res["verdict"] == "PASS"
    assert res["aggregate"]["mrr"] == pytest.approx(1.0)
    assert res["aggregate"]["ndcg@10"] == pytest.approx(1.0)
    sep = res["aggregate"]["score_separation"]
    assert sep["pos_mean"] > sep["neg_mean"]
    assert res["aggregate"]["num_pairs"] == 6


def test_run_rerank_endpoint_down_fails(monkeypatch):
    from benchmark.rerank import accuracy

    def fake_score_pair(model_cfg, query, doc, **kw):
        return 0.0, False, 0.0   # all calls fail

    monkeypatch.setattr(accuracy, "score_pair", fake_score_pair)
    queries = [RetrievalQuery(query="q", candidates=["a", "b"], relevant={0}, source="builtin")]
    res = accuracy.run_rerank(_FakeModel(), queries)
    assert res["verdict"] == "FAIL"
    assert any("all rerank calls failed" in r for r in res["verdict_reasons"])


# --------------------------------------------------------------------------- #
# Native /v1/rerank path
# --------------------------------------------------------------------------- #
class _NativeModel:
    """A rerank_native model config stub (no real endpoint)."""
    name = "bge-reranker"
    hf_repo = "BAAI/bge-reranker-v2-m3"
    rerank_native = True

    @property
    def base_url(self):
        return "http://localhost:9202/v1"


class _FakeHTTPResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _patch_httpx_post(monkeypatch, response):
    """Make common.httpx.post return ``response`` (or raise if it's an Exception)."""
    def fake_post(url, **kw):
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(common.httpx, "post", fake_post)


def test_infer_rerank_aligns_scores_by_index(monkeypatch):
    """Backend returns results out of order → scores re-aligned to input order."""
    payload = {"results": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.1},
        {"index": 1, "relevance_score": 0.5},
    ]}
    _patch_httpx_post(monkeypatch, _FakeHTTPResponse(200, payload))
    res = common.infer_rerank(_NativeModel(), "q", ["d0", "d1", "d2"])
    assert res.ok
    assert res.scores == [0.1, 0.5, 0.9]


def test_infer_rerank_accepts_score_key_variants(monkeypatch):
    """Some backends use ``score`` instead of ``relevance_score``."""
    payload = {"results": [
        {"index": 0, "score": 0.7},
        {"index": 1, "score": 0.2},
    ]}
    _patch_httpx_post(monkeypatch, _FakeHTTPResponse(200, payload))
    res = common.infer_rerank(_NativeModel(), "q", ["a", "b"])
    assert res.ok and res.scores == [0.7, 0.2]


def test_infer_rerank_empty_docs_short_circuits(monkeypatch):
    # No HTTP call for an empty candidate list.
    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("should not POST for empty docs")
    monkeypatch.setattr(common.httpx, "post", boom)
    res = common.infer_rerank(_NativeModel(), "q", [])
    assert res.ok and res.scores == []


def test_infer_rerank_count_mismatch_fails(monkeypatch):
    """Fewer scores than docs → ok=False (a silently-truncated reranker)."""
    payload = {"results": [{"index": 0, "relevance_score": 0.5}]}
    _patch_httpx_post(monkeypatch, _FakeHTTPResponse(200, payload))
    res = common.infer_rerank(_NativeModel(), "q", ["a", "b"])
    assert not res.ok
    assert "!= 2 docs" in res.error


def test_infer_rerank_endpoint_404(monkeypatch):
    _patch_httpx_post(monkeypatch, _FakeHTTPResponse(404, {"error": "no rerank"}))
    res = common.infer_rerank(_NativeModel(), "q", ["a"])
    assert not res.ok and res.error.startswith("HTTP 404")


def test_infer_rerank_connection_error(monkeypatch):
    _patch_httpx_post(monkeypatch, ConnectionError("refused"))
    res = common.infer_rerank(_NativeModel(), "q", ["a"])
    assert not res.ok and "ConnectionError" in res.error


def test_score_pair_routes_to_native(monkeypatch):
    """rerank_native model → score_pair uses /v1/rerank, not the chat endpoint."""
    payload = {"results": [{"index": 0, "relevance_score": 0.84}]}
    _patch_httpx_post(monkeypatch, _FakeHTTPResponse(200, payload))

    def fail_chat(*a, **k):  # pragma: no cover - generative path must not run
        raise AssertionError("native model must not call infer_sync")
    monkeypatch.setattr(scorer, "infer_sync", fail_chat)

    s, ok, _ = scorer.score_pair(_NativeModel(), "q", "doc")
    assert ok and s == pytest.approx(0.84)


def test_score_query_native_batches_list(monkeypatch):
    payload = {"results": [
        {"index": 0, "relevance_score": 0.2},
        {"index": 1, "relevance_score": 0.95},
        {"index": 2, "relevance_score": 0.4},
    ]}
    _patch_httpx_post(monkeypatch, _FakeHTTPResponse(200, payload))
    scores, ok, lat = scorer.score_query_native(_NativeModel(), "q", ["a", "b", "c"])
    assert ok and scores == [0.2, 0.95, 0.4]
    assert lat >= 0.0


def test_run_rerank_native_path(monkeypatch):
    """Native batch path: one query request, score separation + scoring_path tag."""
    from benchmark.rerank import accuracy

    def fake_score_query_native(model_cfg, query, docs):
        # Gold docs (marker "GOLD") score high; one batch latency for the query.
        return [0.9 if "GOLD" in d else 0.1 for d in docs], True, 40.0

    monkeypatch.setattr(accuracy, "score_query_native", fake_score_query_native)

    queries = [
        RetrievalQuery(query="q1", candidates=["noise", "GOLD doc", "noise2"],
                       relevant={1}, qid="q1", source="builtin"),
        RetrievalQuery(query="q2", candidates=["GOLD a", "noise"],
                       relevant={0}, qid="q2", source="builtin"),
    ]
    res = accuracy.run_rerank(_NativeModel(), queries,
                              thresholds={"ndcg_at_10_min": 0.9, "mrr_min": 0.9})
    assert res["verdict"] == "PASS"
    agg = res["aggregate"]
    assert agg["scoring_path"] == "native_rerank"
    assert agg["mrr"] == pytest.approx(1.0)
    assert agg["score_separation"]["pos_mean"] > agg["score_separation"]["neg_mean"]
    # Whole-query rerank latency is recorded for the native path.
    assert agg["query_rerank_latency_ms_stats"]["count"] == 2
    # Per-pair latency = batch / n_docs (amortised), still populated.
    assert agg["single_pair_latency_ms_stats"]["count"] == 5


def test_run_rerank_native_endpoint_down_fails(monkeypatch):
    from benchmark.rerank import accuracy

    def fake_score_query_native(model_cfg, query, docs):
        return [0.0] * len(docs), False, 0.0

    monkeypatch.setattr(accuracy, "score_query_native", fake_score_query_native)
    queries = [RetrievalQuery(query="q", candidates=["a", "b"], relevant={0}, source="builtin")]
    res = accuracy.run_rerank(_NativeModel(), queries)
    assert res["verdict"] == "FAIL"
    assert any("all rerank calls failed" in r for r in res["verdict_reasons"])
