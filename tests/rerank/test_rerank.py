"""Tests for benchmark.rerank.

CPU-only, no endpoint:
- yes/no relevance parsing (English + Chinese, terse + verbose).
- run_rerank orchestrator via a monkeypatched score_pair (deterministic scores).
"""
from __future__ import annotations

import pytest

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
