"""Tests for benchmark.embedding.performance.

All tests run on CPU with no vLLM / endpoint dependency:
- run_embedding_latency via a stubbed infer_embedding with controlled
  latency_ms → p50/p95 stats correct, warmup excluded, samples honored.
- error paths: ok=False and non-positive latency both count as errors.
- measure_rss_dual: no pid / unreadable /proc → available=False (no crash);
  stubbed proc_rss_mb values → resident vs batch RSS distinction.
- run_embedding_performance: output dict shape snapshot.
"""
from __future__ import annotations

import pytest

from benchmark.embedding import performance
from benchmark.embedding.datasets import RetrievalQuery
from common import EmbedResult


class _FakeModel:
    name = "fake-embed"
    hf_repo = "fake/embed"


def _stub_infer(monkeypatch, latencies, ok=True):
    """Stub performance.infer_embedding returning queued latency_ms values.

    The queue covers warmup + sample calls in order; records every call's
    inputs so tests can assert call counts / shapes.
    """
    calls: list = []
    queue = list(latencies)

    def fake_infer_embedding(model_cfg, inputs, **kw):
        calls.append(inputs)
        lat = queue.pop(0) if queue else 1.0
        return EmbedResult(model="fake", ok=ok, embeddings=[[1.0, 0.0]],
                           latency_ms=lat)

    monkeypatch.setattr(performance, "infer_embedding", fake_infer_embedding)
    return calls


# --------------------------------------------------------------------------- #
# run_embedding_latency
# --------------------------------------------------------------------------- #
def test_latency_empty_texts_skipped():
    out = performance.run_embedding_latency(_FakeModel(), [])
    assert out == {"benchmark": "embedding_latency", "model": "fake-embed",
                   "skipped": True}


def test_latency_p50_from_controlled_latencies(monkeypatch):
    # 2 warmup calls (latency ignored) then 4 samples: 10/20/30/40ms.
    calls = _stub_infer(monkeypatch, [999.0, 999.0, 10.0, 20.0, 30.0, 40.0])
    out = performance.run_embedding_latency(
        _FakeModel(), ["a", "b"], samples=4, warmup=2)

    assert len(calls) == 6                      # warmup + samples, no more
    stats = out["single_query_latency_ms_stats"]
    assert stats["count"] == 4                  # warmup 999ms excluded
    assert stats["p50"] == pytest.approx(25.0)  # interpolated median
    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert out["errors"] == 0
    assert out["error_rate"] == 0
    assert out["path"] == "resident-model single-query"


def test_latency_samples_count_honored(monkeypatch):
    calls = _stub_infer(monkeypatch, [])
    out = performance.run_embedding_latency(
        _FakeModel(), ["only-one-text"], samples=7, warmup=3)
    assert len(calls) == 10
    assert out["samples"] == 7
    assert out["single_query_latency_ms_stats"]["count"] == 7
    # Single text is cycled for every call.
    assert all(c == "only-one-text" for c in calls)


def test_latency_all_errors(monkeypatch):
    _stub_infer(monkeypatch, [], ok=False)
    out = performance.run_embedding_latency(
        _FakeModel(), ["a"], samples=4, warmup=0)
    assert out["errors"] == 4
    assert out["error_rate"] == 1.0
    # No valid samples → zeroed stats, not a crash.
    assert out["single_query_latency_ms_stats"]["count"] == 0
    assert out["single_query_latency_ms_stats"]["p50"] == 0


def test_latency_nonpositive_latency_counts_as_error(monkeypatch):
    # ok=True but latency_ms == 0 → not a usable sample (counted as error).
    _stub_infer(monkeypatch, [0.0, 15.0])
    out = performance.run_embedding_latency(
        _FakeModel(), ["a"], samples=2, warmup=0)
    assert out["errors"] == 1
    assert out["error_rate"] == 0.5
    assert out["single_query_latency_ms_stats"]["count"] == 1
    assert out["single_query_latency_ms_stats"]["p50"] == 15.0


def test_latency_output_keys_stable(monkeypatch):
    _stub_infer(monkeypatch, [])
    out = performance.run_embedding_latency(_FakeModel(), ["a"], warmup=0)
    assert set(out.keys()) == {
        "benchmark", "model", "path", "samples",
        "single_query_latency_ms_stats", "errors", "error_rate",
    }
    assert out["benchmark"] == "embedding_latency"
    assert out["samples"] == 12                 # default


# --------------------------------------------------------------------------- #
# measure_rss_dual
# --------------------------------------------------------------------------- #
def test_rss_no_pid_unavailable(monkeypatch):
    calls = _stub_infer(monkeypatch, [])
    out = performance.measure_rss_dual(_FakeModel(), ["a"], server_pid=None)
    assert out["available"] is False
    assert "pid unknown" in out["reason"]
    assert calls == []                          # no inference without a pid


def test_rss_proc_unreadable_marks_unavailable(monkeypatch):
    # proc_rss_mb contract: unreadable /proc → 0.0; report must say
    # unavailable instead of fabricating numbers, and must not crash.
    _stub_infer(monkeypatch, [])
    monkeypatch.setattr(performance, "proc_rss_mb", lambda pid: 0.0)
    out = performance.measure_rss_dual(_FakeModel(), ["a"], server_pid=1234)
    assert out["available"] is False
    assert "/proc/1234/status" in out["reason"]


def test_rss_dual_distinction(monkeypatch):
    calls = _stub_infer(monkeypatch, [])
    rss_values = iter([512.0, 2048.0])          # resident first, then batch
    monkeypatch.setattr(performance, "proc_rss_mb",
                        lambda pid: next(rss_values))
    texts = [f"t{i}" for i in range(10)]
    out = performance.measure_rss_dual(_FakeModel(), texts,
                                       server_pid=42, batch_size=4)

    assert out["available"] is True
    assert out["server_pid"] == 42
    assert out["resident_query_rss_mb"] == 512.0
    assert out["batch_rss_mb"] == 2048.0
    assert "note" in out
    # Call 1: single short query; call 2: batch truncated to batch_size.
    assert calls[0] == "t0"
    assert calls[1] == texts[:4]


def test_rss_empty_texts_uses_fallback_fill(monkeypatch):
    calls = _stub_infer(monkeypatch, [])
    monkeypatch.setattr(performance, "proc_rss_mb", lambda pid: 100.0)
    out = performance.measure_rss_dual(_FakeModel(), [],
                                       server_pid=7, batch_size=8)
    assert out["available"] is True
    assert calls[0] == "查询"                   # single-query fallback
    assert calls[1] == ["填充文本"] * 8          # batch fill honours batch_size


def test_rss_partial_zero_still_available(monkeypatch):
    # Current behavior: only both-zero is unavailable; one nonzero reading
    # is reported as-is.
    _stub_infer(monkeypatch, [])
    rss_values = iter([512.0, 0.0])
    monkeypatch.setattr(performance, "proc_rss_mb",
                        lambda pid: next(rss_values))
    out = performance.measure_rss_dual(_FakeModel(), ["a"], server_pid=9)
    assert out["available"] is True
    assert out["resident_query_rss_mb"] == 512.0
    assert out["batch_rss_mb"] == 0.0


# --------------------------------------------------------------------------- #
# run_embedding_performance — top-level orchestrator shape
# --------------------------------------------------------------------------- #
def test_performance_output_shape(monkeypatch):
    _stub_infer(monkeypatch, [])
    monkeypatch.setattr(performance, "proc_rss_mb", lambda pid: 300.0)
    queries = [RetrievalQuery(query="q1", candidates=["a"], relevant={0},
                              source="builtin")]
    out = performance.run_embedding_performance(
        _FakeModel(), queries, samples=2, server_pid=5)

    assert set(out.keys()) == {"benchmark", "model", "latency", "memory"}
    assert out["benchmark"] == "embedding_performance"
    assert out["model"] == "fake-embed"
    assert out["latency"]["benchmark"] == "embedding_latency"
    assert out["latency"]["samples"] == 2
    assert out["memory"]["available"] is True


def test_performance_empty_queries_falls_back(monkeypatch):
    calls = _stub_infer(monkeypatch, [])
    out = performance.run_embedding_performance(_FakeModel(), [], samples=1)
    # Fallback text "查询" keeps latency measurable; memory unavailable (no pid).
    assert all(c == "查询" for c in calls)
    assert out["latency"]["samples"] == 1
    assert out["memory"]["available"] is False
