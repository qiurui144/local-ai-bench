"""Throughput / percentile methodology (per 2026-06-10 intake perf findings).

1. ``common.percentile`` must linearly interpolate — with N=5 samples the old
   truncating index made p95 == max, overstating tail latency and TPS tails.
2. ``run_throughput`` / ``_concurrency_probe`` must divide token totals by the
   ACTUAL elapsed wall time, not the nominal ``duration_s``: the last request
   always overshoots the deadline, so nominal division overstates TPS.
"""
import asyncio
import types

import pytest

import common
from benchmark import performance as perf


# --------------------------------------------------------------------------- #
# percentile: linear interpolation
# --------------------------------------------------------------------------- #
def test_percentile_interpolates():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    # rank = (n-1) * p/100 = 4 * 0.95 = 3.8 → 4 + 0.8 * (5-4) = 4.8
    assert common.percentile(vals, 95) == pytest.approx(4.8)
    assert common.percentile(vals, 50) == pytest.approx(3.0)


def test_percentile_n5_p95_is_not_max():
    vals = [10.0, 20.0, 30.0, 40.0, 1000.0]
    p95 = common.percentile(vals, 95)
    assert p95 < 1000.0, "p95 of 5 samples must interpolate, not return max"
    # rank = (n-1) * 0.95 = 3.8 → 40 + 0.8 * (1000 - 40) = 808
    assert p95 == pytest.approx(808.0)


def test_percentile_edge_cases():
    assert common.percentile([], 95) == 0.0
    assert common.percentile([7.0], 95) == 7.0
    assert common.percentile([1.0, 2.0], 0) == 1.0
    assert common.percentile([1.0, 2.0], 100) == 2.0


# --------------------------------------------------------------------------- #
# Fake clock: each inference call advances time; deadline overshoot is real
# --------------------------------------------------------------------------- #
class _FakeTime:
    """time-module stand-in driven by the inference stubs."""
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


class _Model:
    name = "stub"
    hf_repo = "org/stub"
    is_vlm = False

    @property
    def base_url(self):
        return "http://localhost:9999/v1"


def test_run_throughput_uses_actual_elapsed(monkeypatch, tmp_path):
    """4 requests x 3s each overshoot a 10s window to t=12; 120 tokens → 10 TPS.

    Dividing by the nominal 10s would claim 12 TPS (a 20% overstatement).
    """
    clock = _FakeTime()
    monkeypatch.setattr(perf, "time", clock)

    def fake_infer(model_cfg, **kw):
        clock.t += 3.0
        return common.InferResult(model="stub", ok=True, output_tokens=30,
                                  input_tokens=10, latency_ms=3000.0,
                                  tokens_per_sec=10.0)

    monkeypatch.setattr(perf, "infer_sync", fake_infer)
    out = perf.run_throughput(_Model(), tmp_path, duration_s=10.0)

    assert out["requests"] == 4
    assert out["total_output_tokens"] == 120
    assert out["aggregate_tps"] == pytest.approx(120 / 12.0)


class _DummyClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_concurrency_probe(monkeypatch, clock, tokens_per_req, advance_s):
    """Stub the probe's collaborators; only perf's module refs are touched
    (a SimpleNamespace stand-in for asyncio, so the global module stays
    untouched even mid-test)."""
    monkeypatch.setattr(perf, "time", clock)
    monkeypatch.setattr(perf.httpx, "AsyncClient", lambda **kw: _DummyClient())

    async def fake_infer_async(client, model_cfg, **kw):
        clock.t += advance_s
        return common.InferResult(model="stub", ok=True,
                                  output_tokens=tokens_per_req,
                                  input_tokens=10,
                                  latency_ms=advance_s * 1000.0,
                                  tokens_per_sec=tokens_per_req / advance_s)

    monkeypatch.setattr(perf, "infer_async", fake_infer_async)

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(
        perf, "asyncio",
        types.SimpleNamespace(sleep=no_sleep, gather=asyncio.gather),
    )


def test_concurrency_probe_uses_actual_elapsed(monkeypatch, tmp_path):
    """3 requests x 4s overshoot a 10s window to t=12; 60 tokens → 5 TPS."""
    clock = _FakeTime()
    _patch_concurrency_probe(monkeypatch, clock, tokens_per_req=20, advance_s=4.0)

    out = asyncio.run(perf._concurrency_probe(_Model(), tmp_path, 1, 10.0))
    assert out["total_requests"] == 3
    assert out["elapsed_s"] == pytest.approx(12.0)
    assert out["aggregate_tps"] == pytest.approx(60 / 12.0)


def test_concurrency_probe_multiworker_tps_invariant(monkeypatch, tmp_path):
    """concurrency=2: whatever the interleaving, TPS == tokens / elapsed.

    Request counts under a shared fake clock depend on scheduler order, so
    assert the methodology invariant instead of exact counts.
    """
    clock = _FakeTime()
    _patch_concurrency_probe(monkeypatch, clock, tokens_per_req=20, advance_s=4.0)

    out = asyncio.run(perf._concurrency_probe(_Model(), tmp_path, 2, 10.0))
    assert out["success"] >= 2
    assert out["elapsed_s"] == pytest.approx(clock.t)
    assert out["aggregate_tps"] == pytest.approx(
        out["success"] * 20 / out["elapsed_s"]
    )
