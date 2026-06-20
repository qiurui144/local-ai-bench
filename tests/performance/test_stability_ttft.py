"""Stability drift / TTFT / concurrency-ladder coverage (behavior backfill).

Documents current ``benchmark.performance`` semantics:

1. ``run_stability`` windows are HARDCODED at 300 s: first window is
   ``ts_offset_s <= 300``, last window is ``ts_offset_s >= duration_s - 300``.
   Samples in between (and error samples) never influence the verdict.
2. The verdict is binary PASS/WARN on ``drift_ratio < 1.30`` — strictly less:
   1.29 → PASS, 1.30 and 1.31 → WARN. There is no FAIL tier.
3. ``first_p95 == 0`` (e.g. an all-error run) short-circuits ``drift_ratio``
   to 1.0, so a run with 100% errors still reports drift "PASS" (the error
   rate field is the only signal). Guard exists to avoid ZeroDivisionError.
4. ``run_ttft`` counts a sample as an error when ``not ok`` OR
   ``ttft_ms <= 0`` (a 200-OK response without first-token timing is an
   error, not a 0 ms TTFT).
5. ``run_concurrency`` is a thin ladder over ``_concurrency_probe``: one step
   per requested concurrency, in order, all sharing ``duration_s``.

Fake-clock idiom follows tests/performance/test_throughput_stats.py
(_FakeTime monkeypatched as ``perf.time``; stubs return common.InferResult).
No real sleeping: the 900 s stability scenario runs entirely on the fake
clock.
"""
import pytest

import common
from benchmark import performance as perf


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


# --------------------------------------------------------------------------- #
# run_stability: 5-min windowing + drift_ratio boundary + guards
# --------------------------------------------------------------------------- #
def _run_stability_scenario(monkeypatch, tmp_path, *, first_ms=1000.0,
                            mid_ms=5000.0, last_ms=1290.0, fail=None):
    """900 s fake-clock run, 5 s sampling: windows are [0, 300] and [600, 900].

    Latency is a step function of the clock: ``first_ms`` before 300 s, a
    ``mid_ms`` spike in the middle, ``last_ms`` from 590 s on (590 not 600 so
    that no mid-latency request *ends* inside the last window — ts_offset_s
    is recorded at request completion). ``fail(i)`` marks the i-th request
    as an error; errored requests still consume clock time.
    """
    clock = _FakeTime()
    monkeypatch.setattr(perf, "time", clock)
    calls = {"n": 0}

    def fake_infer(model_cfg, **kw):
        t = clock.t
        lat = first_ms if t < 300.0 else (mid_ms if t < 590.0 else last_ms)
        clock.t += lat / 1000.0
        i = calls["n"]
        calls["n"] += 1
        ok = not (fail(i) if fail else False)
        return common.InferResult(model="stub", ok=ok, latency_ms=lat,
                                  output_tokens=50)

    monkeypatch.setattr(perf, "infer_sync", fake_infer)
    return perf.run_stability(_Model(), tmp_path, duration_s=900.0,
                              sample_interval_s=5.0)


def test_stability_windowing_ignores_mid_run_spike(monkeypatch, tmp_path):
    """A 5x latency spike between the two 300 s windows must not leak into
    either p95: windows select on ts_offset_s, not on sample index."""
    out = _run_stability_scenario(monkeypatch, tmp_path, mid_ms=5000.0)

    assert out["benchmark"] == "stability"
    assert out["first_5min_p95_ms"] == pytest.approx(1000.0)
    assert out["last_5min_p95_ms"] == pytest.approx(1290.0)
    assert out["errors"] == 0
    assert out["total_samples"] == len(out["all_samples"])
    # the spike really happened (it sits between the windows)
    assert any(s["latency_ms"] == 5000.0 for s in out["all_samples"])


def test_stability_drift_1_29_passes(monkeypatch, tmp_path):
    out = _run_stability_scenario(monkeypatch, tmp_path, last_ms=1290.0)
    assert out["latency_drift_ratio"] == pytest.approx(1.29)
    assert out["drift_verdict"] == "PASS"


def test_stability_drift_1_31_warns(monkeypatch, tmp_path):
    out = _run_stability_scenario(monkeypatch, tmp_path, last_ms=1310.0)
    assert out["latency_drift_ratio"] == pytest.approx(1.31)
    assert out["drift_verdict"] == "WARN"


def test_stability_drift_boundary_1_30_is_warn(monkeypatch, tmp_path):
    """The PASS rule is strictly `< 1.30`: exactly 1.30 is already WARN."""
    out = _run_stability_scenario(monkeypatch, tmp_path, last_ms=1300.0)
    assert out["latency_drift_ratio"] == pytest.approx(1.30)
    assert out["drift_verdict"] == "WARN"


def test_stability_all_errors_no_crash_first_p95_guard(monkeypatch, tmp_path):
    """100% errors → both windows empty → first_p95 == 0 guard kicks in:
    drift_ratio forced to 1.0 and verdict reads "PASS" (current behavior —
    error_rate is the only failure signal)."""
    out = _run_stability_scenario(monkeypatch, tmp_path, fail=lambda i: True)

    assert out["total_samples"] > 0
    assert out["errors"] == out["total_samples"]
    assert out["error_rate"] == 1.0
    assert out["first_5min_p95_ms"] == 0
    assert out["last_5min_p95_ms"] == 0
    assert out["latency_drift_ratio"] == 1.0
    assert out["drift_verdict"] == "PASS"


def test_stability_errors_counted_and_excluded_from_windows(monkeypatch, tmp_path):
    """Alternate ok/error: errors only feed error_rate; window p95s are
    computed from ok samples alone, so they stay at the clean step values."""
    out = _run_stability_scenario(monkeypatch, tmp_path,
                                  fail=lambda i: i % 2 == 1)

    assert 0 < out["errors"] < out["total_samples"]
    assert out["errors"] == sum(1 for s in out["all_samples"] if not s["ok"])
    assert out["error_rate"] == pytest.approx(
        out["errors"] / out["total_samples"])
    assert out["first_5min_p95_ms"] == pytest.approx(1000.0)
    assert out["last_5min_p95_ms"] == pytest.approx(1290.0)
    assert out["drift_verdict"] == "PASS"


# --------------------------------------------------------------------------- #
# run_ttft: interpolated stats + error semantics
# --------------------------------------------------------------------------- #
def test_run_ttft_interpolated_stats(monkeypatch, tmp_path):
    ttfts = iter([100.0, 200.0, 300.0, 400.0, 500.0])

    def fake_stream(model_cfg, **kw):
        t = next(ttfts)
        return common.InferResult(model="stub", ok=True, ttft_ms=t,
                                  latency_ms=t + 1000.0, output_tokens=20)

    monkeypatch.setattr(perf, "infer_stream", fake_stream)
    out = perf.run_ttft(_Model(), tmp_path, samples=5)

    assert out["benchmark"] == "ttft"
    assert out["errors"] == 0
    assert out["error_rate"] == 0
    assert out["ttft_ms_stats"]["count"] == 5
    assert out["ttft_ms_stats"]["p50"] == pytest.approx(300.0)
    # rank = 4 * 0.95 = 3.8 → 400 + 0.8 * (500 - 400) = 480 (not max=500)
    assert out["ttft_ms_stats"]["p95"] == pytest.approx(480.0)
    assert out["total_latency_ms_stats"]["p50"] == pytest.approx(1300.0)
    assert out["total_latency_ms_stats"]["p95"] == pytest.approx(1480.0)


def test_run_ttft_error_counting(monkeypatch, tmp_path):
    """not-ok AND ok-but-ttft_ms==0 both count as errors; their latencies
    are excluded from both stat blocks."""
    canned = iter([
        common.InferResult(model="stub", ok=True, ttft_ms=100.0,
                           latency_ms=500.0),
        common.InferResult(model="stub", ok=False, error="boom"),
        common.InferResult(model="stub", ok=True, ttft_ms=0.0,
                           latency_ms=9999.0),  # no first-token timing
        common.InferResult(model="stub", ok=True, ttft_ms=300.0,
                           latency_ms=700.0),
        common.InferResult(model="stub", ok=True, ttft_ms=200.0,
                           latency_ms=600.0),
    ])
    monkeypatch.setattr(perf, "infer_stream", lambda *a, **kw: next(canned))
    out = perf.run_ttft(_Model(), tmp_path, samples=5)

    assert out["errors"] == 2
    assert out["error_rate"] == pytest.approx(0.4)
    assert out["ttft_ms_stats"]["count"] == 3
    assert out["ttft_ms_stats"]["p50"] == pytest.approx(200.0)
    assert out["total_latency_ms_stats"]["max"] == pytest.approx(700.0)


# --------------------------------------------------------------------------- #
# run_concurrency: ladder wrapper over _concurrency_probe
# --------------------------------------------------------------------------- #
def test_run_concurrency_ladder_steps(monkeypatch, tmp_path):
    seen = []

    async def fake_probe(model_cfg, fixtures_dir, concurrency, duration_s):
        seen.append((model_cfg.name, concurrency, duration_s))
        return {"concurrency": concurrency, "duration_s": duration_s,
                "marker": f"c{concurrency}"}

    monkeypatch.setattr(perf, "_concurrency_probe", fake_probe)
    out = perf.run_concurrency(_Model(), tmp_path, concurrencies=[2, 4, 8],
                               duration_s=7.0)

    assert out["benchmark"] == "concurrency"
    assert out["model"] == "stub"
    assert seen == [("stub", 2, 7.0), ("stub", 4, 7.0), ("stub", 8, 7.0)]
    assert [s["concurrency"] for s in out["steps"]] == [2, 4, 8]
    assert [s["marker"] for s in out["steps"]] == ["c2", "c4", "c8"]
