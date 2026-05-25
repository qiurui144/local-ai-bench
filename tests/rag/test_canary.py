"""Tests for benchmark.rag.canary."""
from __future__ import annotations

import pytest

from benchmark.rag.canary import (
    CanaryGate,
    CanaryThreshold,
    RollbackPolicy,
    ShadowRunner,
    TrafficSplitter,
)


def test_traffic_splitter_invalid_fraction():
    with pytest.raises(ValueError):
        TrafficSplitter(fraction_to_candidate=1.5)


def test_traffic_splitter_deterministic_per_request():
    sp = TrafficSplitter(0.5, salt="t")
    arm1 = sp.route("req-42")
    arm2 = sp.route("req-42")
    assert arm1 == arm2


def test_traffic_splitter_distribution_approximate():
    sp = TrafficSplitter(0.10, salt="x")
    candidate_count = sum(1 for i in range(5000) if sp.route(f"r{i}") == "candidate")
    # 5000 * 0.10 = 500; allow generous tolerance
    assert 350 < candidate_count < 700


def test_canary_gate_hold_when_underpopulated():
    g = CanaryGate(
        thresholds=[CanaryThreshold(metric="x", min_value=0.5)],
        window_size=100,
        min_samples=50,
    )
    g.record("x", 0.9)
    s = g.evaluate()
    assert s.decision == "hold"


def test_canary_gate_promote_when_thresholds_met():
    g = CanaryGate(
        thresholds=[CanaryThreshold(metric="x", min_value=0.5)],
        window_size=200,
        min_samples=10,
    )
    for _ in range(60):
        g.record("x", 0.9)
    s = g.evaluate()
    assert s.decision == "promote"


def test_canary_gate_rollback_on_violation():
    g = CanaryGate(
        thresholds=[CanaryThreshold(metric="x", min_value=0.5)],
        window_size=200,
        min_samples=10,
    )
    for _ in range(60):
        g.record("x", 0.1)
    s = g.evaluate()
    assert s.decision == "rollback"
    assert s.violations


def test_rollback_policy_requires_consecutive_breaches():
    policy = RollbackPolicy(consecutive_breaches=3, cooldown_seconds=1)
    from benchmark.rag.canary import CanaryStatus

    rollback_status = CanaryStatus(decision="rollback", n_samples=100, metric_summary={}, violations=["x<0.5"])
    actions = [policy.step(rollback_status)["action"] for _ in range(2)]
    assert actions[-1] == "noop"
    # Third in a row triggers.
    final = policy.step(rollback_status)["action"]
    assert final == "rollback_now"


def test_shadow_runner_returns_user_control_response():
    sr = ShadowRunner(
        control_fn=lambda r: {"text": "control"},
        candidate_fn=lambda r: {"text": "candidate"},
        comparator_fn=lambda c, k: {"len_delta": float(len(k["text"]) - len(c["text"]))},
    )
    out = sr.serve({"request_id": "r1", "query": "q"})
    assert out["user_response"]["text"] == "control"
    assert out["shadow"].metric_deltas["len_delta"] >= 0


def test_shadow_runner_handles_candidate_failure():
    def boom(r):
        raise RuntimeError("candidate broke")

    sr = ShadowRunner(
        control_fn=lambda r: {"ok": True},
        candidate_fn=boom,
        comparator_fn=lambda c, k: {},
    )
    out = sr.serve({"request_id": "r1"})
    assert out["user_response"]["ok"] is True
    assert "_error" in out["shadow"].candidate_output
