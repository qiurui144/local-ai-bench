"""Tests for benchmark.rag.offline_online_alignment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.rag.offline_online_alignment import (
    AlignmentChecker,
    GoldenItem,
    OfflineRunner,
    OnlineMonitor,
    ProductionTrace,
)


def test_offline_runner_invokes_system_and_evaluator():
    def sys_fn(query):
        return {"answer": "ok"}

    def eval_fn(item, out):
        return {"score": 1.0}

    runner = OfflineRunner(sys_fn, eval_fn)
    rows = runner.run([GoldenItem("a", "q", "ok", [])])
    assert rows[0]["score"] == 1.0


def test_online_monitor_scores_traces():
    def eval_fn(trace):
        return {"latency": trace.latency_ms}

    monitor = OnlineMonitor(eval_fn)
    traces = [ProductionTrace("t1", "q", "a", [], latency_ms=120.0)]
    rows = monitor.run(traces)
    assert rows[0]["latency"] == 120.0


def test_alignment_checker_detects_drift():
    offline = [{"x": 0.9} for _ in range(50)]
    online = [{"x": 0.4} for _ in range(50)]
    checker = AlignmentChecker(tolerance=0.05, min_samples=20)
    rpt = checker.compare(offline, online, "x")
    assert rpt.verdict == "drifted"


def test_alignment_checker_aligned():
    offline = [{"x": 0.9} for _ in range(50)]
    online = [{"x": 0.9} for _ in range(50)]
    checker = AlignmentChecker(tolerance=0.05, min_samples=20)
    rpt = checker.compare(offline, online, "x")
    assert rpt.verdict == "aligned"


def test_alignment_checker_uncertain_when_underpowered():
    offline = [{"x": 0.9}]
    online = [{"x": 0.9}]
    checker = AlignmentChecker(tolerance=0.05, min_samples=20)
    rpt = checker.compare(offline, online, "x")
    assert rpt.verdict == "uncertain"


def test_alignment_compare_all_returns_per_metric():
    offline = [{"x": 0.9, "y": 0.5} for _ in range(30)]
    online = [{"x": 0.85, "y": 0.5} for _ in range(30)]
    checker = AlignmentChecker(min_samples=20)
    out = checker.compare_all(offline, online)
    assert set(out.keys()) >= {"x", "y"}


def test_alignment_write_report(tmp_path: Path):
    offline = [{"x": 0.9} for _ in range(30)]
    online = [{"x": 0.5} for _ in range(30)]
    checker = AlignmentChecker(min_samples=20)
    out = checker.compare_all(offline, online)
    p = AlignmentChecker.write_report(out, tmp_path / "r.json")
    assert p.exists()
    assert "x" in json.loads(p.read_text())


def test_alignment_raises_when_metric_missing():
    offline = [{"x": 0.9}]
    online = [{"y": 0.9}]
    checker = AlignmentChecker(min_samples=1)
    with pytest.raises(ValueError):
        checker.compare(offline, online, "nope")
