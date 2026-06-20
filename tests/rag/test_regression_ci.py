"""Tests for benchmark.rag.regression_ci."""
from __future__ import annotations

import json
from pathlib import Path


from benchmark.rag.regression_ci import (
    GoldenBaselineRow,
    GoldenItem,
    GoldenRunRow,
    detect_regressions,
    flake_recheck,
    load_baseline,
    load_golden_jsonl,
    run_against_golden,
    write_snapshot,
)


def test_load_golden_jsonl(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"item_id": "a", "query": "Q1", "expected": {}}) + "\n")
    items = load_golden_jsonl(p)
    assert len(items) == 1
    assert items[0].query == "Q1"


def test_run_against_golden_collects_metrics():
    items = [GoldenItem(item_id="a", query="q", expected={})]
    rows = run_against_golden(items, lambda it: {"x": 1.5})
    assert rows[0].metrics["x"] == 1.5


def test_write_snapshot_roundtrip(tmp_path: Path):
    rows = [GoldenRunRow(item_id="a", metrics={"x": 0.5})]
    p = write_snapshot(rows, tmp_path / "snap.json")
    assert p.exists()
    loaded = load_baseline(p)
    assert "a" in loaded
    assert loaded["a"].metrics["x"] == 0.5


def test_detect_regressions_flags_drops():
    current = [GoldenRunRow(item_id="a", metrics={"x": 0.4})]
    baseline = {"a": GoldenBaselineRow(item_id="a", metrics={"x": 0.7})}
    rpt = detect_regressions(current, baseline, tolerance=0.05)
    assert len(rpt.regressions) == 1
    assert rpt.regressions[0].metric == "x"


def test_detect_regressions_no_false_positive_below_tolerance():
    current = [GoldenRunRow(item_id="a", metrics={"x": 0.69})]
    baseline = {"a": GoldenBaselineRow(item_id="a", metrics={"x": 0.70})}
    rpt = detect_regressions(current, baseline, tolerance=0.02)
    assert rpt.regressions == []


def test_flake_recheck_filters_flakes():
    from benchmark.rag.regression_ci import RegressionRow

    flagged = [
        RegressionRow(
            item_id="flaky", metric="x", baseline=0.7, current=0.65, delta=-0.05, regressed=True
        ),
    ]

    counter = {"i": 0}

    def flaky_runner(item_id: str):
        counter["i"] += 1
        # Always return ok; not regressed any more.
        return {"x": 0.71}

    surviving = flake_recheck(flagged, flaky_runner, n_retries=3, tolerance=0.02)
    assert surviving == []


def test_flake_recheck_keeps_persistent_regressions():
    from benchmark.rag.regression_ci import RegressionRow

    flagged = [
        RegressionRow(
            item_id="bad", metric="x", baseline=0.7, current=0.5, delta=-0.2, regressed=True
        ),
    ]

    def stable_runner(item_id: str):
        return {"x": 0.5}

    surviving = flake_recheck(flagged, stable_runner, n_retries=3, tolerance=0.05)
    assert len(surviving) == 1


def test_detect_regressions_lower_is_better_polarity():
    # latency style metric: lower is better, so an increase is the regression.
    current = [GoldenRunRow(item_id="a", metrics={"lat": 200})]
    baseline = {"a": GoldenBaselineRow(item_id="a", metrics={"lat": 100})}
    rpt = detect_regressions(current, baseline, tolerance=10, direction="lower_is_better")
    assert len(rpt.regressions) == 1
