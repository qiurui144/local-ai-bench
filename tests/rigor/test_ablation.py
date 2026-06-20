"""Tests for benchmark.rigor.ablation."""
from __future__ import annotations

import pytest

from benchmark.rigor.ablation import (
    AblationConfig,
    AblationSummary,
    fractional_factorial,
    full_factorial,
    one_at_a_time,
    run_ablation,
)


def test_oat_yields_baseline_plus_variants():
    configs = one_at_a_time(
        baseline={"x": 1, "y": 2},
        variants={"x": [10], "y": [20, 30]},
    )
    names = [c.name for c in configs]
    assert "baseline" in names
    assert "x=10" in names
    assert "y=20" in names
    assert "y=30" in names
    assert len(configs) == 4


def test_full_factorial_size():
    cfgs = full_factorial({"a": [0, 1], "b": [10, 20], "c": [100, 200]})
    assert len(cfgs) == 8


def test_fractional_factorial_small_k():
    cfgs = fractional_factorial({"x": (0, 1), "y": (0, 1), "z": (0, 1)})
    assert len(cfgs) >= 4


def test_run_ablation_collects_metrics():
    cfgs = [
        AblationConfig(name="baseline", knobs={"k": 1}),
        AblationConfig(name="k=2", knobs={"k": 2}),
    ]
    summary = run_ablation(cfgs, lambda knobs: {"score": float(knobs["k"]) * 0.1})
    assert isinstance(summary, AblationSummary)
    assert len(summary.outcomes) == 2


def test_ablation_to_table_returns_deltas():
    cfgs = [
        AblationConfig(name="baseline", knobs={"k": 1}),
        AblationConfig(name="k=5", knobs={"k": 5}),
    ]
    summary = run_ablation(cfgs, lambda knobs: {"acc": float(knobs["k"])})
    table = summary.to_table("acc")
    # Each row is (name, value, delta_vs_baseline).
    names = [r[0] for r in table]
    assert "baseline" in names
    assert "k=5" in names
    for name, val, delta in table:
        if name == "k=5":
            assert delta == pytest.approx(4.0)


def test_ablation_top_k_ordering():
    cfgs = [AblationConfig(name=str(i), knobs={"x": i}) for i in range(5)]
    summary = run_ablation(cfgs, lambda knobs: {"acc": float(knobs["x"])})
    top = summary.top_k("acc", k=2, higher_is_better=True)
    assert [o.config.name for o in top] == ["4", "3"]


def test_empty_configs_raise():
    with pytest.raises(ValueError):
        run_ablation([], lambda knobs: {})
