"""Tests for benchmark.rigor.multi_seed_runner."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmark.rigor.multi_seed_runner import (
    SeedRun,
    aggregate,
    detect_rank_flips,
    pin_seeds,
    run_multi_seed,
    two_sigma_significant,
    write_manifest,
)


def test_pin_seeds_makes_random_deterministic():
    import random

    pin_seeds(123)
    a = [random.random() for _ in range(5)]
    pin_seeds(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_run_multi_seed_invokes_per_seed():
    seen = []

    def fn(seed):
        seen.append(seed)
        return {"x": float(seed)}

    runs = run_multi_seed(fn, seeds=(0, 1, 2))
    assert [r.seed for r in runs] == [0, 1, 2]
    assert seen == [0, 1, 2]
    assert [r.metrics["x"] for r in runs] == [0.0, 1.0, 2.0]


def test_aggregate_reports_mean_std():
    runs = [
        SeedRun(seed=0, metrics={"acc": 0.80}, duration_s=0),
        SeedRun(seed=1, metrics={"acc": 0.85}, duration_s=0),
        SeedRun(seed=2, metrics={"acc": 0.90}, duration_s=0),
    ]
    agg = aggregate(runs)
    assert "acc" in agg
    assert agg["acc"].mean == pytest.approx(0.85)
    assert agg["acc"].std > 0
    assert agg["acc"].ci95_lower < 0.85 < agg["acc"].ci95_upper


def test_two_sigma_significant_rejects_small_gap():
    a = SeedRun(seed=0, metrics={"x": 0}, duration_s=0)
    runs_a = [SeedRun(0, {"x": 0.50}, 0), SeedRun(1, {"x": 0.51}, 0), SeedRun(2, {"x": 0.52}, 0)]
    runs_b = [SeedRun(0, {"x": 0.51}, 0), SeedRun(1, {"x": 0.52}, 0), SeedRun(2, {"x": 0.53}, 0)]
    agg_a = aggregate(runs_a)["x"]
    agg_b = aggregate(runs_b)["x"]
    # gap is 0.01, std is ~0.01; not 2-sigma significant
    assert not two_sigma_significant(agg_a, agg_b)


def test_two_sigma_significant_accepts_large_gap():
    runs_a = [SeedRun(0, {"x": 0.50}, 0), SeedRun(1, {"x": 0.51}, 0), SeedRun(2, {"x": 0.52}, 0)]
    runs_b = [SeedRun(0, {"x": 0.80}, 0), SeedRun(1, {"x": 0.81}, 0), SeedRun(2, {"x": 0.82}, 0)]
    agg_a = aggregate(runs_a)["x"]
    agg_b = aggregate(runs_b)["x"]
    assert two_sigma_significant(agg_a, agg_b)


def test_detect_rank_flips_stable_when_consistent():
    runs_a = [SeedRun(0, {"x": 0.9}, 0), SeedRun(1, {"x": 0.9}, 0)]
    runs_b = [SeedRun(0, {"x": 0.5}, 0), SeedRun(1, {"x": 0.5}, 0)]
    report = detect_rank_flips({"A": runs_a, "B": runs_b}, metric="x")
    assert report.stable
    assert report.flips_observed == 0


def test_detect_rank_flips_flagged_on_inversion():
    runs_a = [SeedRun(0, {"x": 0.9}, 0), SeedRun(1, {"x": 0.3}, 0)]
    runs_b = [SeedRun(0, {"x": 0.5}, 0), SeedRun(1, {"x": 0.7}, 0)]
    report = detect_rank_flips({"A": runs_a, "B": runs_b}, metric="x")
    assert not report.stable
    assert report.flips_observed >= 1


def test_write_manifest_creates_json(tmp_path: Path):
    runs = [SeedRun(0, {"x": 1.0}, 0.1), SeedRun(1, {"x": 1.1}, 0.1)]
    agg = aggregate(runs)
    p = write_manifest(tmp_path, runs, agg, extra={"note": "test"})
    assert p.exists()
    text = p.read_text()
    assert "seeds" in text and "aggregates" in text


def test_detect_rank_flips_requires_aligned_seeds():
    runs_a = [SeedRun(0, {"x": 0.5}, 0)]
    runs_b = [SeedRun(1, {"x": 0.6}, 0)]
    with pytest.raises(ValueError):
        detect_rank_flips({"A": runs_a, "B": runs_b}, metric="x")
