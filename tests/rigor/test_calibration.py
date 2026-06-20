"""Tests for benchmark.rigor.calibration."""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.rigor.calibration import (
    adaptive_calibration_error,
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    isotonic_recalibrate,
    platt_recalibrate,
    reliability_curve,
)


def test_brier_perfect_predictions():
    p = [1.0, 1.0, 0.0, 0.0]
    y = [1, 1, 0, 0]
    assert brier_score(p, y) == 0.0


def test_brier_worst_predictions():
    p = [0.0, 1.0]
    y = [1, 0]
    assert brier_score(p, y) == pytest.approx(1.0)


def test_ece_perfectly_calibrated():
    rng = np.random.default_rng(0)
    # Generate well-calibrated synthetic data.
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    rpt = expected_calibration_error(p, y, n_bins=10)
    assert rpt.ece < 0.05


def test_ece_overconfident_bias():
    # Probabilities at 0.9 but only 0.5 actually positive.
    p = [0.9] * 100
    y = [1] * 50 + [0] * 50
    rpt = expected_calibration_error(p, y, n_bins=5)
    assert rpt.ece > 0.3


def test_adaptive_calibration_error_matches_ece_alias():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < p).astype(int)
    rpt = adaptive_calibration_error(p, y, n_bins=5)
    assert rpt.binning == "equal_mass"


def test_brier_skill_score_zero_when_same_as_reference():
    p = [0.5] * 100
    y = [1] * 50 + [0] * 50
    bss = brier_skill_score(p, y)
    assert bss == pytest.approx(0.0, abs=1e-6)


def test_brier_skill_score_positive_when_better():
    p = [0.9] * 50 + [0.1] * 50
    y = [1] * 50 + [0] * 50
    bss = brier_skill_score(p, y)
    assert bss > 0.5


def test_reliability_curve_skips_empty_bins():
    p = [0.05, 0.95, 0.95]
    y = [0, 1, 1]
    xs, ys, ns = reliability_curve(p, y, n_bins=10)
    # Only two bins should be populated.
    assert len(xs) == 2
    assert sum(ns) == 3


def test_platt_recalibrate_improves_overconfidence():
    # Generate overconfident calibration.
    rng = np.random.default_rng(2)
    n = 500
    raw = rng.beta(0.5, 0.5, n)  # u-shape, mostly 0 or 1
    y = (rng.uniform(0, 1, n) < 0.5).astype(int)
    raw_ece = expected_calibration_error(raw, y, n_bins=10).ece
    scaler = platt_recalibrate(raw, y)
    cal = scaler.transform(raw)
    cal_ece = expected_calibration_error(cal, y, n_bins=10).ece
    assert cal_ece <= raw_ece + 0.01  # at least no worse


def test_isotonic_recalibrate_is_monotone():
    p = np.linspace(0, 1, 50)
    y = (p > 0.5).astype(int)
    scaler = isotonic_recalibrate(p, y)
    out = scaler.transform(p)
    # monotone non-decreasing
    diffs = np.diff(out)
    assert np.all(diffs >= -1e-9)


def test_ece_rejects_invalid_input():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 1.2], [1, 0])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.5], [1, 2])
