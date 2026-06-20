"""Tests for benchmark.rigor.effect_size."""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.rigor.effect_size import (
    cles,
    cliffs_delta,
    cohens_d,
    effect_size_report,
    glass_delta,
    hedges_g,
    odds_ratio,
    rank_biserial,
    risk_difference,
)


def test_cohens_d_zero_when_equal():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.0, 2.0, 3.0, 4.0]
    e = cohens_d(a, b)
    assert e.value == 0.0
    assert e.magnitude == "negligible"


def test_cohens_d_large_when_far_apart():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50)
    b = rng.normal(2, 1, 50)
    e = cohens_d(a, b)
    assert e.value < -1.5
    assert e.magnitude == "large"


def test_hedges_g_close_to_cohen_d_for_large_n():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0.5, 1, 200)
    d = cohens_d(a, b).value
    g = hedges_g(a, b).value
    assert abs(d - g) < 0.05


def test_glass_delta_uses_control_sd():
    treatment = [10, 12, 14, 16, 18]
    control = [1, 2, 3, 4, 5]
    e = glass_delta(treatment, control)
    assert e.value > 5.0  # large effect


def test_cliffs_delta_range():
    a = [1, 2, 3]
    b = [4, 5, 6]
    e = cliffs_delta(a, b)
    assert -1.0 <= e.value <= 1.0
    assert e.value == -1.0  # all of a less than all of b


def test_cliffs_delta_identical():
    a = [1, 2, 3]
    b = [1, 2, 3]
    e = cliffs_delta(a, b)
    assert e.value == 0.0


def test_rank_biserial_returns_in_range():
    e = rank_biserial([1, 2, 3], [4, 5, 6])
    assert -1.0 <= e.value <= 1.0


def test_odds_ratio_haldane_correction():
    e = odds_ratio(a_success=0, a_total=10, b_success=5, b_total=10, haldane_correction=True)
    assert e.value > 0  # finite due to +0.5 correction


def test_risk_difference_range():
    e = risk_difference(80, 100, 20, 100)
    assert e.value == pytest.approx(0.6, abs=1e-6)


def test_cles_basic():
    e = cles([1, 2, 3], [0, 0, 0])
    assert e.value == 1.0


def test_effect_size_report_has_all_estimators():
    rng = np.random.default_rng(2)
    a = rng.normal(0.5, 1, 30)
    b = rng.normal(0, 1, 30)
    rpt = effect_size_report(a, b)
    for key in ("cohens_d", "hedges_g", "cliffs_delta", "rank_biserial", "cles"):
        assert key in rpt
