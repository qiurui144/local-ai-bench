"""Tests for benchmark.rigor.power_analysis."""
from __future__ import annotations

import pytest

from benchmark.rigor.power_analysis import (
    minimum_detectable_effect_two_means,
    post_hoc_power_two_means,
    sample_size_paired,
    sample_size_two_means,
    sample_size_two_means_exact,
    sample_size_two_proportions,
)


def test_sample_size_two_means_basic():
    res = sample_size_two_means(effect_size=0.5, alpha=0.05, power=0.80)
    # Cohen's textbook: ~64 per group for d=0.5.
    assert 60 <= res.n_per_group <= 70


def test_sample_size_two_means_exact_matches_approx_at_high_n():
    approx = sample_size_two_means(effect_size=0.3, alpha=0.05, power=0.80)
    exact = sample_size_two_means_exact(effect_size=0.3, alpha=0.05, power=0.80)
    assert abs(approx.n_per_group - exact.n_per_group) <= 5


def test_sample_size_two_proportions_basic():
    res = sample_size_two_proportions(p1=0.5, p2=0.6, alpha=0.05, power=0.80)
    assert res.n_per_group > 0
    assert res.total_n == 2 * res.n_per_group


def test_sample_size_paired_smaller_than_independent():
    paired = sample_size_paired(effect_size=0.5, alpha=0.05, power=0.80)
    indep = sample_size_two_means(effect_size=0.5, alpha=0.05, power=0.80)
    # Paired needs ~half (for equal variance pairs).
    assert paired.n_per_group < indep.n_per_group


def test_post_hoc_power_reasonable():
    res = post_hoc_power_two_means(observed_d=0.5, n_per_group=64, alpha=0.05)
    assert 0.7 < res.power < 0.9


def test_mde_inverse_of_sample_size():
    n = 100
    mde = minimum_detectable_effect_two_means(n_per_group=n, alpha=0.05, power=0.80)
    # Re-derive n from this mde and see they round-trip.
    res = sample_size_two_means(effect_size=mde, alpha=0.05, power=0.80)
    assert abs(res.n_per_group - n) < 10


def test_invalid_effect_size_raises():
    with pytest.raises(ValueError):
        sample_size_two_means(effect_size=0.0)
    with pytest.raises(ValueError):
        sample_size_two_means(effect_size=-0.1)


def test_invalid_proportions_raise():
    with pytest.raises(ValueError):
        sample_size_two_proportions(p1=0.5, p2=0.5)
    with pytest.raises(ValueError):
        sample_size_two_proportions(p1=0.0, p2=0.5)
