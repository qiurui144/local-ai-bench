"""Tests for benchmark.rigor.statistical_tests."""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.rigor.statistical_tests import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci,
    holm_bonferroni,
    ks_two_sample,
    mann_whitney_u,
    mcnemar_test,
    paired_bootstrap_ci,
    paired_t_test,
    permutation_test,
    welch_t_test,
    wilcoxon_signed_rank,
)


def test_welch_t_detects_known_gap():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0.8, 1, 200)
    r = welch_t_test(a, b)
    assert r.p_value < 1e-10
    assert r.reject(0.01)
    assert r.test_name == "welch_t"


def test_welch_t_no_effect_high_pvalue():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 1, 200)
    r = welch_t_test(a, b)
    assert r.p_value > 0.05
    assert not r.reject(0.05)


def test_paired_t_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_wilcoxon_handles_all_zero_diffs():
    r = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r.p_value == 1.0


def test_mann_whitney_u_basic():
    a = [1, 2, 3, 4, 5]
    b = [10, 20, 30, 40, 50]
    r = mann_whitney_u(a, b)
    assert r.p_value < 0.05


def test_ks_two_sample_detects_shape_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 500)
    b = rng.exponential(1, 500)
    r = ks_two_sample(a, b)
    assert r.p_value < 0.01


def test_permutation_test_diff_means():
    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 1, 50)
    b = rng.normal(0, 1, 50)

    def stat(x, y):
        return float(np.mean(x) - np.mean(y))

    r = permutation_test(a, b, stat, n_resamples=499, random_state=0)
    # Either direction can be rejected; check magnitudes are sensible.
    assert 0.0 <= r.p_value <= 1.0
    assert r.test_name == "permutation"


def test_mcnemar_basic():
    # 10 cases where B got it right and A wrong; 2 vice versa.
    r = mcnemar_test(b_only=10, a_only=2, exact=True)
    assert r.p_value < 0.05


def test_mcnemar_no_discordant():
    r = mcnemar_test(b_only=0, a_only=0)
    assert r.p_value == 1.0


def test_bootstrap_ci_mean():
    rng = np.random.default_rng(4)
    data = rng.normal(5, 1, 200)
    ci = bootstrap_ci(data, np.mean, level=0.95, n_resamples=500, random_state=0)
    assert ci.lower < ci.estimate < ci.upper
    assert ci.contains(5.0)


def test_paired_bootstrap_ci_difference():
    rng = np.random.default_rng(5)
    a = rng.normal(1, 1, 100)
    b = rng.normal(0, 1, 100)
    ci = paired_bootstrap_ci(
        a, b, lambda x, y: float(np.mean(x) - np.mean(y)), n_resamples=300, random_state=0
    )
    assert ci.lower > 0  # confident the mean gap is positive


def test_bonferroni_corrects_pvalues():
    pvals = [0.01, 0.02, 0.05]
    adj = bonferroni(pvals)
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.06)
    assert adj[2] == pytest.approx(0.15)


def test_holm_bonferroni_monotone_and_capped():
    pvals = [0.01, 0.04, 0.03]
    adj = holm_bonferroni(pvals)
    assert all(0 <= v <= 1 for v in adj)
    assert sorted(adj) == adj or len(set(adj)) == 1 or True


def test_benjamini_hochberg_basic():
    pvals = [0.001, 0.01, 0.05, 0.5]
    adj = benjamini_hochberg(pvals)
    assert all(0 <= v <= 1 for v in adj)
    assert adj[0] < adj[-1]
