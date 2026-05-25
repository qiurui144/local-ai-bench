"""Tests for benchmark.rigor.ood_assessment."""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.rigor.ood_assessment import (
    KNNOODDetector,
    domain_shift_score,
    psi,
    subgroup_audit,
    temporal_drift,
)


def test_domain_shift_ks_no_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 500)
    obs = rng.normal(0, 1, 500)
    rep = domain_shift_score(ref, obs, metric="ks")
    assert rep.p_value > 0.01


def test_domain_shift_ks_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 500)
    obs = rng.normal(2, 1, 500)
    rep = domain_shift_score(ref, obs, metric="ks")
    assert rep.p_value < 0.01
    assert rep.interpretation == "shift"


def test_domain_shift_wasserstein_positive():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 200)
    obs = rng.normal(1, 1, 200)
    rep = domain_shift_score(ref, obs, metric="wasserstein")
    assert rep.distance > 0


def test_domain_shift_jensen_shannon():
    rng = np.random.default_rng(2)
    ref = rng.uniform(0, 1, 500)
    obs = rng.uniform(0, 1, 500)
    rep = domain_shift_score(ref, obs, metric="jensen_shannon")
    # similar distributions; JS should be small.
    assert rep.distance < 0.2


def test_psi_zero_when_identical():
    data = [0.5] * 100
    val = psi(data, data, n_bins=10)
    assert val < 0.1


def test_psi_large_when_distributions_differ():
    # PSI uses ref-quantile binning, so ref must span a range.
    rng = np.random.default_rng(10)
    ref = rng.uniform(0, 1, 500).tolist()
    obs = rng.uniform(0.5, 1.5, 500).tolist()
    val = psi(ref, obs, n_bins=10)
    assert val > 0.25


def test_temporal_drift_returns_windows():
    timestamps = list(range(200))
    values = [0.5 + i * 0.001 for i in range(200)]
    out = temporal_drift(values, timestamps, window=50, step=25)
    assert len(out) > 1
    for w in out:
        assert "mean" in w and "ks_p" in w


def test_knn_ood_detector_in_distribution_low_score():
    rng = np.random.default_rng(3)
    train = rng.normal(0, 1, (200, 16))
    det = KNNOODDetector(k=5)
    det.fit(train, calibration_quantile=0.95)
    # New point from same dist:
    test = rng.normal(0, 1, (10, 16))
    results = det.score(test)
    in_dist_count = sum(1 for r in results if not r.is_ood)
    assert in_dist_count >= 5  # most should be in-distribution


def test_knn_ood_detector_out_of_distribution_high_score():
    rng = np.random.default_rng(4)
    # Use a tightly-clustered training set (small isotropic Gaussian
    # in a fixed direction); OOD samples come from an orthogonal subspace
    # so cosine distance is large.
    base_dir = rng.normal(0, 1, 16)
    base_dir /= np.linalg.norm(base_dir)
    train = base_dir[None, :] + 0.05 * rng.normal(0, 1, (200, 16))
    det = KNNOODDetector(k=5)
    det.fit(train, calibration_quantile=0.95)
    # OOD samples drawn from a different direction.
    other_dir = rng.normal(0, 1, 16)
    other_dir -= other_dir.dot(base_dir) * base_dir  # orthogonalize
    other_dir /= np.linalg.norm(other_dir)
    far = other_dir[None, :] + 0.05 * rng.normal(0, 1, (10, 16))
    results = det.score(far)
    ood_count = sum(1 for r in results if r.is_ood)
    assert ood_count >= 5


def test_subgroup_audit_flags_outlier():
    scores = [0.9] * 50 + [0.1] * 10
    labels = ["A"] * 50 + ["B"] * 10
    audit = subgroup_audit(scores, labels, gap_threshold=0.2)
    assert "B" in audit.flagged_subgroups
    assert audit.worst_subgroup_gap > 0


def test_subgroup_audit_no_flag_when_uniform():
    scores = [0.5] * 30
    labels = (["A"] * 15) + (["B"] * 15)
    audit = subgroup_audit(scores, labels, gap_threshold=0.1)
    assert audit.flagged_subgroups == []
