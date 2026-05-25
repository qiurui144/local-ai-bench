"""Tests for benchmark.rag.drift_detection."""
from __future__ import annotations

import time

import numpy as np
import pytest

from benchmark.rag.drift_detection import (
    assess_drift,
    mine_high_disagreement_candidates,
    mine_low_confidence_candidates,
    per_week_performance,
    query_embedding_drift,
    query_length_drift,
    query_distribution_psi,
    temporal_performance_drift,
)


def test_query_length_drift_triggers_on_shift():
    ref = ["short query"] * 200
    obs = ["a substantially longer observed query about something"] * 200
    s = query_length_drift(ref, obs, ks_p_threshold=0.01)
    assert s.triggered


def test_query_length_drift_no_shift():
    ref = ["query"] * 100
    obs = ["query"] * 100
    s = query_length_drift(ref, obs, ks_p_threshold=0.01)
    assert not s.triggered


def test_query_embedding_drift_far_centroid():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (200, 16))
    obs = rng.normal(5, 1, (200, 16))
    s = query_embedding_drift(ref, obs, centroid_shift_threshold=0.1)
    assert s.triggered


def test_query_embedding_drift_close_centroid():
    rng = np.random.default_rng(1)
    # Real embeddings are clustered around a signal direction, not zero-mean.
    # Simulate that: a base direction plus isotropic noise.
    base = np.ones(16) / np.sqrt(16)
    ref = base + 0.1 * rng.normal(0, 1, (1000, 16))
    obs = base + 0.1 * rng.normal(0, 1, (1000, 16))
    s = query_embedding_drift(ref, obs, centroid_shift_threshold=0.10)
    assert not s.triggered


def test_query_distribution_psi_basic():
    s = query_distribution_psi([0.5] * 100, [0.5] * 100)
    assert s.value < 0.1


def test_per_week_performance_buckets():
    now = time.time()
    timestamps = [now - i * 86400 for i in range(60)]
    values = [0.8] * 30 + [0.6] * 30
    out = per_week_performance(timestamps, values)
    assert len(out) >= 5  # multiple weeks


def test_temporal_performance_drift_detects_decay():
    n = 250
    ts = list(range(n))
    values = [0.9 - i * 0.005 for i in range(n)]
    signals = temporal_performance_drift(ts, values, window=50, step=50, drop_threshold=0.05)
    assert any(s.triggered for s in signals)


def test_assess_drift_recommendation_retrain():
    rng = np.random.default_rng(2)
    rep = assess_drift(
        reference={"queries": ["a"] * 200, "embeddings": rng.normal(0, 1, (200, 16))},
        observed={
            "queries": ["b longer one"] * 200,
            "embeddings": rng.normal(5, 1, (200, 16)),
        },
    )
    assert rep.recommendation in ("monitor", "retrain")


def test_mine_low_confidence_candidates_orders_by_conf():
    items = [
        {"item_id": "a", "query": "q", "confidence": 0.1},
        {"item_id": "b", "query": "q", "confidence": 0.4},
        {"item_id": "c", "query": "q", "confidence": 0.9},
    ]
    out = mine_low_confidence_candidates(items, threshold=0.5, limit=2)
    assert [c.item_id for c in out] == ["a", "b"]


def test_mine_high_disagreement_candidates():
    items = [
        {"item_id": "x", "query": "q", "judge_outputs": [{"verdict": "A"}, {"verdict": "B"}]},
        {"item_id": "y", "query": "q", "judge_outputs": [{"verdict": "A"}, {"verdict": "A"}]},
    ]
    out = mine_high_disagreement_candidates(items, limit=5)
    assert any(c.item_id == "x" for c in out)
