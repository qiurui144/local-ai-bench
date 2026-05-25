"""Tests for benchmark.rag.judge_calibration."""
from __future__ import annotations

import pytest

from benchmark.rag.judge_calibration import (
    GoldPair,
    calibration_report,
    probabilistic_judge_ece,
    replay_calibration_pairs,
)


def test_calibration_report_perfect_judge():
    pairs = [
        GoldPair(
            pair_id=f"p{i}",
            question="q",
            good_answer="good",
            bad_answer="bad",
        )
        for i in range(5)
    ]

    def perfect_judge(pair, good_is_A):
        good = "A" if good_is_A else "B"
        return [{"winner": good}, {"winner": good}, {"winner": good}]

    rows = replay_calibration_pairs(pairs, perfect_judge, n_runs_per_pair=3, swap_order=True)
    rpt = calibration_report(rows, pairs, n_runs_per_pair=3)
    assert rpt.accuracy == pytest.approx(1.0)
    assert rpt.consistency == pytest.approx(1.0)


def test_calibration_report_random_judge():
    import random

    rng = random.Random(0)
    pairs = [
        GoldPair(pair_id=f"p{i}", question="q", good_answer="good", bad_answer="bad")
        for i in range(20)
    ]

    def random_judge(pair, good_is_A):
        return [{"winner": rng.choice(["A", "B"])} for _ in range(3)]

    rows = replay_calibration_pairs(pairs, random_judge, n_runs_per_pair=3, swap_order=False)
    rpt = calibration_report(rows, pairs, n_runs_per_pair=3)
    # Random judge ~50% accuracy
    assert 0.3 < rpt.accuracy < 0.7


def test_calibration_report_parse_failures_surface():
    pairs = [
        GoldPair(pair_id="p1", question="q", good_answer="good", bad_answer="bad")
    ]

    def broken_judge(pair, good_is_A):
        return [{"_parse_error": "x"}, {"_parse_error": "y"}, {"winner": "A"}]

    rows = replay_calibration_pairs(pairs, broken_judge, n_runs_per_pair=3, swap_order=False)
    rpt = calibration_report(rows, pairs, n_runs_per_pair=3)
    assert rpt.parse_failure_rate > 0


def test_probabilistic_judge_ece_returns_metrics():
    probs = [0.9] * 10 + [0.1] * 10
    correct = [1] * 10 + [0] * 10
    out = probabilistic_judge_ece(probs, correct, n_bins=5)
    assert "ece" in out
    assert "brier" in out
    assert "mce" in out


def test_calibration_position_bias_zero_when_swapped_equal():
    pairs = [
        GoldPair(pair_id=f"p{i}", question="q", good_answer="g", bad_answer="b")
        for i in range(10)
    ]

    def balanced_judge(pair, good_is_A):
        good = "A" if good_is_A else "B"
        return [{"winner": good}] * 3

    rows = replay_calibration_pairs(pairs, balanced_judge, n_runs_per_pair=3, swap_order=True)
    rpt = calibration_report(rows, pairs, n_runs_per_pair=3)
    assert abs(rpt.position_bias) < 0.2  # near zero


def test_calibration_no_rows_raises():
    pairs = [GoldPair(pair_id="p1", question="q", good_answer="g", bad_answer="b")]
    with pytest.raises(ValueError):
        calibration_report([], pairs, n_runs_per_pair=1)
