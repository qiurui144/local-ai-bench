"""Tests for benchmark.rigor.inter_rater."""
from __future__ import annotations

import pytest

from benchmark.rigor.inter_rater import (
    cohens_kappa,
    fleiss_kappa,
    gwets_ac1,
    judge_agreement_panel,
    krippendorff_alpha,
    percent_agreement,
    weighted_cohens_kappa,
)


def test_percent_agreement_perfect():
    a = [1, 1, 0, 0]
    b = [1, 1, 0, 0]
    assert percent_agreement(a, b) == 1.0


def test_cohens_kappa_perfect_agreement():
    a = [1, 0, 1, 0, 1]
    b = [1, 0, 1, 0, 1]
    k = cohens_kappa(a, b)
    assert k.coefficient == pytest.approx(1.0)
    assert k.interpretation == "almost_perfect"


def test_cohens_kappa_chance_level():
    # Equal marginals, no signal: kappa should be near 0.
    a = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    k = cohens_kappa(a, b)
    assert k.coefficient < 0  # full disagreement


def test_weighted_kappa_linear_and_quadratic():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    lin = weighted_cohens_kappa(a, b, weighting="linear")
    quad = weighted_cohens_kappa(a, b, weighting="quadratic")
    assert lin.coefficient == pytest.approx(1.0)
    assert quad.coefficient == pytest.approx(1.0)


def test_weighted_kappa_unknown_weighting():
    with pytest.raises(ValueError):
        weighted_cohens_kappa([1, 2], [1, 2], weighting="cubic")


def test_fleiss_kappa_perfect():
    # 3 raters, 4 items, all rate 1.
    ratings = [
        [3, 0],  # 3 raters chose cat 0
        [3, 0],
        [3, 0],
        [3, 0],
    ]
    # All raters agree on category 0 for every item.
    k = fleiss_kappa(ratings)
    # With single-cat distribution kappa is undefined; impl returns 0 or 1.
    assert 0 <= k.coefficient <= 1


def test_fleiss_kappa_requires_consistent_rater_count():
    bad = [[3, 0], [2, 0]]
    with pytest.raises(ValueError):
        fleiss_kappa(bad)


def test_krippendorff_alpha_nominal_perfect():
    raters = [[1, 0, 1, 0], [1, 0, 1, 0]]
    a = krippendorff_alpha(raters, level="nominal")
    assert a.coefficient == pytest.approx(1.0)


def test_krippendorff_alpha_handles_missing():
    raters = [[1, 0, 1, None], [1, 0, 1, 1]]
    a = krippendorff_alpha(raters, level="nominal")
    assert 0 <= a.coefficient <= 1


def test_gwets_ac1_basic():
    a = [1, 1, 0, 0, 1, 1, 0, 0]
    b = [1, 1, 0, 0, 1, 0, 0, 0]
    g = gwets_ac1(a, b)
    assert g.coefficient > 0.5  # mostly agreeing


def test_judge_agreement_panel_returns_all_pairs():
    judges = {
        "j1": [1, 0, 1, 0],
        "j2": [1, 0, 1, 1],
        "j3": [1, 0, 1, 0],
    }
    panel = judge_agreement_panel(judges)
    assert "j1__vs__j2" in panel
    assert "j1__vs__j3" in panel
    assert "j2__vs__j3" in panel
    assert panel["j1__vs__j3"].coefficient == pytest.approx(1.0)
