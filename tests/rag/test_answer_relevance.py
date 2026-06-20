"""Tests for benchmark.rag.answer_relevance."""
from __future__ import annotations

import pytest

from benchmark.rag.answer_relevance import (
    JudgedAnswer,
    answer_relevance_report,
    bleu_4,
    chrf,
    intent_satisfaction_rate,
    over_refusal_rate,
    partial_credit_score,
    rouge_l,
    semantic_similarity_via_embeddings,
    under_refusal_rate,
)


def test_intent_satisfaction_rate():
    j = [
        JudgedAnswer("a", True, 1.0, False, False),
        JudgedAnswer("b", False, 0.0, False, False),
    ]
    assert intent_satisfaction_rate(j) == 0.5


def test_partial_credit_score_mean():
    j = [
        JudgedAnswer("a", True, 0.8, False, False),
        JudgedAnswer("b", True, 0.6, False, False),
    ]
    assert partial_credit_score(j) == pytest.approx(0.7)


def test_over_refusal_rate_counts_only_eligible():
    j = [
        JudgedAnswer("a", True, 1.0, refused=True, expected_refusal=False),  # over_refusal
        JudgedAnswer("b", True, 1.0, refused=False, expected_refusal=True),  # eligible for under
    ]
    assert over_refusal_rate(j) == 1.0  # only one not-expected-refusal item, and it refused
    assert under_refusal_rate(j) == 1.0


def test_rouge_l_self_is_one():
    r = rouge_l("the cat sat", "the cat sat")
    assert r["f1"] == pytest.approx(1.0)


def test_rouge_l_zero_when_no_overlap():
    r = rouge_l("hello world", "completely different")
    assert r["f1"] == 0.0


def test_bleu_4_perfect_match_high():
    score = bleu_4("the quick brown fox", "the quick brown fox")
    assert score > 0.5  # at least; BLEU has smoothing


def test_bleu_4_no_overlap_below_match():
    # With Laplace smoothing BLEU never reaches 0, but still well below
    # the perfect-match score.
    no_overlap = bleu_4("the quick brown fox", "zzzz xxxx yyyy wwww")
    match = bleu_4("the quick brown fox", "the quick brown fox")
    assert no_overlap < match * 0.6


def test_chrf_basic():
    score = chrf("abcabc", "abcabc")
    assert score == pytest.approx(1.0)


def test_chrf_zero_for_disjoint():
    score = chrf("aaa", "bbb")
    # chrf could be near zero since char n-grams differ.
    assert score < 0.1


def test_semantic_similarity_via_embeddings_self_one():
    s = semantic_similarity_via_embeddings([1.0, 0.0], [1.0, 0.0])
    assert s == pytest.approx(1.0)


def test_semantic_similarity_orthogonal_zero():
    s = semantic_similarity_via_embeddings([1.0, 0.0], [0.0, 1.0])
    assert s == pytest.approx(0.0)


def test_answer_relevance_report_aggregates():
    j = [
        JudgedAnswer("a", True, 1.0, False, False),
        JudgedAnswer("b", False, 0.5, False, False),
    ]
    rpt = answer_relevance_report(j)
    assert rpt.n_items == 2
    assert rpt.intent_satisfaction == 0.5
