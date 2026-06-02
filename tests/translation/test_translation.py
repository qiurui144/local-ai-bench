"""Tests for benchmark.translation.

All tests here run on CPU with no vLLM / GPU dependency:
- SacreBLEU / chrF on known hyp/ref pairs (uses the ``sacrebleu`` package if
  installed, otherwise the pure-Python fallback — both bounded in [0, 100]).
- numerical validation of metric outputs.
- dataset loaders (Flores built-in fallback + custom JSONL).
- terminology match rate + prompt builders.
COMET requires a GPU and is asserted to skip gracefully (never raises).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from benchmark.translation import prompts
from benchmark.translation.accuracy import (
    COMET_UNAVAILABLE,
    compute_chrf,
    compute_comet,
    compute_sacrebleu,
    term_match_rate,
    validate_metrics,
)
from benchmark.translation.datasets import (
    TranslationPair,
    load_custom_jsonl,
    load_flores,
)

_HAS_SACREBLEU = importlib.util.find_spec("sacrebleu") is not None
_DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets" / "translation"


# --------------------------------------------------------------------------- #
# SacreBLEU / chrF
# --------------------------------------------------------------------------- #
def test_bleu_perfect_match_is_100():
    refs = ["The cat sat on the mat.", "It was a sunny day."]
    score = compute_sacrebleu(refs, refs, tgt_lang="en")
    assert score == pytest.approx(100.0, abs=1e-6)


def test_bleu_zero_on_total_mismatch():
    hyps = ["zzz qqq www"]
    refs = ["the quick brown fox"]
    score = compute_sacrebleu(hyps, refs, tgt_lang="en")
    assert 0.0 <= score < 5.0


def test_bleu_partial_match_in_range():
    hyps = ["the cat sat on a mat"]
    refs = ["the cat sat on the mat"]
    score = compute_sacrebleu(hyps, refs, tgt_lang="en")
    assert 0.0 < score < 100.0


def test_bleu_bounds_always_0_to_100():
    hyps = ["a", "b c", ""]
    refs = ["a", "b d", "x"]
    score = compute_sacrebleu(hyps, refs, tgt_lang="en")
    assert 0.0 <= score <= 100.0


def test_bleu_chinese_tokenizer_runs():
    # zh tokenizer path (char-level); just assert bounded + non-crash
    hyps = ["今天天气很好"]
    refs = ["今天天气很好"]
    score = compute_sacrebleu(hyps, refs, tgt_lang="zh")
    assert score == pytest.approx(100.0, abs=1e-3)


def test_chrf_perfect_match_is_100():
    refs = ["hello world"]
    score = compute_chrf(refs, refs)
    assert score == pytest.approx(100.0, abs=1e-6)


def test_chrf_partial_in_range():
    score = compute_chrf(["hello word"], ["hello world"])
    assert 0.0 < score < 100.0


def test_chrf_bounds():
    score = compute_chrf(["abc", ""], ["abd", "x"])
    assert 0.0 <= score <= 100.0


def test_empty_inputs_return_zero():
    assert compute_sacrebleu([], [], "en") == 0.0
    assert compute_chrf([], []) == 0.0


@pytest.mark.skipif(not _HAS_SACREBLEU, reason="sacrebleu package not installed")
def test_sacrebleu_package_used_when_available():
    # With the real package, an exact match on a sentence long enough to have
    # 4-grams scores 100 (single-word inputs legitimately score 0 — no 4-gram).
    refs = ["the quick brown fox jumps over the lazy dog"]
    assert compute_sacrebleu(refs, refs, tgt_lang="en") == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Numerical validation
# --------------------------------------------------------------------------- #
def test_validate_metrics_passes_clean():
    assert validate_metrics(["a translation"], 30.0, 55.0) == []


def test_validate_metrics_flags_all_empty():
    problems = validate_metrics(["", ""], 0.0, 0.0)
    assert any("empty" in p for p in problems)


def test_validate_metrics_flags_out_of_range():
    problems = validate_metrics(["x"], 150.0, -1.0)
    assert any("BLEU" in p for p in problems)
    assert any("chrF" in p for p in problems)


def test_validate_metrics_flags_nan():
    problems = validate_metrics(["x"], float("nan"), float("inf"))
    assert any("NaN/Inf" in p for p in problems)


# --------------------------------------------------------------------------- #
# Terminology match rate (L3)
# --------------------------------------------------------------------------- #
def test_term_match_rate_full():
    hyps = ["Vectorization is the first step of RAG."]
    glossaries = [{"向量化": "vectorization", "RAG": "RAG"}]
    out = term_match_rate(hyps, glossaries)
    assert out["term_match_rate"] == pytest.approx(1.0)
    assert out["matched_terms"] == 2


def test_term_match_rate_partial():
    hyps = ["Vectorization is the first step of the system."]
    glossaries = [{"向量化": "vectorization", "RAG": "RAG"}]
    out = term_match_rate(hyps, glossaries)
    assert out["term_match_rate"] == pytest.approx(0.5)


def test_term_match_rate_empty_glossary():
    out = term_match_rate(["anything"], [{}])
    assert out["total_terms"] == 0
    assert out["term_match_rate"] == 0.0


def test_term_match_case_insensitive():
    out = term_match_rate(["uses an EMBEDDING model"], [{"嵌入": "embedding"}])
    assert out["term_match_rate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def test_flores_builtin_fallback_zh_en(monkeypatch):
    # Force offline so the test exercises the built-in fallback fast and
    # deterministically (no HF network round-trip / retry loop).
    monkeypatch.setenv("TRANSLATION_OFFLINE", "1")
    pairs = load_flores("zh", "en", num_samples=3)
    assert len(pairs) <= 3
    assert all(isinstance(p, TranslationPair) for p in pairs)
    assert all(p.src and p.ref for p in pairs)
    assert all(p.src_lang == "zh" and p.tgt_lang == "en" for p in pairs)
    assert all(p.source == "builtin" for p in pairs)


def test_flores_rejects_unsupported_lang():
    with pytest.raises(ValueError):
        load_flores("fr", "en")


def test_custom_jsonl_loads_real_corpus():
    path = _DATASETS_DIR / "custom_zh_en.jsonl"
    assert path.exists(), "shipped custom corpus must exist"
    pairs = load_custom_jsonl(path)
    assert len(pairs) >= 40
    assert all(p.src and p.ref for p in pairs)
    assert all(p.source == "custom" for p in pairs)
    # at least some carry an L3 glossary
    assert any(p.glossary for p in pairs)
    # mixed directions present
    dirs = {(p.src_lang, p.tgt_lang) for p in pairs}
    assert ("zh", "en") in dirs and ("en", "zh") in dirs


def test_custom_jsonl_num_samples_cap():
    path = _DATASETS_DIR / "custom_zh_en.jsonl"
    pairs = load_custom_jsonl(path, num_samples=5)
    assert len(pairs) == 5


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
def test_l1_prompt_contains_source():
    p = prompts.l1_single_sentence("你好世界", "zh", "en")
    assert "你好世界" in p
    assert "English" in p


def test_l2_prompt_numbers_sentences():
    p = prompts.l2_context_consistency(["a", "b", "c"], "en", "zh")
    assert "1. a" in p and "3. c" in p
    assert "CONSISTENT" in p


def test_l3_prompt_lists_glossary():
    p = prompts.l3_terminology("向量化很重要", "zh", "en", {"向量化": "vectorization"})
    assert "向量化 -> vectorization" in p


# --------------------------------------------------------------------------- #
# COMET — GPU-only, must skip gracefully (never raises)
# --------------------------------------------------------------------------- #
def test_comet_skips_without_gpu():
    out = compute_comet(["hi"], ["你好"], ["hello"])
    assert out["available"] in (True, False)
    if not out["available"]:
        assert out["reason"] in (COMET_UNAVAILABLE, "empty input")


def test_comet_empty_input():
    out = compute_comet([], [], [])
    assert out["available"] is False
