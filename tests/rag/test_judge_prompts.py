"""Tests for benchmark.rag.judge_prompts."""
from __future__ import annotations

import pytest

from benchmark.rag.judge_prompts import (
    JudgeConfig,
    aggregate_runs,
    build_groundedness_prompt,
    build_pairwise_prompt,
    build_relevance_prompt,
    run_judge,
)


def test_build_groundedness_prompt_includes_question_and_evidence():
    cfg = JudgeConfig(n_runs=3)
    msgs = build_groundedness_prompt(
        question="why is the sky blue?",
        answer="Because of Rayleigh scattering",
        evidence=[{"id": "e1", "text": "Rayleigh scattering explanation"}],
        config=cfg,
    )
    assert msgs[0]["role"] == "system"
    assert "evidence-only" in msgs[0]["content"].lower()
    assert "e1" in msgs[1]["content"]


def test_build_relevance_prompt_has_user_content():
    cfg = JudgeConfig()
    msgs = build_relevance_prompt("Q", "A", cfg)
    assert msgs[1]["role"] == "user"
    assert "Q" in msgs[1]["content"]
    assert "A" in msgs[1]["content"]


def test_build_pairwise_prompt_has_winner_field_template():
    msgs = build_pairwise_prompt(
        question="Q",
        answer_a="Alpha",
        answer_b="Beta",
        evidence=[{"id": "e1", "text": "evidence"}],
    )
    assert any("winner" in m["content"] for m in msgs)


def test_aggregate_runs_median():
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    assert aggregate_runs(rows, "x", "median") == 2.0


def test_aggregate_runs_majority_truthy():
    rows = [{"v": True}, {"v": True}, {"v": False}]
    assert aggregate_runs(rows, "v", "majority") == 1.0


def test_aggregate_runs_unknown_method():
    with pytest.raises(ValueError):
        aggregate_runs([{"v": 1}], "v", "no_such_method")


def test_run_judge_parses_clean_json():
    cfg = JudgeConfig(n_runs=2)
    msgs = [{"role": "user", "content": "x"}]

    def fake_invoke(messages, temperature):
        return '{"winner": "A", "confidence": 0.8}'

    out = run_judge(fake_invoke, msgs, cfg)
    assert len(out) == 2
    assert out[0]["winner"] == "A"


def test_run_judge_strips_code_fences():
    cfg = JudgeConfig(n_runs=1)
    msgs = [{"role": "user", "content": "x"}]

    def fake_invoke(messages, temperature):
        return '```json\n{"winner": "B"}\n```'

    out = run_judge(fake_invoke, msgs, cfg)
    assert out[0]["winner"] == "B"


def test_run_judge_surfaces_parse_error():
    cfg = JudgeConfig(n_runs=1)
    msgs = [{"role": "user", "content": "x"}]

    def bad_invoke(messages, temperature):
        return "not json at all"

    out = run_judge(bad_invoke, msgs, cfg)
    assert "_parse_error" in out[0]
