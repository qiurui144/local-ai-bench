from __future__ import annotations

from pathlib import Path

from common import ModelConfig
from benchmark.long_context.runner import (
    _append_case_result,
    _extract_expected_option,
    _extract_option,
    _estimate_tokens,
    _fit_text_middle,
    _fit_text_window,
    _fit_window_prompt,
    _keyword_recall_score,
    _score_longbench,
    _span_recall_score,
)
from run_benchmark import _is_long_context_required


def test_option_extraction_accepts_common_formats():
    assert _extract_option("(C) because...") == "C"
    assert _extract_option("The answer is B.") == "B"
    assert _extract_expected_option("(D) Full answer text") == "D"


def test_middle_fit_marks_truncation():
    fitted = _fit_text_middle("a" * 20000, 256)
    assert fitted.truncated is True
    assert "middle truncated" in fitted.text
    assert fitted.final_est_tokens < fitted.original_est_tokens


def test_window_fit_preserves_target_region():
    text = "A" * 5000 + " TARGET AIRPLANE MANUAL LINE " + "B" * 5000
    fitted, window = _fit_text_window(text, text.index("TARGET"), 256)
    assert fitted.truncated is True
    assert "TARGET AIRPLANE MANUAL LINE" in fitted.text
    assert window["window_start"] > 0
    assert window["window_end"] < len(text)


def test_prompt_window_fit_accounts_for_fixed_prompt_budget():
    text = "A" * 5000 + " TARGET AIRPLANE MANUAL LINE " + "B" * 5000
    prefix = "Context:\n"
    suffix = "\nQuestion: What follows the target line?"
    prompt, fitted, window = _fit_window_prompt(
        prefix,
        text,
        text.index("TARGET"),
        suffix,
        max_input_tokens=1024,
        target_context_tokens=1024,
        safety=0.70,
    )
    assert prompt.startswith(prefix)
    assert prompt.endswith(suffix)
    assert _estimate_tokens(prompt) < 1024
    assert "TARGET AIRPLANE MANUAL LINE" in prompt
    assert window["window_start"] > 0


def test_aviation_manual_scores_accept_partial_terms():
    expected = "Autoflight system provides guidance and control modes."
    assert _span_recall_score("guidance and control modes", expected) > 0.0
    assert _span_recall_score(expected, expected) == 1.0
    assert _keyword_recall_score("APU, ECAM, HYD", ["APU", "ECAM", "HYD"]) == 1.0


def test_longbench_retrieval_score_uses_paragraph_number():
    assert _score_longbench("passage_retrieval_en", "Paragraph 7", ["Paragraph 7"]) == 1.0
    assert _score_longbench("passage_retrieval_en", "Paragraph 3", ["Paragraph 7"]) == 0.0


def test_long_context_gate_is_required_for_20b_chat_models():
    assert _is_long_context_required(ModelConfig(name="qwen3-30b-a3b", task_type="text_only"))
    assert not _is_long_context_required(ModelConfig(name="qwen3-8b", task_type="text_only"))
    assert not _is_long_context_required(
        ModelConfig(name="qwen3-embedding-30b", task_type="text_only", capabilities=("embedding",))
    )


def test_case_result_checkpoint_writes_jsonl(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    _append_case_result({"case_result_log": str(path)}, "longbench", {"ok": True, "score": 1.0})

    assert path.read_text(encoding="utf-8").strip() == '{"suite": "longbench", "ok": true, "score": 1.0}'
