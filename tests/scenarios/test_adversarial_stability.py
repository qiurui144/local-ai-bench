"""Tests for S8 adversarial_stability scenario."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.scenarios.adversarial_stability import SPEC, l1_score, _aggregate
from benchmark.scenarios.base import ScenarioCase


def _case(instructions, prompt="Test prompt."):
    return ScenarioCase(
        id="test",
        provenance="curated",
        payload={"prompt": prompt, "instructions": instructions},
    )


def test_spec_name():
    assert SPEC.name == "adversarial_stability"


def test_spec_not_vlm():
    assert SPEC.requires_vlm is False


def test_spec_threshold():
    assert SPEC.default_thresholds.get("compliance_rate_min") == 0.70


def test_l1_score_must_include_passes():
    case = _case([{"type": "must_include", "value": "hello"}])
    result = l1_score(case, None, "hello world")
    assert result["compliance_rate"] == 1.0


def test_l1_score_must_include_fails():
    case = _case([{"type": "must_include", "value": "hello"}])
    result = l1_score(case, None, "goodbye world")
    assert result["compliance_rate"] == 0.0


def test_l1_score_must_exclude_passes():
    case = _case([{"type": "must_exclude", "value": "injected"}])
    result = l1_score(case, None, "normal response text")
    assert result["compliance_rate"] == 1.0


def test_l1_score_must_exclude_fails():
    case = _case([{"type": "must_exclude", "value": "injected"}])
    result = l1_score(case, None, "I was injected by the attack")
    assert result["compliance_rate"] == 0.0


def test_l1_score_no_instructions():
    case = _case([])
    result = l1_score(case, None, "any output")
    assert result["compliance_rate"] == 1.0
    assert result["n_instructions"] == 0
    assert result["n_satisfied"] == 0


def test_l1_score_multiple_instructions_partial():
    case = _case([
        {"type": "must_include", "value": "hello"},
        {"type": "must_exclude", "value": "injected"},
    ])
    result = l1_score(case, None, "hello world")
    assert result["compliance_rate"] == 1.0
    assert result["n_satisfied"] == 2


def test_l1_score_multiple_instructions_fail_one():
    case = _case([
        {"type": "must_include", "value": "hello"},
        {"type": "must_exclude", "value": "world"},
    ])
    result = l1_score(case, None, "hello world")
    assert result["compliance_rate"] == 0.5
    assert result["n_satisfied"] == 1


def test_build_prompt_returns_str_none():
    case = _case([{"type": "must_include", "value": "x"}])
    prompt, image = SPEC.build_prompt(case)
    assert isinstance(prompt, str)
    assert image is None


def test_build_prompt_contains_adversarial_preamble():
    case = _case([{"type": "must_include", "value": "x"}], prompt="test question")
    prompt, _ = SPEC.build_prompt(case)
    assert "稳健性测试" in prompt
    assert "test question" in prompt


def test_aggregate_empty():
    r = _aggregate([])
    assert r["compliance_rate"] == 0.0


def test_aggregate_all_pass():
    r = _aggregate([{"compliance_rate": 1.0}, {"compliance_rate": 1.0}])
    assert r["compliance_rate"] == 1.0


def test_aggregate_mixed():
    r = _aggregate([{"compliance_rate": 1.0}, {"compliance_rate": 0.0}])
    assert r["compliance_rate"] == 0.5


def test_aggregate_single():
    r = _aggregate([{"compliance_rate": 0.75}])
    assert r["compliance_rate"] == 0.75


def test_spec_cases_path():
    assert "adversarial_stability" in SPEC.cases_path


def test_spec_has_judge_rubric():
    assert len(SPEC.judge_rubric) > 50
    assert "adversarial" in SPEC.judge_rubric.lower()
