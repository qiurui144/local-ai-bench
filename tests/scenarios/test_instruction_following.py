"""Unit tests for S4: instruction_following scenario."""
import pytest
from benchmark.scenarios.base import ScenarioCase
from benchmark.scenarios.instruction_following import SPEC, _check


# ─────────────────────────────────────────────────────────────
# _check: per-instruction verifier
# ─────────────────────────────────────────────────────────────

class TestCheck:
    def test_json_valid_ok(self):
        assert _check('{"a": 1}', {"type": "json_valid"})

    def test_json_valid_fail(self):
        assert not _check("not json", {"type": "json_valid"})

    def test_json_has_keys_dict(self):
        assert _check('{"name": "x", "price": 1}', {"type": "json_has_keys", "value": ["name", "price"]})

    def test_json_has_keys_list_first_item(self):
        assert _check('[{"language": "Python", "use_case": "data"}]',
                      {"type": "json_has_keys", "value": ["language", "use_case"]})

    def test_json_has_keys_missing_key(self):
        assert not _check('{"name": "x"}', {"type": "json_has_keys", "value": ["name", "price"]})

    def test_starts_with_match(self):
        assert _check("  总结：分布式系统很复杂", {"type": "starts_with", "value": "总结："})

    def test_starts_with_no_match(self):
        assert not _check("分布式系统很复杂", {"type": "starts_with", "value": "总结："})

    def test_ends_with_match(self):
        assert _check("这是总结。", {"type": "ends_with", "value": "总结。"})

    def test_must_include_case_insensitive(self):
        assert _check("Hello World", {"type": "must_include", "value": "hello"})

    def test_must_include_miss(self):
        assert not _check("Hello World", {"type": "must_include", "value": "python"})

    def test_must_exclude_ok(self):
        assert _check("Some text with no such keyword here", {"type": "must_exclude", "value": "banned"})

    def test_must_exclude_fail(self):
        assert not _check("This contains banned content", {"type": "must_exclude", "value": "banned"})

    def test_bullet_items_min(self):
        text = "- item one\n- item two\n- item three"
        assert _check(text, {"type": "bullet_items_min", "value": 3})
        assert not _check(text, {"type": "bullet_items_min", "value": 4})

    def test_bullet_items_various_markers(self):
        text = "• item1\n* item2\n- item3"
        assert _check(text, {"type": "bullet_items_min", "value": 3})

    def test_numbered_items_min(self):
        text = "1. first\n2. second\n3. third\n4. fourth"
        assert _check(text, {"type": "numbered_items_min", "value": 4})
        assert not _check(text, {"type": "numbered_items_min", "value": 5})

    def test_char_count_max_pass(self):
        assert _check("short", {"type": "char_count_max", "value": 10})

    def test_char_count_max_fail(self):
        assert not _check("this is a long string", {"type": "char_count_max", "value": 5})

    def test_char_count_min_pass(self):
        assert _check("long enough string", {"type": "char_count_min", "value": 5})

    def test_char_count_min_fail(self):
        assert not _check("hi", {"type": "char_count_min", "value": 100})

    def test_unknown_type_returns_false(self):
        assert not _check("anything", {"type": "unknown_type", "value": None})


# ─────────────────────────────────────────────────────────────
# l1_score: uses raw_content (not parsed_json)
# ─────────────────────────────────────────────────────────────

def _case(instructions):
    return ScenarioCase(id="t1", provenance="curated", payload={
        "prompt": "test prompt",
        "instructions": instructions,
    })


def test_l1_all_satisfied():
    case = _case([
        {"type": "must_include", "value": "python"},
        {"type": "char_count_max", "value": 100},
    ])
    s = SPEC.l1_score(case, None, "I love Python programming.")
    assert s["compliance_rate"] == 1.0
    assert s["n_satisfied"] == 2


def test_l1_partially_satisfied():
    case = _case([
        {"type": "must_include", "value": "python"},
        {"type": "char_count_max", "value": 3},   # fail: too long
    ])
    s = SPEC.l1_score(case, None, "Python is great.")
    assert s["compliance_rate"] == 0.5
    assert s["n_satisfied"] == 1


def test_l1_no_instructions():
    case = _case([])
    s = SPEC.l1_score(case, None, "anything")
    assert s["compliance_rate"] == 1.0
    assert s["n_instructions"] == 0


def test_l1_empty_content():
    case = _case([{"type": "must_include", "value": "hello"}])
    s = SPEC.l1_score(case, None, "")
    assert s["compliance_rate"] == 0.0


def test_l1_parsed_json_ignored_for_scoring():
    case = _case([{"type": "must_include", "value": "raw"}])
    # parsed_json says nothing about "raw"; raw_content has it
    s = SPEC.l1_score(case, {"key": "value"}, "the raw output has raw content")
    assert s["compliance_rate"] == 1.0


# ─────────────────────────────────────────────────────────────
# aggregate
# ─────────────────────────────────────────────────────────────

def test_aggregate_mean():
    agg = SPEC.aggregate_l1([
        {"compliance_rate": 1.0, "n_instructions": 2, "n_satisfied": 2},
        {"compliance_rate": 0.5, "n_instructions": 2, "n_satisfied": 1},
    ])
    assert agg["compliance_rate"] == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────
# spec shape
# ─────────────────────────────────────────────────────────────

def test_spec_shape():
    assert SPEC.name == "instruction_following"
    assert not SPEC.requires_vlm
    assert SPEC.cases_path == "datasets/scenarios/instruction_following/cases.jsonl"
    assert "compliance_rate_min" in SPEC.default_thresholds


def test_build_prompt_includes_task():
    case = _case([{"type": "json_valid", "value": None}])
    prompt, image = SPEC.build_prompt(case)
    assert image is None
    assert "test prompt" in prompt
    assert "格式" in prompt  # preamble
