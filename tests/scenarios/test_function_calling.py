"""Unit tests for S6: function_calling scenario."""
import pytest
from benchmark.scenarios.base import ScenarioCase
from benchmark.scenarios.function_calling import SPEC


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "获取天气",
    "parameters": {
        "city": {"type": "string", "required": True},
        "date": {"type": "string", "required": False},
    },
}

_SEARCH_TOOL = {
    "name": "search_web",
    "description": "搜索网页",
    "parameters": {
        "query": {"type": "string", "required": True},
    },
}


def _case(tools, messages, expected):
    return ScenarioCase(id="t1", provenance="curated", payload={
        "tools": tools,
        "messages": messages,
        "expected": expected,
    })


# ─────────────────────────────────────────────────────────────
# l1_score: name_match
# ─────────────────────────────────────────────────────────────

def test_name_match_exact():
    case = _case([_WEATHER_TOOL], [{"role": "user", "content": "北京天气"}],
                 {"name": "get_weather", "arguments": {"city": "北京"}})
    s = SPEC.l1_score(case, {"name": "get_weather", "arguments": {"city": "北京"}})
    assert s["name_match"] == 1


def test_name_match_wrong():
    case = _case([_WEATHER_TOOL, _SEARCH_TOOL],
                 [{"role": "user", "content": "北京天气"}],
                 {"name": "get_weather", "arguments": {"city": "北京"}})
    s = SPEC.l1_score(case, {"name": "search_web", "arguments": {"query": "北京天气"}})
    assert s["name_match"] == 0


# ─────────────────────────────────────────────────────────────
# l1_score: arg_recall / precision / f1
# ─────────────────────────────────────────────────────────────

def test_perfect_args():
    case = _case([_WEATHER_TOOL], [], {"name": "get_weather", "arguments": {"city": "北京"}})
    s = SPEC.l1_score(case, {"name": "get_weather", "arguments": {"city": "北京"}})
    assert s["arg_recall"] == 1.0
    assert s["arg_f1"] == 1.0


def test_arg_value_case_insensitive():
    case = _case([_WEATHER_TOOL], [],
                 {"name": "get_weather", "arguments": {"city": "beijing"}})
    s = SPEC.l1_score(case, {"name": "get_weather", "arguments": {"city": "Beijing"}})
    assert s["arg_recall"] == 1.0


def test_missing_required_arg():
    case = _case([_WEATHER_TOOL], [],
                 {"name": "get_weather", "arguments": {"city": "上海"}})
    s = SPEC.l1_score(case, {"name": "get_weather", "arguments": {}})
    assert s["arg_recall"] == 0.0
    assert s["arg_f1"] == 0.0


def test_no_expected_args_trivially_pass():
    case = _case([_WEATHER_TOOL], [],
                 {"name": "get_weather", "arguments": {}})
    s = SPEC.l1_score(case, {"name": "get_weather", "arguments": {}})
    assert s["arg_recall"] == 1.0
    assert s["arg_f1"] == 1.0


def test_extra_optional_args_reduce_precision():
    case = _case([_WEATHER_TOOL], [],
                 {"name": "get_weather", "arguments": {"city": "广州"}})
    # Model outputs extra arg "date" not in expected
    s = SPEC.l1_score(case, {"name": "get_weather", "arguments": {
        "city": "广州", "date": "2024-06-01"
    }})
    assert s["arg_recall"] == 1.0   # city matched
    assert s["arg_precision"] == 0.5  # 1/2 output args correct


def test_unparseable_output():
    case = _case([_WEATHER_TOOL], [],
                 {"name": "get_weather", "arguments": {"city": "北京"}})
    s = SPEC.l1_score(case, None)
    assert s["name_match"] == 0
    assert s["arg_f1"] == 0.0


# ─────────────────────────────────────────────────────────────
# aggregate
# ─────────────────────────────────────────────────────────────

def test_aggregate():
    agg = SPEC.aggregate_l1([
        {"name_match": 1, "arg_recall": 1.0, "arg_precision": 1.0, "arg_f1": 1.0},
        {"name_match": 0, "arg_recall": 0.0, "arg_precision": 0.0, "arg_f1": 0.0},
    ])
    assert agg["name_accuracy"] == pytest.approx(0.5)
    assert agg["arg_f1"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────
# spec shape + prompt
# ─────────────────────────────────────────────────────────────

def test_spec_shape():
    assert SPEC.name == "function_calling"
    assert not SPEC.requires_vlm
    assert "name_accuracy_min" in SPEC.default_thresholds
    assert "arg_f1_min" in SPEC.default_thresholds


def test_build_prompt_contains_tool_and_message():
    case = _case([_WEATHER_TOOL],
                 [{"role": "user", "content": "北京天气怎么样"}],
                 {"name": "get_weather", "arguments": {"city": "北京"}})
    prompt, image = SPEC.build_prompt(case)
    assert image is None
    assert "get_weather" in prompt
    assert "北京天气怎么样" in prompt
    assert "city" in prompt
    assert "JSON" in prompt


def test_build_prompt_multi_tool():
    case = _case([_WEATHER_TOOL, _SEARCH_TOOL],
                 [{"role": "user", "content": "搜索信息"}],
                 {"name": "search_web", "arguments": {"query": "test"}})
    prompt, _ = SPEC.build_prompt(case)
    assert "get_weather" in prompt
    assert "search_web" in prompt
