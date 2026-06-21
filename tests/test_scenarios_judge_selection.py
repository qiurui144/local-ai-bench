"""Tests for scenarios judge model selection logic."""
import pytest

from benchmark.scenarios.judge import select_judge_model


class _M:
    def __init__(self, name, vram=0):
        self.name = name
        self.vram_estimate_gb = vram


def test_prefer_7b_judge():
    available = [_M("qwen2.5-1.5b-k3"), _M("qwen2.5-7b-amd"), _M("bge-embedding")]
    judge = select_judge_model(available)
    assert "7b" in judge.name.lower()


def test_fallback_small_model_when_no_7b():
    available = [_M("qwen3-0.6b-amd"), _M("bge-m3")]
    judge = select_judge_model(available)
    assert judge.name == "qwen3-0.6b-amd"


def test_no_models_raises():
    with pytest.raises(RuntimeError, match="No judge model"):
        select_judge_model([])


def test_returns_first_available_when_no_priority_match():
    available = [_M("unknown-model-xyz")]
    judge = select_judge_model(available)
    assert judge.name == "unknown-model-xyz"


def test_prefers_larger_when_multiple_priority_match():
    """When two 7B models match, prefer the higher-vram one (better quality)."""
    available = [_M("qwen2.5-7b-k3", vram=4.5), _M("qwen2.5-7b-amd", vram=5.0)]
    judge = select_judge_model(available)
    assert judge.name == "qwen2.5-7b-amd"
