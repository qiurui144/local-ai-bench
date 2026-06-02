"""Tests for benchmark.performance.run_prefill_decode (PP/TG separation).

CPU-only, no endpoint — infer_stream is monkeypatched to return deterministic
InferResults so the PP/TG tok/s math is verified against hand-computed values.
"""
from __future__ import annotations

import pytest

from common import InferResult
from benchmark import performance


class _FakeModel:
    name = "fake-llm"
    hf_repo = "fake/llm"
    is_vlm = False


def test_pp_tg_math(monkeypatch, tmp_path):
    # prompt 100 tok, TTFT 200 ms (prefill), total 1200 ms → decode 1000 ms,
    # 50 completion tok → PP = 100 / 0.2 = 500 t/s ; TG = 50 / 1.0 = 50 t/s.
    def fake_stream(model_cfg, **kw):
        return InferResult(
            model="fake", ok=True, content="x",
            input_tokens=100, output_tokens=50,
            latency_ms=1200.0, ttft_ms=200.0,
        )

    monkeypatch.setattr(performance, "infer_stream", fake_stream)
    res = performance.run_prefill_decode(_FakeModel(), tmp_path, samples=3, decode_tokens=128)

    assert res["measured"] == 3
    assert res["prefill"]["tok_per_sec"]["p50"] == pytest.approx(500.0)
    assert res["decode"]["tok_per_sec"]["p50"] == pytest.approx(50.0)
    assert res["prefill"]["avg_prompt_tokens"] == pytest.approx(100.0)
    assert res["decode"]["avg_decode_tokens"] == pytest.approx(50.0)


def test_pp_tg_skips_samples_without_usage(monkeypatch, tmp_path):
    # No usage / no TTFT → cannot split → counted as no_usage, not measured.
    def fake_stream(model_cfg, **kw):
        return InferResult(model="fake", ok=True, content="x",
                           input_tokens=0, output_tokens=0,
                           latency_ms=500.0, ttft_ms=0.0)

    monkeypatch.setattr(performance, "infer_stream", fake_stream)
    res = performance.run_prefill_decode(_FakeModel(), tmp_path, samples=2)
    assert res["measured"] == 0
    assert res["no_usage_samples"] == 2


def test_pp_tg_counts_errors(monkeypatch, tmp_path):
    def fake_stream(model_cfg, **kw):
        return InferResult(model="fake", ok=False, error="HTTP 500")

    monkeypatch.setattr(performance, "infer_stream", fake_stream)
    res = performance.run_prefill_decode(_FakeModel(), tmp_path, samples=2)
    assert res["errors"] == 2
    assert res["measured"] == 0
