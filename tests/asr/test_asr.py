"""Tests for benchmark.asr.

CPU-only, no audio / ONNX model:
- CER / WER edit-distance metrics on known ref/hyp pairs (Chinese + English).
- RTF computation.
- transcript validation (empty output → FAIL).
- run_asr orchestrator via an injected transcriber + a tmp manifest,
  plus the graceful BLOCKED paths (no dataset / no backend).
"""
from __future__ import annotations

import json

import pytest

from benchmark.asr import metrics
from benchmark.asr.datasets import load_asr_manifest
from benchmark.asr.runner import run_asr


# --------------------------------------------------------------------------- #
# CER / WER / RTF
# --------------------------------------------------------------------------- #
def test_cer_perfect():
    assert metrics.cer("甚至出现交易几乎停滞的情况", "甚至出现交易几乎停滞的情况") == 0.0


def test_cer_one_substitution():
    # 8 chars, 1 wrong → CER 1/8 (punctuation normalized away)
    assert metrics.cer("今天天气很好啊。", "今天天气很坏啊") == pytest.approx(1 / 7)


def test_cer_handles_punctuation_and_case():
    assert metrics.cer("Hello, World!", "hello world") == 0.0


def test_wer_word_level():
    assert metrics.wer("the cat sat", "the cat sat") == 0.0
    assert metrics.wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_corpus_cer_aggregates_by_total_chars():
    refs = ["你好", "今天天气"]
    hyps = ["你好", "今天天器"]   # 1 char wrong out of 6 total
    assert metrics.corpus_cer(refs, hyps) == pytest.approx(1 / 6)


def test_rtf():
    assert metrics.rtf(0.5, 5.0) == pytest.approx(0.1)
    assert metrics.rtf(1.0, 0.0) == 0.0   # guard against zero-duration


def test_validate_transcript():
    assert metrics.validate_transcript("有内容")["ok"] is True
    assert metrics.validate_transcript("")["ok"] is False
    assert metrics.validate_transcript("   ")["ok"] is False


# --------------------------------------------------------------------------- #
# Manifest loader
# --------------------------------------------------------------------------- #
def test_load_asr_manifest(tmp_path):
    p = tmp_path / "m.jsonl"
    rows = [
        {"audio": "a.wav", "text": "你好", "duration": 1.0, "uid": "u1"},
        {"audio": "/abs/b.wav", "text": "再见", "uid": "u2"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    samples = load_asr_manifest(p, audio_root=tmp_path)
    assert len(samples) == 2
    assert samples[0].audio == tmp_path / "a.wav"      # relative resolved
    assert str(samples[1].audio) == "/abs/b.wav"        # absolute kept


def test_load_asr_manifest_missing_returns_empty(tmp_path):
    assert load_asr_manifest(tmp_path / "nope.jsonl") == []


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class _FakeModel:
    name = "fake-asr"
    hf_repo = "fake/asr"


def _write_manifest(tmp_path, rows):
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


def test_run_asr_perfect_with_injected_transcriber(tmp_path):
    rows = [
        {"audio": "a.wav", "text": "你好世界", "duration": 2.0},
        {"audio": "b.wav", "text": "今天天气很好", "duration": 3.0},
    ]
    manifest = _write_manifest(tmp_path, rows)
    gold = {str(tmp_path / "a.wav"): "你好世界", str(tmp_path / "b.wav"): "今天天气很好"}

    def transcribe(path):
        return gold[str(path)]

    res = run_asr(_FakeModel(), manifest_path=manifest, audio_root=tmp_path,
                  transcribe_fn=transcribe)
    assert res["status"] == "ok"
    assert res["verdict"] == "PASS"
    assert res["aggregate"]["cer"] == 0.0
    assert res["aggregate"]["rtf_mean"] < 1.0


def test_run_asr_high_cer_fails(tmp_path):
    rows = [{"audio": "a.wav", "text": "今天天气很好我们去散步", "duration": 2.0}]
    manifest = _write_manifest(tmp_path, rows)

    def transcribe(path):
        return "完全不同的内容输出"   # high CER

    res = run_asr(_FakeModel(), manifest_path=manifest, audio_root=tmp_path,
                  transcribe_fn=transcribe, thresholds={"cer_max": 0.15, "rtf_max": 1.0})
    assert res["verdict"] == "FAIL"
    assert any("CER" in r for r in res["verdict_reasons"])


def test_run_asr_empty_output_fails(tmp_path):
    rows = [{"audio": "a.wav", "text": "你好", "duration": 1.0}]
    manifest = _write_manifest(tmp_path, rows)
    res = run_asr(_FakeModel(), manifest_path=manifest, audio_root=tmp_path,
                  transcribe_fn=lambda p: "")
    assert res["verdict"] == "FAIL"
    assert any("empty" in r.lower() for r in res["verdict_reasons"])


def test_run_asr_no_dataset_blocked():
    res = run_asr(_FakeModel(), manifest_path=None)
    assert res["status"] == "blocked"
    assert res["verdict"] == "SKIP"


def test_run_asr_no_backend_blocked(tmp_path):
    rows = [{"audio": "a.wav", "text": "你好", "duration": 1.0}]
    manifest = _write_manifest(tmp_path, rows)
    # No transcribe_fn and no model_dir → no backend available → blocked.
    res = run_asr(_FakeModel(), manifest_path=manifest, audio_root=tmp_path)
    assert res["status"] == "blocked"
    assert "backend" in res["reason"]
