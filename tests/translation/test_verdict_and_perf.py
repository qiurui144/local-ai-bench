"""Verdict mapping + per-direction performance runner (coverage backfill).

1. ``accuracy.run_translation`` verdict block: stubbed scorers drive the
   threshold → PASS/FAIL mapping (BLEU 20.0 / chrF 40.0 / term-match 0.80
   defaults), incl. the COMET-unavailable skip path (must NOT fail the run)
   and the L3 terminology gate.
2. ``performance.run_translation_ttft`` / ``run_translation_throughput`` /
   ``run_translation_performance``: stubbed ``infer_stream`` / ``infer_sync``
   plus a fake clock (same idiom as tests/performance/test_throughput_stats.py)
   verify per-direction keys, TTFT stats, throughput numbers and error paths.

Backfill pins current behavior; the nominal-vs-elapsed TPS divergence has been
fixed in benchmark/translation/performance.py (elapsed wall time + elapsed_s).
"""
from __future__ import annotations

import sys
import types

import pytest

from benchmark.translation import accuracy
from benchmark.translation import performance as tperf
from benchmark.translation.accuracy import COMET_UNAVAILABLE
from benchmark.translation.datasets import TranslationPair

import common


class _Model:
    name = "stub-model"


def _pairs(n=2, glossary=None, src_lang="zh", tgt_lang="en"):
    return [
        TranslationPair(
            src=f"源文{i}",
            ref=f"reference {i}",
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            source="custom",
            glossary=dict(glossary) if glossary else {},
        )
        for i in range(n)
    ]


def _stub_scorers(monkeypatch, *, bleu, chrf, comet=None,
                  hyps=None):
    """Stub the scoring functions run_translation resolves via module globals."""
    monkeypatch.setattr(
        accuracy, "translate_batch",
        lambda cfg, srcs, sl, tl, **kw: list(hyps) if hyps is not None
        else [f"hyp {i}" for i in range(len(srcs))],
    )
    monkeypatch.setattr(accuracy, "compute_sacrebleu", lambda h, r, t: bleu)
    monkeypatch.setattr(accuracy, "compute_chrf", lambda h, r: chrf)
    monkeypatch.setattr(
        accuracy, "compute_comet",
        lambda h, s, r: comet or {"available": True, "score": 0.85,
                                  "model": "stub-comet"},
    )


# --------------------------------------------------------------------------- #
# run_translation verdict block
# --------------------------------------------------------------------------- #
def test_no_pairs_returns_skipped():
    out = accuracy.run_translation(_Model(), [])
    assert out["skipped"] is True
    assert out["reason"] == "no pairs"
    assert out["benchmark"] == "translation"


def test_verdict_pass_at_exact_thresholds(monkeypatch):
    # thresholds are strict '<': bleu == bleu_min and chrf == chrf_min PASS
    _stub_scorers(monkeypatch, bleu=20.0, chrf=40.0)
    out = accuracy.run_translation(_Model(), _pairs())
    assert out["verdict"] == "PASS"
    assert out["verdict_reasons"] == []
    agg = out["aggregate"]
    assert agg["bleu"] == 20.0 and agg["chrf"] == 40.0
    assert agg["num_pairs"] == 2 and agg["empty_rate"] == 0
    assert agg["src_lang"] == "zh" and agg["tgt_lang"] == "en"
    assert agg["data_source"] == "custom"
    assert agg["terminology"] is None  # l1: no term gate
    assert len(out["per_pair"]) == 2


def test_verdict_fail_bleu_just_below(monkeypatch):
    _stub_scorers(monkeypatch, bleu=19.9, chrf=55.0)
    out = accuracy.run_translation(_Model(), _pairs())
    assert out["verdict"] == "FAIL"
    assert out["verdict_reasons"] == ["FAIL: BLEU 19.9 < 20.0"]


def test_verdict_fail_chrf_just_below(monkeypatch):
    _stub_scorers(monkeypatch, bleu=30.0, chrf=39.9)
    out = accuracy.run_translation(_Model(), _pairs())
    assert out["verdict"] == "FAIL"
    assert out["verdict_reasons"] == ["FAIL: chrF 39.9 < 40.0"]


def test_verdict_fail_collects_both_metric_reasons(monkeypatch):
    _stub_scorers(monkeypatch, bleu=5.0, chrf=10.0)
    out = accuracy.run_translation(_Model(), _pairs())
    assert out["verdict"] == "FAIL"
    assert len(out["verdict_reasons"]) == 2


def test_custom_thresholds_override_defaults(monkeypatch):
    _stub_scorers(monkeypatch, bleu=15.0, chrf=35.0)
    out = accuracy.run_translation(
        _Model(), _pairs(), thresholds={"bleu_min": 10.0, "chrf_min": 30.0}
    )
    assert out["verdict"] == "PASS"


def test_comet_unavailable_is_skipped_not_fail(monkeypatch):
    _stub_scorers(monkeypatch, bleu=30.0, chrf=50.0,
                  comet={"available": False, "reason": COMET_UNAVAILABLE})
    out = accuracy.run_translation(_Model(), _pairs())
    assert out["verdict"] == "PASS"
    assert out["aggregate"]["comet"] == {"available": False,
                                         "reason": COMET_UNAVAILABLE}


def test_run_comet_false_skips_comet_entirely(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("compute_comet must not be called")

    _stub_scorers(monkeypatch, bleu=30.0, chrf=50.0)
    monkeypatch.setattr(accuracy, "compute_comet", boom)
    out = accuracy.run_translation(_Model(), _pairs(), run_comet=False)
    assert out["aggregate"]["comet"] == {"available": False,
                                         "reason": "disabled"}
    assert out["verdict"] == "PASS"


def test_l3_term_match_at_threshold_passes(monkeypatch):
    # 4 of 5 glossary terms present → 0.80, strict '<' → PASS
    glossary = {"a": "alpha", "b": "beta", "c": "gamma",
                "d": "delta", "e": "epsilon"}
    hyp = "alpha beta gamma delta"
    _stub_scorers(monkeypatch, bleu=30.0, chrf=50.0, hyps=[hyp])
    out = accuracy.run_translation(
        _Model(), _pairs(n=1, glossary=glossary), level="l3", run_comet=False
    )
    assert out["aggregate"]["terminology"]["term_match_rate"] == pytest.approx(0.8)
    assert out["verdict"] == "PASS"


def test_l3_term_match_below_threshold_fails(monkeypatch):
    glossary = {"a": "alpha", "b": "beta", "c": "gamma",
                "d": "delta", "e": "epsilon"}
    hyp = "alpha beta gamma"  # 3/5 = 0.6
    _stub_scorers(monkeypatch, bleu=30.0, chrf=50.0, hyps=[hyp])
    out = accuracy.run_translation(
        _Model(), _pairs(n=1, glossary=glossary), level="l3", run_comet=False
    )
    assert out["verdict"] == "FAIL"
    assert "FAIL: term-match 60% < 80%" in out["verdict_reasons"]


def test_l1_ignores_glossary_term_gate(monkeypatch):
    # same failing glossary, but level=l1 → terminology not evaluated
    glossary = {"a": "alpha", "b": "beta"}
    _stub_scorers(monkeypatch, bleu=30.0, chrf=50.0, hyps=["no terms here"])
    out = accuracy.run_translation(
        _Model(), _pairs(n=1, glossary=glossary), level="l1", run_comet=False
    )
    assert out["aggregate"]["terminology"] is None
    assert out["verdict"] == "PASS"


def test_all_empty_hyps_fails_numerical_validation(monkeypatch):
    _stub_scorers(monkeypatch, bleu=30.0, chrf=50.0, hyps=["", ""])
    out = accuracy.run_translation(_Model(), _pairs(), run_comet=False)
    assert out["verdict"] == "FAIL"
    assert any("numerical validation" in r for r in out["verdict_reasons"])
    assert out["aggregate"]["empty_rate"] == pytest.approx(1.0)
    assert out["aggregate"]["nonempty_hyps"] == 0


def test_nan_metric_fails_validation(monkeypatch):
    _stub_scorers(monkeypatch, bleu=float("nan"), chrf=50.0)
    out = accuracy.run_translation(_Model(), _pairs(), run_comet=False)
    assert out["verdict"] == "FAIL"
    assert any("NaN/Inf" in r for r in out["verdict_reasons"])


def test_per_pair_truncates_to_60_chars(monkeypatch):
    long_hyp = "x" * 200
    _stub_scorers(monkeypatch, bleu=30.0, chrf=50.0, hyps=[long_hyp])
    pair = TranslationPair(src="s" * 200, ref="r" * 200,
                           src_lang="zh", tgt_lang="en", source="custom")
    out = accuracy.run_translation(_Model(), [pair], run_comet=False)
    pp = out["per_pair"][0]
    assert len(pp["src"]) == 60 and len(pp["ref"]) == 60
    assert len(pp["hyp"]) == 60 and pp["source"] == "custom"


# --------------------------------------------------------------------------- #
# translate_batch / translate_passage (stubbed infer_sync)
# --------------------------------------------------------------------------- #
def _sync_content(ok=True, content=""):
    return common.InferResult(model="stub", ok=ok, content=content)


def test_translate_batch_strips_and_blanks_failures(monkeypatch):
    results = iter([_sync_content(content="  hello  "),
                    _sync_content(ok=False)])
    monkeypatch.setattr(accuracy, "infer_sync", lambda cfg, **kw: next(results))
    hyps = accuracy.translate_batch(_Model(), ["a", "b"], "zh", "en")
    assert hyps == ["hello", ""]


def test_translate_batch_l3_uses_glossary_prompt(monkeypatch):
    prompts_seen = []

    def fake(cfg, *, prompt, **kw):
        prompts_seen.append(prompt)
        return _sync_content(content="ok")

    monkeypatch.setattr(accuracy, "infer_sync", fake)
    accuracy.translate_batch(_Model(), ["源"], "zh", "en",
                             level="l3", glossary={"向量化": "vectorization"})
    assert "向量化 -> vectorization" in prompts_seen[0]


def test_translate_passage_strips_numbering_and_pads(monkeypatch):
    monkeypatch.setattr(
        accuracy, "infer_sync",
        lambda cfg, **kw: _sync_content(content="1. Hello\n2) World\n"),
    )
    out = accuracy.translate_passage(_Model(), ["a", "b", "c"], "en", "zh")
    assert out == ["Hello", "World", ""]


def test_translate_passage_truncates_extra_lines(monkeypatch):
    monkeypatch.setattr(
        accuracy, "infer_sync",
        lambda cfg, **kw: _sync_content(content="1. x\n2. y\n3. z\n"),
    )
    assert accuracy.translate_passage(_Model(), ["a", "b"], "en", "zh") == ["x", "y"]


def test_translate_passage_failure_returns_empties(monkeypatch):
    monkeypatch.setattr(accuracy, "infer_sync",
                        lambda cfg, **kw: _sync_content(ok=False))
    assert accuracy.translate_passage(_Model(), ["a", "b"], "en", "zh") == ["", ""]


# --------------------------------------------------------------------------- #
# pure-Python fallback scorers (sacrebleu import forced to fail)
# --------------------------------------------------------------------------- #
def test_fallback_bleu_and_chrf_without_sacrebleu(monkeypatch):
    # sys.modules[name] = None makes `import sacrebleu` raise ImportError
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    bleu = accuracy.compute_sacrebleu(["the cat sat"], ["the cat sat"], "en")
    assert 0.0 < bleu <= 100.0
    bleu_zh = accuracy.compute_sacrebleu(["今天天气"], ["今天天气"], "zh")
    assert 0.0 < bleu_zh <= 100.0
    chrf = accuracy.compute_chrf(["hello world"], ["hello world"])
    assert chrf == pytest.approx(100.0)
    assert accuracy.compute_chrf(["zzz"], ["abc"]) == pytest.approx(0.0)


def test_fallback_bleu_brevity_penalty(monkeypatch):
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    short = accuracy.compute_sacrebleu(["the cat"], ["the cat sat on the mat"], "en")
    full = accuracy.compute_sacrebleu(
        ["the cat sat on the mat"], ["the cat sat on the mat"], "en")
    assert short < full


# --------------------------------------------------------------------------- #
# COMET import paths (fake torch / comet modules)
# --------------------------------------------------------------------------- #
def _fake_torch(has_gpu):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: has_gpu))


def test_comet_no_gpu_returns_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setitem(sys.modules, "comet", types.SimpleNamespace(
        download_model=lambda m: pytest.fail("must not download without GPU"),
        load_from_checkpoint=lambda c: None,
    ))
    out = accuracy.compute_comet(["h"], ["s"], ["r"])
    assert out == {"available": False, "reason": COMET_UNAVAILABLE}


def test_comet_gpu_path_scores(monkeypatch):
    model = types.SimpleNamespace(
        predict=lambda data, gpus, progress_bar: {"system_score": 0.91})
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    monkeypatch.setitem(sys.modules, "comet", types.SimpleNamespace(
        download_model=lambda m: "ckpt",
        load_from_checkpoint=lambda c: model,
    ))
    out = accuracy.compute_comet(["h"], ["s"], ["r"])
    assert out["available"] is True
    assert out["score"] == pytest.approx(0.91)


def test_comet_gpu_probe_failure_treated_as_no_gpu(monkeypatch):
    def cuda_boom():
        raise RuntimeError("driver mismatch")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=cuda_boom)))
    monkeypatch.setitem(sys.modules, "comet", types.SimpleNamespace(
        download_model=lambda m: "ckpt", load_from_checkpoint=lambda c: None))
    out = accuracy.compute_comet(["h"], ["s"], ["r"])
    assert out == {"available": False, "reason": COMET_UNAVAILABLE}


def test_comet_exception_never_raises(monkeypatch):
    def boom(m):
        raise RuntimeError("checkpoint download failed")

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    monkeypatch.setitem(sys.modules, "comet", types.SimpleNamespace(
        download_model=boom, load_from_checkpoint=lambda c: None))
    out = accuracy.compute_comet(["h"], ["s"], ["r"])
    assert out == {"available": False, "reason": COMET_UNAVAILABLE}


# --------------------------------------------------------------------------- #
# performance: fake clock + stubbed infer (idiom of test_throughput_stats.py)
# --------------------------------------------------------------------------- #
class _FakeTime:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def _stream_result(ok=True, ttft_ms=100.0, latency_ms=500.0):
    return common.InferResult(model="stub", ok=ok, ttft_ms=ttft_ms,
                              latency_ms=latency_ms, content="hi")


def test_ttft_skipped_on_empty_pairs():
    out = tperf.run_translation_ttft(_Model(), [])
    assert out["skipped"] is True
    assert out["benchmark"] == "translation_ttft"


def test_ttft_stats_and_lang_pair(monkeypatch):
    ttfts = iter([100.0, 200.0, 300.0, 400.0, 500.0])

    def fake_stream(model_cfg, **kw):
        t = next(ttfts)
        return _stream_result(ttft_ms=t, latency_ms=t + 1000.0)

    monkeypatch.setattr(tperf, "infer_stream", fake_stream)
    out = tperf.run_translation_ttft(_Model(), _pairs(n=2), samples=5)

    assert out["lang_pair"] == "zh->en"
    assert out["samples"] == 5
    assert out["errors"] == 0 and out["error_rate"] == 0
    assert out["ttft_ms_stats"]["count"] == 5
    assert out["ttft_ms_stats"]["p50"] == pytest.approx(300.0)
    assert out["ttft_ms_stats"]["min"] == 100.0
    assert out["ttft_ms_stats"]["max"] == 500.0
    assert out["total_latency_ms_stats"]["p50"] == pytest.approx(1300.0)


def test_ttft_cycles_through_pairs(monkeypatch):
    seen = []

    def fake_stream(model_cfg, *, prompt, **kw):
        seen.append(prompt)
        return _stream_result()

    monkeypatch.setattr(tperf, "infer_stream", fake_stream)
    tperf.run_translation_ttft(_Model(), _pairs(n=2), samples=4)
    # pairs cycle i % len(pairs): 0,1,0,1
    assert len(seen) == 4
    assert seen[0] == seen[2] and seen[1] == seen[3]
    assert seen[0] != seen[1]


def test_ttft_counts_failures_and_zero_ttft_as_errors(monkeypatch):
    results = iter([
        _stream_result(ok=True, ttft_ms=100.0),
        _stream_result(ok=False, ttft_ms=0.0),       # failed call
        _stream_result(ok=True, ttft_ms=0.0),        # ok but no first token
        _stream_result(ok=True, ttft_ms=300.0),
    ])
    monkeypatch.setattr(tperf, "infer_stream", lambda cfg, **kw: next(results))
    out = tperf.run_translation_ttft(_Model(), _pairs(n=1), samples=4)
    assert out["errors"] == 2
    assert out["error_rate"] == pytest.approx(0.5)
    assert out["ttft_ms_stats"]["count"] == 2


def test_throughput_skipped_on_empty_pairs():
    out = tperf.run_translation_throughput(_Model(), [])
    assert out["skipped"] is True
    assert out["benchmark"] == "translation_throughput"


def _sync_result(ok=True, output_tokens=30, input_tokens=10,
                 latency_ms=3000.0, tps=10.0):
    return common.InferResult(model="stub", ok=ok, output_tokens=output_tokens,
                              input_tokens=input_tokens, latency_ms=latency_ms,
                              tokens_per_sec=tps)


def _patch_throughput(monkeypatch, clock, result_fn):
    monkeypatch.setattr(tperf, "time", clock)

    def fake_sync(model_cfg, **kw):
        clock.t += 3.0
        return result_fn()

    monkeypatch.setattr(tperf, "infer_sync", fake_sync)


def test_throughput_counts_and_token_totals(monkeypatch):
    """4 requests x 3s fill a 10s window (deadline overshoot to t=12)."""
    clock = _FakeTime()
    _patch_throughput(monkeypatch, clock, _sync_result)
    out = tperf.run_translation_throughput(_Model(), _pairs(n=2),
                                           duration_s=10.0)
    assert out["requests"] == 4
    assert out["errors"] == 0
    assert out["total_output_tokens"] == 120
    assert out["total_input_tokens"] == 40
    assert out["lang_pair"] == "zh->en"
    assert out["latency_stats_ms"]["count"] == 4
    assert out["latency_stats_ms"]["p50"] == pytest.approx(3000.0)
    assert out["per_request_tps_stats"]["p50"] == pytest.approx(10.0)
    assert out["per_request_tps_stats"]["p95"] == pytest.approx(10.0)


def test_throughput_tps_should_use_actual_elapsed(monkeypatch):
    clock = _FakeTime()
    _patch_throughput(monkeypatch, clock, _sync_result)
    out = tperf.run_translation_throughput(_Model(), _pairs(n=2),
                                           duration_s=10.0)
    assert out["aggregate_tps"] == pytest.approx(120 / 12.0)


def test_throughput_all_errors(monkeypatch):
    clock = _FakeTime()
    _patch_throughput(monkeypatch, clock,
                      lambda: _sync_result(ok=False, output_tokens=0, tps=0.0))
    out = tperf.run_translation_throughput(_Model(), _pairs(n=1),
                                           duration_s=10.0)
    assert out["requests"] == 4
    assert out["errors"] == 4
    assert out["total_output_tokens"] == 0
    assert out["aggregate_tps"] == 0.0
    assert out["per_request_tps_stats"] == {"p50": 0, "p95": 0}
    assert out["latency_stats_ms"]["count"] == 0


def test_throughput_nonpositive_tps_excluded_from_per_request_stats(monkeypatch):
    clock = _FakeTime()
    _patch_throughput(monkeypatch, clock, lambda: _sync_result(tps=0.0))
    out = tperf.run_translation_throughput(_Model(), _pairs(n=1),
                                           duration_s=10.0)
    # ok requests still accumulate tokens, but tps<=0 never enters the stats
    assert out["total_output_tokens"] == 120
    assert out["per_request_tps_stats"] == {"p50": 0, "p95": 0}


# --------------------------------------------------------------------------- #
# run_translation_performance: per-direction orchestration
# --------------------------------------------------------------------------- #
def test_performance_runner_per_direction_keys(monkeypatch):
    calls = {"ttft": [], "tp": []}

    def fake_ttft(cfg, pairs, *, samples):
        calls["ttft"].append((pairs[0].src_lang, samples))
        return {"benchmark": "translation_ttft"}

    def fake_tp(cfg, pairs, *, duration_s):
        calls["tp"].append((pairs[0].src_lang, duration_s))
        return {"benchmark": "translation_throughput"}

    monkeypatch.setattr(tperf, "run_translation_ttft", fake_ttft)
    monkeypatch.setattr(tperf, "run_translation_throughput", fake_tp)

    out = tperf.run_translation_performance(
        _Model(),
        {"zh->en": _pairs(n=1),
         "en->zh": _pairs(n=1, src_lang="en", tgt_lang="zh")},
        ttft_samples=3,
        throughput_duration_s=5.0,
    )
    assert out["benchmark"] == "translation_performance"
    assert set(out["directions"]) == {"zh->en", "en->zh"}
    for block in out["directions"].values():
        assert set(block) == {"ttft", "throughput"}
    assert calls["ttft"] == [("zh", 3), ("en", 3)]
    assert calls["tp"] == [("zh", 5.0), ("en", 5.0)]


def test_performance_runner_skip_set(monkeypatch):
    monkeypatch.setattr(tperf, "run_translation_ttft",
                        lambda cfg, pairs, *, samples: {"ok": 1})
    monkeypatch.setattr(
        tperf, "run_translation_throughput",
        lambda cfg, pairs, *, duration_s: pytest.fail("must be skipped"),
    )
    out = tperf.run_translation_performance(
        _Model(), {"zh->en": _pairs(n=1)}, skip={"throughput"}
    )
    assert set(out["directions"]["zh->en"]) == {"ttft"}


def test_performance_runner_end_to_end_with_stubs(monkeypatch):
    """Cover the real sub-runners through the orchestrator (no stubs of them)."""
    clock = _FakeTime()
    monkeypatch.setattr(tperf, "time", clock)
    monkeypatch.setattr(tperf, "infer_stream",
                        lambda cfg, **kw: _stream_result())

    def fake_sync(model_cfg, **kw):
        clock.t += 3.0
        return _sync_result()

    monkeypatch.setattr(tperf, "infer_sync", fake_sync)
    out = tperf.run_translation_performance(
        _Model(), {"zh->en": _pairs(n=1)},
        ttft_samples=2, throughput_duration_s=6.0,
    )
    block = out["directions"]["zh->en"]
    assert block["ttft"]["ttft_ms_stats"]["count"] == 2
    assert block["throughput"]["requests"] == 2
    assert block["throughput"]["total_output_tokens"] == 60
