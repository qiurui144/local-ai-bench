import benchmark.conditioned.runner as cr
from common import InferResult

PARAS = ["某甲与某乙因合同纠纷诉至法院,经审理查明相关事实。" * 8] * 30


def _ok(content, ttft=100.0, in_tok=1000):
    return InferResult(model="m", ok=True, content=content, ttft_ms=ttft,
                       input_tokens=in_tok, output_tokens=8, latency_ms=200.0,
                       tokens_per_sec=40.0)


def _patch_corpus(monkeypatch):
    monkeypatch.setattr(cr, "load_cail_paragraphs", lambda **kw: PARAS)


class _Cfg:
    name = "m1"
    hf_repo = "org/m1"
    port = 8123
    max_model_len = None
    base_url = "http://localhost:8123/v1"


def test_ladder_happy_path_all_rungs(monkeypatch):
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(cr, "_effective_max_len", lambda m: 40960)
    rows = cr.load_needles(tmp_path_root() / "datasets/conditioned/needles.jsonl")

    def fake_infer(model_cfg, *, prompt, **kw):
        # 全知 stub:按 prompt 末尾的问题句精确匹配 golden(缓存 A/B 的概括题
        # 不含 "问题:" → 落到 "不知道",两次一致 → cache OK)
        question = prompt.rsplit("问题:", 1)[-1].strip()
        for r in rows:
            if r["question"] == question:
                return _ok(f"答案是 {r['answer']}")
        return _ok("不知道")

    monkeypatch.setattr(cr, "infer_sync", fake_infer)
    monkeypatch.setattr(cr, "infer_stream", fake_infer)
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    ladder = out["context_ladder"]
    assert set(ladder) == {"1k", "4k"}
    assert ladder["1k"]["needle_recall"] == 1.0
    assert out["cache"]["output_consistent"] is True
    assert out["verdict"] == "PASS"


def tmp_path_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]   # 真仓根(读 shipped needles.jsonl)


def test_rung_exceeding_max_len_skipped_not_fail(monkeypatch):
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(cr, "_effective_max_len", lambda m: 2048)
    monkeypatch.setattr(cr, "infer_sync", lambda m, *, prompt, **kw: _ok("答案是 41"))
    monkeypatch.setattr(cr, "infer_stream", lambda m, *, prompt, **kw: _ok("答案是 41"))
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 8192]}, root=tmp_path_root())
    assert out["context_ladder"]["8k"]["verdict"] == "SKIPPED"
    assert out["verdict"] != "FAIL"


def test_all_rungs_skipped_blocked(monkeypatch):
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(cr, "_effective_max_len", lambda m: 512)
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [4096, 8192]}, root=tmp_path_root())
    assert out["verdict"] == "BLOCKED"


def test_cache_inconsistency_fails(monkeypatch):
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(cr, "_effective_max_len", lambda m: 40960)
    replies = iter(["第一次回答", "第二次不同回答"])
    monkeypatch.setattr(cr, "infer_sync", lambda m, *, prompt, **kw: _ok("答案是 41"))
    monkeypatch.setattr(cr, "infer_stream",
                        lambda m, *, prompt, **kw: _ok(next(replies, "x")))
    out = cr.run_conditioned(_Cfg(), {"context_ladder": []}, root=tmp_path_root())
    assert out["cache"]["output_consistent"] is False
    assert out["verdict"] == "FAIL"


def test_rung_error_recorded_run_continues(monkeypatch):
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(cr, "_effective_max_len", lambda m: 40960)
    calls = {"n": 0}

    def flaky(model_cfg, *, prompt, **kw):
        calls["n"] += 1
        return InferResult(model="m", ok=False, error="HTTP 500")

    monkeypatch.setattr(cr, "infer_sync", flaky)
    monkeypatch.setattr(cr, "infer_stream", flaky)
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]}, root=tmp_path_root())
    assert out["context_ladder"]["1k"]["errors"] == 8
    assert "4k" in out["context_ladder"]           # 没有崩整跑


def _err(model_cfg, *, prompt, **kw):
    return InferResult(model="m", ok=False, error="HTTP 500")


def _stub_ladder(monkeypatch, acc_by_target, needle_by_target=None, cache=None):
    """直接 stub _run_rung/_run_cache_ab,精确控制边界值。"""
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(cr, "_effective_max_len", lambda m: 10**9)

    def fake_rung(model_cfg, target, probes, paragraphs, max_tokens):
        return {"task_accuracy": acc_by_target[target],
                "needle_recall": (needle_by_target or {}).get(target, 1.0),
                "ttft_ms": 1.0, "tps": 1.0, "prompt_tokens_target": target,
                "prompt_tokens_actual": target,
                "tokens_estimation": "approx(1.6 chars/token)",
                "errors": 0, "n": 8}

    monkeypatch.setattr(cr, "_run_rung", fake_rung)
    monkeypatch.setattr(cr, "_run_cache_ab", lambda m, p, t: dict(cache) if cache else {
        "ttft_cold_ms": 100.0, "ttft_warm_ms": 50.0, "speedup": 2.0,
        "output_consistent": True})


# ─── FIX-2a: 空跑绝不 PASS ───

def test_empty_ladder_and_cache_error_blocked(monkeypatch):
    """空 context_ladder + 缓存 A/B 全错 = 零测量 → BLOCKED,绝不 PASS。"""
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(cr, "_effective_max_len", lambda m: 40960)
    monkeypatch.setattr(cr, "infer_sync", _err)
    monkeypatch.setattr(cr, "infer_stream", _err)
    out = cr.run_conditioned(_Cfg(), {"context_ladder": []}, root=tmp_path_root())
    assert out["verdict"] == "BLOCKED"
    assert any("cache" in r for r in out["verdict_reasons"])


# ─── FIX-2b: 缓存 A/B 全错 → 一致性不可证,verdict 封顶 WARN ───

def test_cache_both_error_caps_pass_to_warn(monkeypatch):
    _stub_ladder(monkeypatch, {1024: 1.0, 4096: 1.0},
                 cache={"error": "HTTP 500"})
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    assert out["verdict"] == "WARN"
    assert any("unverifiable" in r for r in out["verdict_reasons"])


def test_cache_error_does_not_demote_fail(monkeypatch):
    _stub_ladder(monkeypatch, {1024: 1.0, 4096: 0.64},     # drop 0.36 → FAIL
                 cache={"error": "HTTP 500"})
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    assert out["verdict"] == "FAIL"


# ─── FIX-4: D4 quality-drop 阈值边界(drop > 0.20 WARN / > 0.35 FAIL,严格大于) ───

def test_quality_drop_below_warn_threshold_passes(monkeypatch):
    _stub_ladder(monkeypatch, {1024: 1.0, 4096: 0.81})     # drop 0.19
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    assert out["quality_degradation"]["drop"] == 0.19
    assert out["verdict"] == "PASS"
    assert not any("quality drop" in r for r in out["verdict_reasons"])


def test_quality_drop_above_warn_threshold_warns(monkeypatch):
    _stub_ladder(monkeypatch, {1024: 1.0, 4096: 0.79})     # drop 0.21
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    assert out["verdict"] == "WARN"
    assert any("quality drop" in r for r in out["verdict_reasons"])


def test_quality_drop_above_fail_threshold_fails(monkeypatch):
    _stub_ladder(monkeypatch, {1024: 1.0, 4096: 0.64})     # drop 0.36
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    assert out["verdict"] == "FAIL"


# ─── FIX-3: needle 信号独立于 quality-drop,WARN 不吞、FAIL 不降 ───

def test_needle_recall_low_warns_with_reason(monkeypatch):
    _stub_ladder(monkeypatch, {1024: 1.0, 4096: 1.0},
                 needle_by_target={1024: 1.0, 4096: 0.4})
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    assert out["verdict"] == "WARN"
    assert any("needle_recall" in r for r in out["verdict_reasons"])


def test_needle_reason_survives_prior_quality_warn(monkeypatch):
    """两个独立信号:quality-drop 已 WARN 时 needle 理由仍必须出现。"""
    _stub_ladder(monkeypatch, {1024: 1.0, 4096: 0.79},     # drop 0.21 → WARN
                 needle_by_target={1024: 1.0, 4096: 0.4})
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    assert out["verdict"] == "WARN"
    assert any("needle_recall" in r for r in out["verdict_reasons"])
    assert any("quality drop" in r for r in out["verdict_reasons"])


def test_needle_never_demotes_fail(monkeypatch):
    _stub_ladder(monkeypatch, {1024: 1.0, 4096: 0.64},     # drop 0.36 → FAIL
                 needle_by_target={1024: 1.0, 4096: 0.4})
    out = cr.run_conditioned(_Cfg(), {"context_ladder": [1024, 4096]},
                             root=tmp_path_root())
    assert out["verdict"] == "FAIL"
    assert any("needle_recall" in r for r in out["verdict_reasons"])


def test_corpus_load_failure_blocked(monkeypatch):
    monkeypatch.setattr(cr, "load_cail_paragraphs",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("offline")))
    out = cr.run_conditioned(_Cfg(), {}, root=tmp_path_root())
    assert out["verdict"] == "BLOCKED"
