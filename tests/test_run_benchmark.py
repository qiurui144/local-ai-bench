"""run_benchmark.main() exit-code contract.

A benchmark harness must never report success when nothing was measured —
but ``--model all`` is documented as "run the models that are up", so a
down model in all-mode is a SKIP, not a FAIL. The contract:

- named model errors (e.g. not ready)        → exit 2 (you asked for it)
- all-mode, some models down, some measured  → verdict of the measured ones
- ZERO models produced any measurement       → exit 2 (empty run is not a PASS)
"""
import sys

import run_benchmark as rb


def _stub_model(name, *, port=9999, vram=1.0, rerank_native=False, capabilities=None):
    class _M:
        hf_repo = "org/stub"
        quantization = "fp16"
        hardware_min = "n/a"
    _M.name = name
    _M.port = port
    _M.vram_estimate_gb = vram
    _M.rerank_native = rerank_native
    _M.capabilities = capabilities
    return _M()


def _patch_pipeline(monkeypatch, tmp_path, results_by_name, argv_model="all"):
    """Route main() through stubs: fixed per-model results, tmp reports dir."""
    golden = tmp_path / "golden.json"
    golden.write_text("{}", encoding="utf-8")
    models = [_stub_model(n) for n in results_by_name]
    monkeypatch.setattr(rb, "load_models", lambda p: list(models))
    monkeypatch.setattr(rb, "load_benchmarks_config", lambda p: {})
    monkeypatch.setattr(
        rb, "run_all_for_model",
        lambda m, g, s, b: dict(results_by_name[m.name], model=m.name),
    )
    monkeypatch.setattr(rb, "REPORTS", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark.py", "--golden", str(golden), "--model", argv_model,
    ])


def test_named_model_not_ready_exits_2(monkeypatch, tmp_path):
    """Explicitly requested model errors → exit 2, never a silent 0."""
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {}, "error": "model_not_ready"},
    }, argv_model="m1")
    assert rb.main() == 2


def test_all_mode_zero_measurements_exits_2(monkeypatch, tmp_path):
    """all-mode with every model down = empty run; must NOT exit 0."""
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {}, "error": "model_not_ready"},
        "m2": {"benchmarks": {}, "error": "model_not_ready"},
    })
    assert rb.main() == 2


def test_all_mode_down_model_is_skip_not_fail(monkeypatch, tmp_path):
    """all-mode runs "the models that are up": one down + one PASS → exit 0."""
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {}, "error": "model_not_ready"},
        "m2": {"benchmarks": {"accuracy": {"verdict": "PASS"}}},
    })
    assert rb.main() == 0


def test_clean_pass_still_exits_zero(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {"accuracy": {"verdict": "PASS"}}},
    })
    assert rb.main() == 0


def test_dimension_fail_exits_2(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {"accuracy": {"verdict": "FAIL"}}},
    })
    assert rb.main() == 2


def test_dimension_warn_exits_1(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {"accuracy": {"verdict": "WARN"}}},
    })
    assert rb.main() == 1


def test_dimension_blocked_exits_1(monkeypatch, tmp_path):
    """维度级 BLOCKED(如 general_ability 数据全缺)= WARN 出码,绝不静默 0。"""
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {"accuracy": {"verdict": "BLOCKED"}}},
    })
    assert rb.main() == 1


def test_scenarios_verdict_feeds_exit_code(monkeypatch, tmp_path):
    """scenarios 维度的 FAIL 必须传导到 exit 2(进 verdict dims 元组)。"""
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {"scenarios": {"verdict": "FAIL"}}},
    })
    assert rb.main() == 2


def test_general_ability_verdict_feeds_exit_code(monkeypatch, tmp_path):
    """general_ability 是质量维度:FAIL 必须传导到 exit 2(D1 接线)。"""
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {"general_ability": {"verdict": "FAIL"}}},
    })
    assert rb.main() == 2


def test_run_all_dispatches_scenarios(monkeypatch, tmp_path):
    """run_all_for_model 在不 skip 时调用 scenarios 维度(stub 验证接线)。"""
    called = {}

    def fake_run_scenarios(model_cfg, *, judge_cfg, cfg):
        called["model"] = model_cfg.name
        called["judge"] = getattr(judge_cfg, "name", None)
        return {"verdict": "PASS"}

    monkeypatch.setattr(rb, "run_scenarios", fake_run_scenarios)
    monkeypatch.setattr(rb, "wait_model_ready", lambda *a, **kw: True)
    monkeypatch.setattr(rb, "get_vram_info", lambda *a, **kw: {})
    m = _stub_model("m1")
    skip = {"accuracy", "ttft", "throughput", "prefill_decode", "concurrency",
            "stability", "translation", "embedding", "rerank", "asr",
            "general_ability", "conditioned"}
    result = rb.run_all_for_model(m, {}, skip, {"scenarios": {"judge_model": None}})
    assert result["benchmarks"]["scenarios"]["verdict"] == "PASS"
    assert called["model"] == "m1"


_NON_SCENARIO_SKIP = {
    "accuracy", "ttft", "throughput", "prefill_decode", "concurrency",
    "stability", "translation", "embedding", "rerank", "asr",
    "general_ability", "conditioned",
}


def test_run_all_skips_scenarios_for_non_chat_models(monkeypatch):
    """embedding/rerank/asr 服务端点没有可用 /chat/completions —
    scenarios 必须不 dispatch（否则 error_rate=1.0 → 健康部署假 FAIL）。"""
    monkeypatch.setattr(rb, "wait_model_ready", lambda *a, **kw: True)
    monkeypatch.setattr(rb, "get_vram_info", lambda *a, **kw: {})
    monkeypatch.setattr(
        rb, "run_scenarios",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    # Use capabilities field (consumed by common._is_chat_capable) instead of _model_hint
    for cap in ("embedding", "rerank", "asr"):
        m = _stub_model(f"stub-{cap}", capabilities=(cap,))
        result = rb.run_all_for_model(m, {}, set(_NON_SCENARIO_SKIP), {})
        assert "scenarios" not in result["benchmarks"], cap
    # rerank_native flag alone (no hint) must also exclude
    monkeypatch.setattr(rb, "_model_hint", lambda m, key: False)
    m = _stub_model("stub-native-rr", rerank_native=True)
    result = rb.run_all_for_model(m, {}, set(_NON_SCENARIO_SKIP), {})
    assert "scenarios" not in result["benchmarks"]


def test_run_all_dispatches_scenarios_for_plain_chat_model(monkeypatch):
    """普通 chat 模型（无任何 *_capable hint）仍要跑 scenarios。"""
    monkeypatch.setattr(rb, "wait_model_ready", lambda *a, **kw: True)
    monkeypatch.setattr(rb, "get_vram_info", lambda *a, **kw: {})
    monkeypatch.setattr(rb, "_model_hint", lambda m, key: False)
    monkeypatch.setattr(rb, "_resolve_judge", lambda name, m: None)
    monkeypatch.setattr(
        rb, "run_scenarios",
        lambda model_cfg, *, judge_cfg, cfg: {"verdict": "PASS"},
    )
    m = _stub_model("plain-chat")
    result = rb.run_all_for_model(m, {}, set(_NON_SCENARIO_SKIP), {})
    assert result["benchmarks"]["scenarios"]["verdict"] == "PASS"


# ─── render_markdown scenarios section ───

def _result_with_scenarios():
    return {
        "model": "m1",
        "benchmarks": {
            "scenarios": {
                "verdict": "WARN",
                "judge_model": "qwen3-30b",
                "judge_calibration": {"anchor_agreement": 0.83, "passed": True},
                "scenarios": {
                    "case_logic": {
                        "verdict": "WARN",
                        "l1": {"label_accuracy": 0.72, "finding_f1": 0.55},
                        "l2_judge": {"mean": 0.61, "std": 0.04, "seeds": 3},
                        "provenance": "builtin synthetic",
                        "verdict_reasons": ["finding_f1 0.55 near threshold"],
                    },
                },
                "verdict_reasons": ["S2 scored on builtin synthetic fallback"],
            },
        },
    }


def test_render_markdown_includes_scenarios_section():
    md = rb.render_markdown(_result_with_scenarios())
    assert "真实场景" in md
    assert "case_logic" in md
    assert "WARN" in md
    assert "finding_f1 0.55 near threshold" in md
    assert "S2 scored on builtin synthetic fallback" in md
    assert "builtin synthetic" in md          # provenance
    assert "0.61 ± 0.04 (N=3)" in md          # L2 judge
    assert "agreement 0.83 (PASS)" in md      # judge calibration


def test_render_markdown_without_scenarios_has_no_section():
    md = rb.render_markdown({"model": "m1", "benchmarks": {}})
    assert "真实场景" not in md


# ─── _resolve_judge quality + liveness ───

def _judge_pool(monkeypatch, *, ready=True):
    models = [
        _stub_model("dut", vram=20),
        _stub_model("embed-huge", vram=999, capabilities=("embedding",)),  # never a judge
        _stub_model("chat-big", vram=240),
        _stub_model("chat-small", vram=35),
        _stub_model("native-rr", vram=1, rerank_native=True),
        _stub_model("no-port", vram=500, port=0),
    ]
    monkeypatch.setattr(rb, "load_models", lambda p: list(models))
    monkeypatch.setattr(
        rb, "wait_model_ready", lambda m, timeout_s=10.0: ready)
    return models


def test_resolve_judge_fallback_picks_largest_text_model(monkeypatch):
    models = _judge_pool(monkeypatch)
    judge = rb._resolve_judge(None, models[0])
    assert judge is not None and judge.name == "chat-big"


def test_resolve_judge_endpoint_not_ready_returns_none(monkeypatch):
    models = _judge_pool(monkeypatch, ready=False)
    assert rb._resolve_judge(None, models[0]) is None


def test_resolve_judge_prefers_ready_same_target_candidate(monkeypatch):
    dut = _stub_model("dut-amd", vram=20)
    dut.target = "amd-win-x86"
    global_vl = _stub_model("qwen2.5-vl-7b-fp16", port=8002, vram=18)
    same_target = _stub_model("qwen2.5-7b-amd-win", vram=10)
    same_target.target = "amd-win-x86"
    other_target = _stub_model("qwen2.5-7b-intel-win", vram=10)
    other_target.target = "intel-win-x86"
    models = [dut, global_vl, same_target, other_target]

    monkeypatch.setattr(rb, "load_models", lambda p: list(models))
    monkeypatch.setattr(rb, "wait_model_ready", lambda m, timeout_s=2.0: m is same_target)

    judge = rb._resolve_judge(None, dut)
    assert judge is same_target


def test_resolve_judge_named_not_found_returns_none(monkeypatch):
    models = _judge_pool(monkeypatch)
    assert rb._resolve_judge("no-such-model", models[0]) is None


# ─── multi-seed (--seeds N) wiring ───

def _merged_report(tmp_path, model="m1"):
    """Load {model}_{ts}.json, skipping the per-seed raw archives."""
    import json

    path = next(p for p in tmp_path.glob(f"{model}_*.json")
                if "_seed" not in p.stem)
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_multi_seed(monkeypatch, tmp_path, run_all_fn, *, seeds, model="m1"):
    """Like _patch_pipeline but with a custom run_all_for_model + --seeds N."""
    golden = tmp_path / "golden.json"
    golden.write_text("{}", encoding="utf-8")
    models = [_stub_model(model)]
    monkeypatch.setattr(rb, "load_models", lambda p: list(models))
    monkeypatch.setattr(rb, "load_benchmarks_config", lambda p: {})
    monkeypatch.setattr(rb, "run_all_for_model", run_all_fn)
    monkeypatch.setattr(rb, "REPORTS", tmp_path)
    argv = ["run_benchmark.py", "--golden", str(golden), "--model", model]
    if seeds is not None:
        argv += ["--seeds", str(seeds)]
    monkeypatch.setattr(sys, "argv", argv)


def test_seeds_3_runs_model_3_times_and_aggregates(monkeypatch, tmp_path):
    """--seeds 3 → run_all_for_model called 3×; report carries multi_seed
    with n_seeds==3 and correct mean/std for a per-call-varying metric."""
    calls = {"n": 0}

    def fake_run_all(m, g, s, b):
        i = calls["n"]
        calls["n"] += 1
        return {
            "model": m.name,
            "benchmarks": {"accuracy": {
                "verdict": "PASS",
                "aggregate": {"category_precision": 0.5 + 0.1 * i},
            }},
        }

    _patch_multi_seed(monkeypatch, tmp_path, fake_run_all, seeds=3)
    assert rb.main() == 0
    assert calls["n"] == 3
    ms = _merged_report(tmp_path)["multi_seed"]
    assert ms["n_seeds"] == 3
    stat = ms["metrics"]["accuracy.aggregate.category_precision"]
    assert abs(stat["mean"] - 0.6) < 1e-9
    assert abs(stat["std"] - 0.1) < 1e-9          # ddof=1 over [0.5, 0.6, 0.7]
    assert stat["ci95_lower"] <= stat["mean"] <= stat["ci95_upper"]


def test_seeds_worst_verdict_wins(monkeypatch, tmp_path):
    """Verdicts are never averaged: seeds [PASS, FAIL, PASS] → exit 2."""
    verdicts = iter(["PASS", "FAIL", "PASS"])

    def fake_run_all(m, g, s, b):
        return {"model": m.name,
                "benchmarks": {"accuracy": {"verdict": next(verdicts)}}}

    _patch_multi_seed(monkeypatch, tmp_path, fake_run_all, seeds=3)
    assert rb.main() == 2


def test_default_single_seed_has_no_multi_seed_key(monkeypatch, tmp_path):
    """Regression: default --seeds 1 keeps the report shape unchanged."""
    def fake_run_all(m, g, s, b):
        return {"model": m.name,
                "benchmarks": {"accuracy": {"verdict": "PASS"}}}

    _patch_multi_seed(monkeypatch, tmp_path, fake_run_all, seeds=None)
    assert rb.main() == 0
    assert "multi_seed" not in _merged_report(tmp_path)
    # single-seed runs must not emit per-seed archives either
    assert not list(tmp_path.glob("m1_*_seed*.json"))


def test_dimension_wrappers_honor_benchmark_config(monkeypatch, tmp_path):
    class Ctx:
        fixtures = tmp_path

    calls = {}

    monkeypatch.setattr(
        rb,
        "run_ttft",
        lambda m, fixtures, *, samples: calls.setdefault("ttft", (fixtures, samples)),
    )
    monkeypatch.setattr(
        rb,
        "run_throughput",
        lambda m, fixtures, *, duration_s: calls.setdefault("throughput", (fixtures, duration_s)),
    )
    monkeypatch.setattr(
        rb,
        "run_prefill_decode",
        lambda m, fixtures, *, samples, decode_tokens: calls.setdefault(
            "prefill_decode", (fixtures, samples, decode_tokens)
        ),
    )
    monkeypatch.setattr(
        rb,
        "run_concurrency",
        lambda m, fixtures, *, concurrencies, duration_s: calls.setdefault(
            "concurrency", (fixtures, concurrencies, duration_s)
        ),
    )
    monkeypatch.setattr(
        rb,
        "run_stability",
        lambda m, fixtures, *, duration_s, sample_interval_s: calls.setdefault(
            "stability", (fixtures, duration_s, sample_interval_s)
        ),
    )

    m = _stub_model("m")
    rb._run_ttft_dim(m, {"samples": 2}, Ctx())
    rb._run_throughput_dim(m, {"duration_s": 3}, Ctx())
    rb._run_prefill_decode_dim(m, {"samples": 4, "decode_tokens": 32}, Ctx())
    rb._run_concurrency_dim(m, {"concurrencies": [2, "4"], "duration_s": 5}, Ctx())
    rb._run_stability_dim(m, {"duration_s": 6, "sample_interval_s": 1.5}, Ctx())

    assert calls["ttft"] == (tmp_path, 2)
    assert calls["throughput"] == (tmp_path, 3.0)
    assert calls["prefill_decode"] == (tmp_path, 4, 32)
    assert calls["concurrency"] == (tmp_path, [2, 4], 5.0)
    assert calls["stability"] == (tmp_path, 6.0, 1.5)


def test_seeds_per_seed_raw_archived_and_fail_traceable(monkeypatch, tmp_path):
    """I1: a FAIL in seed 1 must not vanish from the archive — every seed's
    raw JSON is saved as {stem}_seed{k}.json and the merged report's
    multi_seed.per_seed records which seed failed and why."""
    import json

    verdicts = iter(["PASS", "FAIL", "PASS"])

    def fake_run_all(m, g, s, b):
        return {"model": m.name,
                "benchmarks": {"accuracy": {
                    "verdict": next(verdicts),
                    "aggregate": {"category_precision": 0.5},
                }}}

    _patch_multi_seed(monkeypatch, tmp_path, fake_run_all, seeds=3)
    assert rb.main() == 2
    # all three raw archives exist; seed 1 carries the FAIL
    for k in range(3):
        assert list(tmp_path.glob(f"m1_*_seed{k}.json")), f"seed{k} raw missing"
    seed1 = json.loads(next(tmp_path.glob("m1_*_seed1.json"))
                       .read_text(encoding="utf-8"))
    assert seed1["benchmarks"]["accuracy"]["verdict"] == "FAIL"
    # seed-0 raw archive stays raw (no multi_seed block injected)
    seed0 = json.loads(next(tmp_path.glob("m1_*_seed0.json"))
                       .read_text(encoding="utf-8"))
    assert "multi_seed" not in seed0
    # merged report records WHICH seed failed
    per_seed = _merged_report(tmp_path)["multi_seed"]["per_seed"]
    assert per_seed[1]["seed"] == 1
    assert per_seed[1]["verdict_summary"]["accuracy"] == "FAIL"
    assert per_seed[0]["verdict_summary"]["accuracy"] == "PASS"
    assert per_seed[1]["error"] is None


def test_seeds_per_seed_entries_carry_duration(monkeypatch, tmp_path):
    """A-I2: per_seed entries carry a real (numeric, >= 0) duration_s."""
    def fake_run_all(m, g, s, b):
        return {"model": m.name,
                "benchmarks": {"accuracy": {"verdict": "PASS"}}}

    _patch_multi_seed(monkeypatch, tmp_path, fake_run_all, seeds=2)
    rb.main()
    per_seed = _merged_report(tmp_path)["multi_seed"]["per_seed"]
    assert len(per_seed) == 2
    for entry in per_seed:
        assert isinstance(entry["duration_s"], float)
        assert entry["duration_s"] >= 0.0


def test_seeds_empty_intersection_warns(monkeypatch, tmp_path):
    """I2: one seed with empty benchmarks empties the metric intersection —
    the merged report must say so, not silently render an empty section."""
    runs = iter([
        {"benchmarks": {"accuracy": {
            "verdict": "PASS", "aggregate": {"category_precision": 0.5}}}},
        {"benchmarks": {"accuracy": {
            "verdict": "PASS", "aggregate": {"category_precision": 0.6}}}},
        {"benchmarks": {}},
    ])

    def fake_run_all(m, g, s, b):
        return dict(next(runs), model=m.name)

    _patch_multi_seed(monkeypatch, tmp_path, fake_run_all, seeds=3)
    rb.main()
    ms = _merged_report(tmp_path)["multi_seed"]
    assert ms["metrics"] == {}
    assert "metric intersection empty" in ms["warning"]
    md = next(p for p in tmp_path.glob("m1_*.md")).read_text(encoding="utf-8")
    assert "metric intersection empty" in md


def test_seeds_integer_counters_dropped_from_metrics(monkeypatch, tmp_path):
    """I3: int-valued counters (n_cases &co) aggregate meaninglessly and crowd
    the capped markdown table — only fractional quality metrics survive."""
    calls = {"n": 0}

    def fake_run_all(m, g, s, b):
        i = calls["n"]
        calls["n"] += 1
        return {"model": m.name,
                "benchmarks": {"accuracy": {
                    "verdict": "PASS",
                    "aggregate": {"n_cases": 15,
                                  "category_precision": 0.5 + 0.1 * i},
                }}}

    _patch_multi_seed(monkeypatch, tmp_path, fake_run_all, seeds=3)
    assert rb.main() == 0
    ms = _merged_report(tmp_path)["multi_seed"]
    assert "accuracy.aggregate.category_precision" in ms["metrics"]
    assert "accuracy.aggregate.n_cases" not in ms["metrics"]
    assert "warning" not in ms


def test_render_markdown_multi_seed_section():
    md = rb.render_markdown({
        "model": "m1", "benchmarks": {},
        "multi_seed": {"n_seeds": 3, "metrics": {
            "accuracy.aggregate.category_precision": {
                "mean": 0.6, "std": 0.1,
                "ci95_lower": 0.35, "ci95_upper": 0.85},
        }},
    })
    assert "## Multi-seed (N=3)" in md
    assert "accuracy.aggregate.category_precision" in md
    assert "0.6000 ± 0.1000" in md


def test_seeds_stable_unit_quality_metric_survives_aggregation():
    """FIX-5: 三个 seed 全 1.0 的 needle_recall 是质量指标不是计数器 —
    整数启发式只丢「全精确整数且 max>1」的叶子(n_cases=15 仍丢)。"""
    seed_runs = [
        {"benchmarks": {"accuracy": {
            "verdict": "PASS",
            "aggregate": {"needle_recall": 1.0, "n_cases": 15},
        }}}
        for _ in range(3)
    ]
    ms = rb.aggregate_multi_seed(seed_runs)
    assert "accuracy.aggregate.needle_recall" in ms["metrics"]
    assert ms["metrics"]["accuracy.aggregate.needle_recall"]["mean"] == 1.0
    assert "accuracy.aggregate.n_cases" not in ms["metrics"]
    assert "warning" not in ms


# ─── --skip key validation (FIX-5) ───

def test_skip_unknown_key_exits_2(monkeypatch, tmp_path, caplog):
    """--skip stabilty (typo) 必须 exit 2 并提示可选维度,不能静默全跑。"""
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {"accuracy": {"verdict": "PASS"}}},
    }, argv_model="m1")
    monkeypatch.setattr(sys, "argv", sys.argv + ["--skip", "stabilty"])
    assert rb.main() == 2
    assert "stabilty" in caplog.text
    assert "stability" in caplog.text          # 提示里列出合法 key


def test_skip_valid_key_still_runs(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, tmp_path, {
        "m1": {"benchmarks": {"accuracy": {"verdict": "PASS"}}},
    }, argv_model="m1")
    monkeypatch.setattr(sys, "argv", sys.argv + ["--skip", "stability,ttft"])
    assert rb.main() == 0


def test_remote_all_continues_after_model_failure(monkeypatch):
    import common
    from common import TargetConfig
    from benchmark.executor import remote as remote_mod

    class _Model:
        def __init__(self, name):
            self.name = name
            self.target = "win"

    class _Exec:
        def __init__(self, target):
            self.target = target
            self.last_error = None

        def run_benchmark(self, model_name, extra_args=(), install_first=False,
                          raise_on_error=True):
            calls.append(model_name)
            self.last_error = RuntimeError("boom") if model_name == "bad" else None

    calls = []
    target = TargetConfig(
        name="win",
        platform="windows",
        arch="x86_64",
        connection="ssh",
        runtime="ollama",
        ip_env=None,
        ssh_user_env=None,
        ssh_pass_env=None,
    )
    monkeypatch.setattr(common, "load_targets", lambda: {"win": target})
    monkeypatch.setattr(rb, "load_models", lambda p: [_Model("bad"), _Model("good")])
    monkeypatch.setattr(remote_mod, "RemoteExecutor", _Exec)
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark.py", "--target", "win", "--model", "all",
    ])
    assert rb.main() == 2
    assert calls == ["bad", "good"]


def test_remote_single_model_quality_failure_does_not_abort_controller(monkeypatch):
    import common
    from common import TargetConfig
    from benchmark.executor import remote as remote_mod

    calls = []

    class _Exec:
        def __init__(self, target):
            self.target = target
            self.last_error = RuntimeError("remote benchmark exited with code 2")

        def run_benchmark(self, model_name, extra_args=(), install_first=False,
                          raise_on_error=True):
            calls.append((model_name, raise_on_error))

    target = TargetConfig(
        name="win",
        platform="windows",
        arch="x86_64",
        connection="ssh",
        runtime="ollama",
        ip_env=None,
        ssh_user_env=None,
        ssh_pass_env=None,
    )
    monkeypatch.setattr(common, "load_targets", lambda: {"win": target})
    monkeypatch.setattr(remote_mod, "RemoteExecutor", _Exec)
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark.py", "--target", "win", "--model", "bad",
    ])
    assert rb.main() == 0
    assert calls == [("bad", True)]


# ─── SCHEMA_VERSION single-sourced (FIX-5) ───

def test_schema_version_single_source():
    from benchmark import compare as bc
    from benchmark.registry import SCHEMA_VERSION
    assert rb.SCHEMA_VERSION == SCHEMA_VERSION
    assert bc.SCHEMA_VERSION is SCHEMA_VERSION


# ─── models.yaml scenarios thresholds block ───

def test_models_yaml_scenarios_thresholds_load():
    import pathlib

    from common import load_benchmarks_config
    cfg = load_benchmarks_config(
        pathlib.Path(rb.__file__).parent / "models.yaml")
    sc = cfg["scenarios"]
    assert sc["num_cases"] == 50
    th = sc["thresholds"]
    assert th["wechat_intent"]["intent_accuracy_min"] == 0.70
    assert th["case_logic"]["label_accuracy_min"] == 0.60
    assert th["article_knowledge"]["claim_accuracy_min"] == 0.60


def test_models_yaml_per_model_benchmark_overrides_load():
    import pathlib

    from common import load_models

    models = load_models(pathlib.Path(rb.__file__).parent / "models.yaml")
    amd = next(m for m in models if m.name == "llama3.2-3b-amd-win")
    assert amd.benchmarks["throughput"]["thresholds"]["tps_min"] == 20
    assert amd.benchmarks["ttft"]["thresholds"]["p95_ttft_ms_max"] == 2500


def test_run_all_merges_model_benchmark_overrides(monkeypatch):
    captured = {}

    def run_dim(model, cfg, ctx):
        captured.update(cfg)
        return {"verdict": "PASS"}

    spec = rb.DimensionSpec("accuracy", quality=True, run=run_dim)
    monkeypatch.setattr(rb, "DIMENSIONS", {"accuracy": spec})
    monkeypatch.setattr(rb, "get_hardware_profile", lambda m: {})
    monkeypatch.setattr(rb, "get_vram_info", lambda: {})
    monkeypatch.setattr(rb, "harness_version", lambda: "test")

    model = rb.ModelConfig(
        name="edge",
        port=0,
        benchmarks={"accuracy": {"thresholds": {"min_score": 0.5}, "samples": 3}},
    )
    result = rb.run_all_for_model(
        model,
        {},
        set(),
        {"accuracy": {"thresholds": {"min_score": 0.9, "other": 1.0}}},
    )

    assert captured["thresholds"] == {"min_score": 0.5, "other": 1.0}
    assert captured["samples"] == 3
    assert result["benchmark_config"]["accuracy"]["thresholds"]["min_score"] == 0.5


def test_run_all_honors_model_configured_skip(monkeypatch):
    called = []

    def run_dim(model, cfg, ctx):
        called.append("accuracy")
        return {"verdict": "PASS"}

    spec = rb.DimensionSpec("accuracy", quality=True, run=run_dim)
    monkeypatch.setattr(rb, "DIMENSIONS", {"accuracy": spec})
    monkeypatch.setattr(rb, "get_hardware_profile", lambda m: {})
    monkeypatch.setattr(rb, "get_vram_info", lambda: {})
    monkeypatch.setattr(rb, "harness_version", lambda: "test")

    model = rb.ModelConfig(name="edge", port=0, benchmarks={"skip": ["accuracy"]})
    result = rb.run_all_for_model(model, {}, set(), {"accuracy": {}})
    assert called == []
    assert result["benchmark_config"]["skip"] == ["accuracy"]


def test_remote_all_dispatch_filters_models_by_target(monkeypatch, tmp_path):
    golden = tmp_path / "golden.json"
    golden.write_text("{}", encoding="utf-8")
    models = [
        rb.ModelConfig(name="amd-model", target="amd-win-x86"),
        rb.ModelConfig(name="intel-model", target="intel-win-x86"),
        rb.ModelConfig(name="local-model"),
    ]
    calls = []

    class _Target:
        name = "intel-win-x86"

        def is_local(self):
            return False

    class _Remote:
        def __init__(self, target):
            self.target = target
            self.last_error = None

        def run_benchmark(self, model, extra_args, install_first=False,
                          raise_on_error=True):
            calls.append((model, list(extra_args), install_first))
            self.last_error = None

    import common
    monkeypatch.setattr(common, "load_targets", lambda: {"intel-win-x86": _Target()})
    monkeypatch.setattr(rb, "load_models", lambda p: list(models))
    import benchmark.executor.remote as remote
    monkeypatch.setattr(remote, "RemoteExecutor", _Remote)
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark.py", "--golden", str(golden),
        "--target", "intel-win-x86", "--model", "all", "--install-first",
    ])

    assert rb.main() == 0
    assert calls == [("intel-model", ["--target", "intel-win-x86"], True)]


# ─── --compare wiring (Task 8) ───

def test_compare_flag_delegates(monkeypatch, tmp_path):
    import benchmark.compare as bc
    monkeypatch.setattr(bc, "run_compare", lambda b, c, d, cfg: 2)
    monkeypatch.setattr(rb, "load_benchmarks_config", lambda p: {})
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--compare", "a", "b"])
    assert rb.main() == 2
