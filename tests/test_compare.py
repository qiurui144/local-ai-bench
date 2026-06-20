"""D2 --compare replaceability verdict (offline, fixture reports only).

2σ discipline is hard-coded: single-seed input caps at INCONCLUSIVE;
harness_version / schema_version / condition mismatch refuses comparison;
hardware_profile mismatch forces the performance side INCONCLUSIVE.
"""
import json

from benchmark.compare import compare_reports, run_compare

BENCH_CFG = {"throughput": {"thresholds": {"tps_min": 30}}}


def _report(model, acc, *, hv="abc123", schema=1, tps=50.0, cond=None, hw="A"):
    r = {"model": model, "harness_version": hv,
         "condition": cond or {"context_tokens": None, "cache_mode": None},
         "hardware_profile": {"gpu": hw, "driver": "d", "cuda": "c",
                              "vllm": "v", "hostname_hash": "h"},
         "benchmarks": {
             "accuracy": {"verdict": "PASS", "aggregate": {"category_precision": acc}},
             "throughput": {"aggregate_tps": tps},
         }}
    if schema is not None:
        r["schema_version"] = schema
    return r


def _group(tmp_path, model, accs, **kw):
    seeds = [_report(model, a, **kw) for a in accs]
    stem = f"{model}_20260611_000000"
    (tmp_path / f"{stem}.json").write_text(json.dumps(seeds[0]), encoding="utf-8")
    for k, s in enumerate(seeds):
        (tmp_path / f"{stem}_seed{k}.json").write_text(json.dumps(s), encoding="utf-8")
    return {"merged": seeds[0], "seeds": seeds, "path": str(tmp_path / f"{stem}.json")}


def test_equivalent_multiseed_replaceable(tmp_path):
    base = _group(tmp_path, "a", [0.80, 0.81, 0.79])
    cand = _group(tmp_path, "b", [0.80, 0.82, 0.80])
    out = compare_reports(base, cand, BENCH_CFG)
    assert out["verdict"] == "REPLACEABLE"
    key = "accuracy.aggregate.category_precision"
    assert out["quality"][key]["significant"] is False


def test_significant_regression_not_replaceable(tmp_path):
    base = _group(tmp_path, "a", [0.90, 0.90, 0.90])
    cand = _group(tmp_path, "b", [0.50, 0.51, 0.50])
    out = compare_reports(base, cand, BENCH_CFG)
    assert out["verdict"] == "NOT_REPLACEABLE"


def test_single_seed_capped_inconclusive(tmp_path):
    base = _group(tmp_path, "a", [0.90])
    cand = _group(tmp_path, "b", [0.50])     # 看着回退也只能 INCONCLUSIVE
    out = compare_reports(base, cand, BENCH_CFG)
    assert out["verdict"] == "INCONCLUSIVE"
    assert any("seed" in r for r in out["reasons"])


def test_harness_version_mismatch_inconclusive(tmp_path):
    out = compare_reports(_group(tmp_path, "a", [0.8] * 3),
                          _group(tmp_path, "b", [0.8] * 3, hv="zzz"), BENCH_CFG)
    assert out["verdict"] == "INCONCLUSIVE"


def test_legacy_v0_report_inconclusive(tmp_path):
    out = compare_reports(_group(tmp_path, "a", [0.8] * 3, schema=None),
                          _group(tmp_path, "b", [0.8] * 3), BENCH_CFG)
    assert out["verdict"] == "INCONCLUSIVE"


def test_cross_hardware_perf_inconclusive(tmp_path):
    out = compare_reports(_group(tmp_path, "a", [0.8] * 3, hw="A100"),
                          _group(tmp_path, "b", [0.8] * 3, hw="RK3588"), BENCH_CFG)
    assert out["performance"]["candidate_thresholds"] == "INCONCLUSIVE"
    assert out["verdict"] == "INCONCLUSIVE"


def test_candidate_perf_below_threshold_not_replaceable(tmp_path):
    out = compare_reports(_group(tmp_path, "a", [0.8] * 3),
                          _group(tmp_path, "b", [0.8] * 3, tps=10.0), BENCH_CFG)
    assert out["verdict"] == "NOT_REPLACEABLE"


def test_candidate_report_benchmark_config_overrides_global_threshold(tmp_path):
    base = _group(tmp_path, "a", [0.8] * 3)
    cand = _group(tmp_path, "b", [0.8] * 3, tps=10.0)
    cand["merged"]["benchmark_config"] = {
        "throughput": {"thresholds": {"tps_min": 8}}
    }
    for seed in cand["seeds"]:
        seed["benchmark_config"] = cand["merged"]["benchmark_config"]

    out = compare_reports(base, cand, BENCH_CFG)
    assert out["performance"]["candidate_thresholds"] == "PASS"
    assert out["verdict"] == "REPLACEABLE"


def test_condition_mismatch_inconclusive(tmp_path):
    """D2: context_tokens 4096 vs None — 不同 condition 的报告不可比。"""
    out = compare_reports(
        _group(tmp_path, "a", [0.8] * 3,
               cond={"context_tokens": 4096, "cache_mode": None}),
        _group(tmp_path, "b", [0.8] * 3), BENCH_CFG)
    assert out["verdict"] == "INCONCLUSIVE"
    assert any("condition" in r for r in out["reasons"])


def test_load_group_no_prefix_collision(tmp_path):
    """FIX-1: 'qwen3' 不得吞掉 qwen3_mini_*.json(文件名前缀碰撞)。"""
    from benchmark.compare import load_group
    _group(tmp_path, "qwen3_mini", [0.8] * 3)
    assert load_group(tmp_path, "qwen3") is None
    g = load_group(tmp_path, "qwen3_mini")
    assert g is not None and g["merged"]["model"] == "qwen3_mini"


def test_load_group_model_field_mismatch_is_not_found(tmp_path):
    """FIX-1: 文件名匹配但报告 model 字段不符 → 视为无报告。"""
    import json as _json
    from benchmark.compare import load_group
    (tmp_path / "qwen3_20260611_000000.json").write_text(
        _json.dumps(_report("other-model", 0.8)), encoding="utf-8")
    assert load_group(tmp_path, "qwen3") is None


def test_run_compare_prefix_collision_inconclusive(tmp_path):
    """FIX-1 端到端: 请求 qwen3 但只有 qwen3_mini 报告 → INCONCLUSIVE(exit 1),
    绝不能拿错模型报告得出 verdict。"""
    _group(tmp_path, "qwen3_mini", [0.8] * 3)
    _group(tmp_path, "b", [0.8] * 3)
    assert run_compare("qwen3", "b", tmp_path, BENCH_CFG) == 1
    out = json.loads(next(tmp_path.glob("compare_qwen3_vs_b_*.json"))
                     .read_text(encoding="utf-8"))
    assert out["verdict"] == "INCONCLUSIVE"
    assert any("no report found" in r for r in out["reasons"])


def test_run_compare_exit_codes_and_artifacts(tmp_path, monkeypatch):
    _group(tmp_path, "a", [0.8, 0.8, 0.8])
    _group(tmp_path, "b", [0.8, 0.8, 0.8])
    code = run_compare("a", "b", tmp_path, BENCH_CFG)
    assert code == 0
    assert list(tmp_path.glob("compare_a_vs_b_*.json"))
    assert list(tmp_path.glob("compare_a_vs_b_*.md"))
    assert run_compare("a", "missing-model", tmp_path, BENCH_CFG) == 1
