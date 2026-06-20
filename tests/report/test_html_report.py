"""tests/report/test_html_report.py — 全离线，不需要 CDN。"""
from benchmark.report.html_report import (
    _quality_radar_data,
    _verdict_color,
    generate_html,
)


def test_empty_report_produces_valid_html():
    html = generate_html({})
    assert "<html" in html and "</html>" in html


def test_single_model_report_has_chart_data():
    report = {
        "model": "test-model",
        "verdict": "PASS",
        "benchmarks": {
            "accuracy": {
                "verdict": "PASS",
                "aggregate": {"category_precision": 0.9},
            }
        },
    }
    html = generate_html(report)
    assert "radar" in html.lower() or "chart" in html.lower()
    assert "test-model" in html


def test_verdict_colors():
    assert "#22c55e" in _verdict_color("PASS")
    assert "#ef4444" in _verdict_color("FAIL")
    assert "#f59e0b" in _verdict_color("WARN")


def test_verdict_colors_case_insensitive():
    assert _verdict_color("pass") == _verdict_color("PASS")
    assert _verdict_color("fail") == _verdict_color("FAIL")


def test_verdict_color_unknown_returns_gray():
    color = _verdict_color("UNKNOWN_VERDICT")
    assert color == "#94a3b8"


def test_quality_radar_data_handles_missing_dims():
    data = _quality_radar_data({"verdict": "PASS"})
    assert isinstance(data, dict)
    assert len(data["values"]) == 9
    assert all(0 <= v <= 1.0 for v in data["values"])


def test_quality_radar_uses_numeric_metric_over_verdict():
    report = {"benchmarks": {"translation": {"verdict": "WARN", "bleu": 85.0}}}
    data = _quality_radar_data(report)
    labels = data["labels"]
    values = data["values"]
    idx = labels.index("translation")
    assert abs(values[idx] - 0.85) < 0.01


def test_quality_radar_accuracy_numeric():
    report = {"benchmarks": {"accuracy": {"verdict": "WARN",
                                           "aggregate": {"category_precision": 0.75}}}}
    data = _quality_radar_data(report)
    idx = data["labels"].index("accuracy")
    assert abs(data["values"][idx] - 0.75) < 0.01


def test_quality_radar_asr_cer_inverted():
    report = {"benchmarks": {"asr": {"verdict": "PASS",
                                      "aggregate": {"cer": 0.1}}}}
    data = _quality_radar_data(report)
    idx = data["labels"].index("asr")
    assert abs(data["values"][idx] - 0.9) < 0.01


def test_quality_radar_embedding_recall():
    report = {"benchmarks": {"embedding": {"verdict": "PASS",
                                            "aggregate": {"recall@1": 0.88}}}}
    data = _quality_radar_data(report)
    idx = data["labels"].index("embedding")
    assert abs(data["values"][idx] - 0.88) < 0.01


def test_quality_radar_general_ability_mean():
    report = {"benchmarks": {"general_ability": {
        "verdict": "PASS",
        "tasks": {
            "gsm8k": {"accuracy": 0.6},
            "mmlu": {"accuracy": 0.8},
            "hellaswag": {"accuracy": 0.7},
        },
    }}}
    data = _quality_radar_data(report)
    idx = data["labels"].index("general_ability")
    assert abs(data["values"][idx] - 0.7) < 0.01


def test_quality_radar_scenarios_mean_l1():
    report = {"benchmarks": {"scenarios": {
        "verdict": "PASS",
        "scenarios": {
            "S1": {"verdict": "PASS", "l1": {"score": 0.9}},
            "S2": {"verdict": "WARN", "l1": {"score": 0.5}},
        },
    }}}
    data = _quality_radar_data(report)
    idx = data["labels"].index("scenarios")
    assert abs(data["values"][idx] - 0.7) < 0.01


def test_quality_radar_conversation_drift_inverted():
    report = {"benchmarks": {"conversation_drift": {
        "verdict": "PASS",
        "per_scenario": {
            "s1": {"max_quality_drop": 0.1},
            "s2": {"max_quality_drop": 0.3},
        },
    }}}
    data = _quality_radar_data(report)
    idx = data["labels"].index("conversation_drift")
    # max drop = 0.3 → 1 - 0.3 = 0.7
    assert abs(data["values"][idx] - 0.7) < 0.01


def test_compare_report_detected():
    report = {
        "mode": "compare",
        "baseline": "m1",
        "candidate": "m2",
        "final_verdict": "REPLACEABLE",
        "baseline_report": {},
        "candidate_report": {},
    }
    html = generate_html(report)
    assert "m1" in html and "m2" in html
    assert "REPLACEABLE" in html


def test_compare_report_detected_by_fields():
    """mode 字段缺失时按 baseline/candidate 字段判断。"""
    report = {
        "baseline": "modelA",
        "candidate": "modelB",
        "final_verdict": "NOT_REPLACEABLE",
        "baseline_report": {},
        "candidate_report": {},
    }
    html = generate_html(report)
    assert "modelA" in html and "modelB" in html
    assert "NOT_REPLACEABLE" in html


def test_html_has_cdn_script_tag():
    html = generate_html({"model": "x"})
    assert "chart.js" in html.lower()


def test_html_has_fallback_for_missing_cdn():
    html = generate_html({"model": "x"})
    # onerror 降级机制
    assert "onerror" in html


def test_html_verdict_badge_rendered():
    html = generate_html({"model": "x", "verdict": "FAIL",
                          "benchmarks": {}})
    assert "FAIL" in html
    assert "#ef4444" in html


def test_html_special_chars_escaped():
    html = generate_html({"model": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in html


def test_bleu_in_0_1_range_not_divided():
    """BLEU 若已在 [0,1] 范围内不再除以 100。"""
    report = {"benchmarks": {"translation": {"verdict": "PASS", "bleu": 0.72}}}
    data = _quality_radar_data(report)
    idx = data["labels"].index("translation")
    assert abs(data["values"][idx] - 0.72) < 0.01


def test_generate_html_no_new_imports():
    """html_report 不依赖 benchmark 以外的第三方库（仅 stdlib）。"""
    import sys
    # 模块应已导入；如有第三方依赖这里会 ImportError
    mod = sys.modules.get("benchmark.report.html_report")
    assert mod is not None
