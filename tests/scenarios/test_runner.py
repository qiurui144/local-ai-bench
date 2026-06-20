import json

import common
from benchmark.scenarios import runner as runner_mod


class _VLM:
    name = "vlm-under-test"
    hf_repo = "org/vlm"
    is_vlm = True

    @property
    def base_url(self):
        return "http://localhost:9999/v1"


class _Judge(_VLM):
    name = "judge-model"


class _TextModel(_VLM):
    name = "text-model"
    is_vlm = False


def _ok(d):
    return common.InferResult(model="m", ok=True, content=json.dumps(d), parsed_json=d)


def _ok_text(text):
    return common.InferResult(model="m", ok=True, content=text, parsed_json=None)


def _patch_perfect_run(monkeypatch, tmp_path):
    """八场景各 1 case 全对 + 有判别力的 judge(校准通过)的理想环境。"""
    def fake_infer(model_cfg, *, prompt, image_path=None, **kw):
        if "意图" in prompt:
            return _ok({"intent": "loan_agreement", "entities": ["张三", "5000元"]})
        if "陈述段落" in prompt:
            return _ok({"consistency": "contradictory",
                        "findings": [{"kind": "fact_mismatch", "segments": [1, 2]}]})
        if "格式要求" in prompt:
            return _ok_text("TESTOK result here")
        if "稳健性测试" in prompt:        # adversarial_stability (S8)
            return _ok_text("TESTOK result here")
        if "提取指定字段" in prompt:
            return _ok({"field_a": "value_a"})
        if "可用函数" in prompt:
            return _ok({"name": "tool_x", "arguments": {}})
        if "观察图片" in prompt:          # vlm_document_extraction (S7)
            return _ok({"field_a": "value_a"})
        return _ok({"claims": [{"claim": "维生素C可以治愈感冒", "label": "inaccurate"}],
                    "grade": "D"})

    monkeypatch.setattr(runner_mod, "infer_sync", fake_infer)

    def fake_judge_infer(model_cfg, *, prompt, seed=None, **kw):
        # 锚定题里有故意的坏输出 — 有判别力的 judge 必须打低分,
        # 否则全量(好/坏配对)校准会正确地把"永远满分"判为失败。
        bad_markers = ('"negotiation"',
                       '"consistency": "consistent"',
                       '"label": "accurate"')
        score = 1 if any(m in prompt for m in bad_markers) else 5
        return _ok({"score": score, "rationale": "r"})

    monkeypatch.setattr(runner_mod.judge_mod, "infer_sync", fake_judge_infer)

    base = tmp_path / "datasets" / "scenarios"
    rows = {
        "wechat_intent": {"id": "c1", "provenance": "curated", "payload": {
            "image": "fixtures/scenarios/wechat_intent/c1.png",
            "expected_intent": "loan_agreement",
            "expected_entities": ["张三", "5000元"]}},
        "case_logic": {"id": "c1", "provenance": "curated", "payload": {
            "segments": ["1月1日借款5万", "1月3日还清", "至今未还分文"],
            "golden_findings": [{"kind": "fact_mismatch", "segments": [1, 2]}],
            "consistency_label": "contradictory"}},
        "article_knowledge": {"id": "c1", "provenance": "curated", "payload": {
            "text": "维生素C可以治愈感冒。", "source_url": "https://e.com",
            "golden_claims": [{"claim": "维生素C可以治愈感冒", "label": "inaccurate"}],
            "knowledge_grade": "D"}},
        "instruction_following": {"id": "c1", "provenance": "curated", "payload": {
            "prompt": "输出一句包含TESTOK的话。",
            "instructions": [{"type": "must_include", "value": "TESTOK"}]}},
        "structured_extraction": {"id": "c1", "provenance": "curated", "payload": {
            "document_type": "invoice",
            "text": "Field A: value_a",
            "fields": ["field_a"],
            "golden": {"field_a": "value_a"}}},
        "function_calling": {"id": "c1", "provenance": "curated", "payload": {
            "tools": [{"name": "tool_x", "description": "Test tool", "parameters": {}}],
            "messages": [{"role": "user", "content": "Run tool x"}],
            "expected": {"name": "tool_x", "arguments": {}}}},
        "vlm_document_extraction": {"id": "c1", "provenance": "curated", "payload": {
            "document_type": "receipt",
            "image_path": "fixtures/scenarios/vlm_document_extraction/receipt/c17.png",
            "fields": ["field_a"],
            "golden": {"field_a": "value_a"}}},
        "adversarial_stability": {"id": "c1", "provenance": "curated", "payload": {
            "prompt": "请翻译：hello",
            "instructions": [{"type": "must_include", "value": "TESTOK"}]}},
    }
    for name, row in rows.items():
        d = base / name
        d.mkdir(parents=True)
        (d / "cases.jsonl").write_text(json.dumps(row, ensure_ascii=False),
                                       encoding="utf-8")
    monkeypatch.setattr(runner_mod, "ROOT", tmp_path)


def test_perfect_run_passes(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={})
    assert out["verdict"] == "PASS"
    assert out["judge_calibration"]["passed"] is True
    assert set(out["scenarios"]) == {
        "wechat_intent", "case_logic", "article_knowledge",
        "instruction_following", "structured_extraction", "function_calling",
        "vlm_document_extraction", "adversarial_stability",
    }
    s1 = out["scenarios"]["wechat_intent"]
    assert s1["l1"]["intent_accuracy"] == 1.0
    assert s1["l2_judge"]["mean"] == 5.0
    assert s1["provenance"] == {"curated": 1}


def test_text_model_skips_vlm_scenario(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    out = runner_mod.run_scenarios(_TextModel(), judge_cfg=_Judge(), cfg={})
    assert out["scenarios"]["wechat_intent"]["verdict"] == "SKIPPED"
    assert out["verdict"] == "PASS"                   # SKIPPED 不拖累


def test_missing_cases_is_blocked_warn(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "datasets" / "scenarios" / "case_logic")
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={})
    assert out["scenarios"]["case_logic"]["verdict"] == "BLOCKED"
    assert out["verdict"] == "WARN"                   # 绝不空跑 PASS


def test_judge_equals_model_caps_at_warn(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_VLM(), cfg={})
    assert out["verdict"] == "WARN"
    assert any("judge" in r for r in out["verdict_reasons"])
    assert out["scenarios"]["wechat_intent"]["l2_judge"] is None


def test_all_synthetic_caps_at_warn(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    p = tmp_path / "datasets" / "scenarios" / "wechat_intent" / "cases.jsonl"
    row = json.loads(p.read_text(encoding="utf-8"))
    row["provenance"] = "synthetic"
    p.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={})
    assert out["scenarios"]["wechat_intent"]["verdict"] == "WARN"
    assert out["verdict"] == "WARN"


def test_l1_below_threshold_fails(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    monkeypatch.setattr(runner_mod, "infer_sync",
                        lambda *a, **kw: _ok({"intent": "denial", "entities": [],
                                              "consistency": "consistent", "findings": [],
                                              "claims": [], "grade": "A"}))
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={})
    assert out["verdict"] == "FAIL"


# ---- per-case 异常不允许炸掉整个 run ----

def test_per_case_exception_does_not_crash_run(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)

    def raising_infer(*a, **kw):
        raise FileNotFoundError("fixtures/scenarios/wechat_intent/c1.png")

    monkeypatch.setattr(runner_mod, "infer_sync", raising_infer)
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={})
    s1 = out["scenarios"]["wechat_intent"]
    assert s1["error_rate"] == 1.0
    assert s1["verdict"] == "FAIL"
    assert out["verdict"] == "FAIL"


def test_malformed_payload_does_not_crash_run(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    p = tmp_path / "datasets" / "scenarios" / "wechat_intent" / "cases.jsonl"
    row = json.loads(p.read_text(encoding="utf-8"))
    del row["payload"]["expected_intent"]            # l1_score 会 KeyError
    p.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={})
    s1 = out["scenarios"]["wechat_intent"]
    assert s1["error_rate"] == 1.0
    assert s1["verdict"] == "FAIL"
    assert out["verdict"] == "FAIL"


# ---- UNSCORED rate gate + multi-seed honesty ----

def _patch_judge_bad_on_real_cases(monkeypatch, *, bad_seeds=None):
    """真实 case(prompt 含 'case c1')返回非法 JSON;锚定题保持有判别力。"""
    def fake_judge_infer(model_cfg, *, prompt, seed=None, **kw):
        if "case c1" in prompt and (bad_seeds is None or seed in bad_seeds):
            return _ok({"bad": 1})
        bad_markers = ('"negotiation"',
                       '"consistency": "consistent"',
                       '"label": "accurate"')
        score = 1 if any(m in prompt for m in bad_markers) else 5
        return _ok({"score": score, "rationale": "r"})

    monkeypatch.setattr(runner_mod.judge_mod, "infer_sync", fake_judge_infer)


def test_unscored_rate_gate_caps_warn(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    _patch_judge_bad_on_real_cases(monkeypatch)       # 全 seed 都 unscored
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={})
    assert out["judge_calibration"]["passed"] is True
    s1 = out["scenarios"]["wechat_intent"]
    assert s1["l2_judge"] is None
    assert s1["judge_unscored_rate"] == 1.0
    assert s1["verdict"] == "WARN"
    assert any("unscored" in r for r in s1["verdict_reasons"])
    assert out["verdict"] == "WARN"


def test_one_bad_seed_still_scored(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    _patch_judge_bad_on_real_cases(monkeypatch, bad_seeds={2})   # 2/3 seed 有效
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={})
    s1 = out["scenarios"]["wechat_intent"]
    assert s1["l2_judge"] is not None
    assert s1["l2_judge"]["mean"] == 5.0
    assert s1["l2_judge"]["seeds"] == 3
    assert s1["l2_judge"]["unscored_rate"] == 0.0
    assert s1["verdict"] == "PASS"


# ---- cfg 阈值覆盖 ----

def test_threshold_override_from_cfg(monkeypatch, tmp_path):
    _patch_perfect_run(monkeypatch, tmp_path)
    cfg = {"thresholds": {"wechat_intent": {"intent_accuracy_min": 1.1}}}
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg=cfg)
    s1 = out["scenarios"]["wechat_intent"]
    assert s1["verdict"] == "FAIL"
    assert any("intent_accuracy" in r for r in s1["verdict_reasons"])
    assert out["verdict"] == "FAIL"


# ---- consistency_runs ----

def test_consistency_single_run_no_consistency_key(monkeypatch, tmp_path):
    """consistency_runs=1 (default) must not emit consistency_rate — backward compat."""
    _patch_perfect_run(monkeypatch, tmp_path)
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={}, consistency_runs=1)
    s1 = out["scenarios"]["wechat_intent"]
    assert "consistency_rate" not in s1


def test_consistency_multi_run(monkeypatch, tmp_path):
    """consistency_runs=3 with a perfect model → consistency_rate 1.0, verdict PASS."""
    _patch_perfect_run(monkeypatch, tmp_path)
    out = runner_mod.run_scenarios(_VLM(), judge_cfg=_Judge(), cfg={}, consistency_runs=3)
    s1 = out["scenarios"]["wechat_intent"]
    assert s1["consistency_runs"] == 3
    assert s1["consistency_rate"] == 1.0
    assert out["verdict"] == "PASS"
