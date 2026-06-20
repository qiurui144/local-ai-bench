"""benchmark.accuracy 离线测试 —— _normalize / judge_case / run_accuracy 判定契约。

Coverage backfill：infer_sync 通过 monkeypatch 打在 benchmark.accuracy
模块自己的引用上，全程不打网络。曾以 strict xfail 记录的 4 个真实 bug
（BUG-1 must_not_say 子串前缀误判 / BUG-2 实体 digit-append 误命中 /
BUG-3 description null 崩溃 / BUG-4 缺 acceptance_criteria 时 None*100）
已在源码修复，对应测试转为常规断言正确行为。
"""
import json
from pathlib import Path

from benchmark import accuracy
from common import InferResult, ModelConfig

_REPO = Path(__file__).resolve().parent.parent
ACC = json.loads(
    (_REPO / "golden" / "expectations.json").read_text(encoding="utf-8")
)["acceptance_criteria"]


def _model(task_type="vlm"):
    return ModelConfig(name="stub-vlm", hf_repo="org/stub", port=9999,
                       vram_estimate_gb=1.0, role="dut", task_type=task_type)


def _pred(category="communication", desc="alice says hi", *, ok=True, **kw):
    defaults = dict(
        model="stub", ok=ok, error="" if ok else "boom",
        parsed_json={"category": category, "description": desc} if ok else None,
        input_tokens=900, output_tokens=200,
        finish_reason="stop", latency_ms=1000.0,
    )
    defaults.update(kw)
    return InferResult(**defaults)


def _case(i, *, category="communication", entities=("alice",), facts=(),
          must_not=(), budget=None):
    c = {
        "id": f"case_{i}",
        "image": f"{i}.jpg",
        "expected_category": category,
        "must_identify_entities": list(entities),
        "must_identify_facts": list(facts),
    }
    if must_not:
        c["must_not_say"] = list(must_not)
    if budget:
        c["token_budget"] = budget
    return c


def _run(monkeypatch, tmp_path, cases, preds_by_image, criteria=ACC,
         create_images=True):
    """跑 run_accuracy，infer_sync 替换为按图片名查表的 stub。"""
    golden = {"cases": cases}
    if criteria is not None:
        golden["acceptance_criteria"] = criteria
    if create_images:
        for c in cases:
            (tmp_path / c["image"]).write_bytes(b"\xff\xd8")
    monkeypatch.setattr(
        accuracy, "infer_sync",
        lambda model_cfg, **kw: preds_by_image[kw["image_path"].name],
    )
    return accuracy.run_accuracy(_model(), golden, tmp_path)


# ─── _normalize ───

def test_normalize_strips_spaces_commas_yen_and_lowercases():
    assert accuracy._normalize(" ¥ 1,200 OK ") == "1200ok"
    assert accuracy._normalize("金融，类") == "金融类"          # 全角逗号
    assert accuracy._normalize("Alice") == accuracy._normalize("ALICE")


def test_normalize_non_str_passthrough_skips_lowercasing():
    # 非 str 走 str(s) 直通，不做 lower/替换 —— "None"/"True" 保留大写
    assert accuracy._normalize(120) == "120"
    assert accuracy._normalize(None) == "None"
    assert accuracy._normalize(True) == "True"


def test_normalize_keeps_tabs_newlines_and_fullwidth_yen():
    # 仅 U+0020 空格被删；tab/newline/全角 ￥(U+FFE5) 保留（current behavior）
    assert accuracy._normalize("a\tb\nc") == "a\tb\nc"
    assert accuracy._normalize("￥1200") == "￥1200"


# ─── judge_case ───

def test_judge_case_exact_match_full_structure():
    case = _case(0, entities=("Alice", "¥1200"),
                 budget={"input": [700, 1800], "output": [80, 500]})
    pred = _pred("Communication ,", "Alice received ¥ 1,200")
    r = accuracy.judge_case(case, pred)
    assert r["category_correct"] is True            # 归一化后等值
    assert r["entity_hits"] == 2 and r["entity_total"] == 2
    assert r["must_not_violations"] == []
    assert r["input_in_range"] is True and r["output_in_range"] is True
    assert r["possibly_truncated"] is False
    assert r["ok"] is True and r["error"] == ""
    assert r["case_id"] == "case_0" and r["image"] == "0.jpg"
    assert r["predicted_category"] == "Communication ,"


def test_judge_case_digit_shift_entity_not_matched():
    # ¥120（位移错误）不能命中期望实体 ¥1200
    case = _case(0, entities=("¥1200",))
    r = accuracy.judge_case(case, _pred(desc="received ¥120 from alice"))
    assert r["entity_hits"] == 0


def test_judge_case_category_digit_shift_not_equal():
    case = _case(0, category="¥1200")
    r = accuracy.judge_case(case, _pred(category="¥120"))
    assert r["category_correct"] is False


def test_judge_case_must_not_say_violation_detected():
    case = _case(0, must_not=("holdharmless",))
    r = accuracy.judge_case(case, _pred(desc="this is a HoldHarmless clause"))
    assert r["must_not_violations"] == ["holdharmless"]


def test_judge_case_none_prediction():
    case = _case(0, entities=("alice",), must_not=("bad",))
    r = accuracy.judge_case(case, _pred(ok=False, parsed_json=None))
    assert r["category_correct"] is False
    assert r["entity_hits"] == 0
    assert r["must_not_violations"] == []
    assert r["predicted_category"] == ""
    assert r["ok"] is False and r["error"] == "boom"


def test_judge_case_fact_matches_on_first_eight_chars():
    # fact 只取前 8 个字符做前缀匹配
    case = _case(0, facts=("reconciliation_window", "zzzzzzzz_missing"))
    r = accuracy.judge_case(case, _pred(desc="did a reconcilXXX today"))
    assert r["fact_hits"] == 1 and r["fact_total"] == 2


def test_judge_case_token_budget_and_truncation_boundaries():
    case = _case(0, budget={"input": [700, 1800], "output": [80, 500]})
    r = accuracy.judge_case(case, _pred(input_tokens=699, output_tokens=500))
    assert r["input_in_range"] is False and r["output_in_range"] is True
    # 截断判定：finish_reason=length 或 output ≥ 800*0.95 = 760
    assert accuracy.judge_case(case, _pred(output_tokens=759))[
        "possibly_truncated"] is False
    assert accuracy.judge_case(case, _pred(output_tokens=760))[
        "possibly_truncated"] is True
    assert accuracy.judge_case(case, _pred(finish_reason="length"))[
        "possibly_truncated"] is True


def test_judge_case_description_truncated_to_100_chars():
    r = accuracy.judge_case(_case(0), _pred(desc="x" * 300))
    assert r["predicted_description"] == "x" * 100


def test_correct_amount_must_not_trigger_prefix_forbidden_terms():
    # golden chat_01 实拍：must_not_say ["120","12"]，正确答案 ¥1200
    case = _case(0, entities=("¥1200",), must_not=("120", "12"))
    r = accuracy.judge_case(case, _pred(desc="received ¥1200 from alice"))
    assert r["must_not_violations"] == []


def test_entity_digit_append_superstring_is_not_a_hit():
    case = _case(0, entities=("¥1200",))
    r = accuracy.judge_case(case, _pred(desc="received ¥12000"))
    assert r["entity_hits"] == 0


def test_judge_case_null_description_does_not_crash():
    r = accuracy.judge_case(_case(0), _pred(parsed_json={
        "category": "communication", "description": None}))
    assert r["category_correct"] is True


# ─── run_accuracy：skip / 缺图 / 空跑 ───

def test_run_accuracy_skips_text_only_model(monkeypatch):
    monkeypatch.setattr(
        accuracy, "infer_sync",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not infer")),
    )
    out = accuracy.run_accuracy(_model(task_type="text_only"), {}, Path("/nope"))
    assert out == {"skipped": True, "reason": "text_only model"}


def test_run_accuracy_skips_missing_images(monkeypatch, tmp_path):
    cases = [_case(0), _case(1)]
    (tmp_path / "0.jpg").write_bytes(b"\xff\xd8")   # 只创建 case_0 的图
    monkeypatch.setattr(accuracy, "infer_sync", lambda m, **kw: _pred())
    out = accuracy.run_accuracy(_model(), {"cases": cases,
                                           "acceptance_criteria": ACC}, tmp_path)
    assert [r["case_id"] for r in out["per_case"]] == ["case_0"]


def test_run_accuracy_zero_measured_cases_is_blocked(monkeypatch, tmp_path):
    """全部图片缺失 → 物料阻塞，不把缺 fixture 误判为模型质量 FAIL。"""
    out = _run(monkeypatch, tmp_path, [_case(0)], {}, create_images=False)
    assert out["per_case"] == []
    assert out["status"] == "blocked"
    assert out["verdict"] == "SKIP"
    assert out["reason"] == "no VLM fixture images found"
    assert out["aggregate"]["total_cases"] == 0
    assert out["missing_images"] == ["0.jpg"]


# ─── run_accuracy：阈值边界 → verdict 映射（真实 expectations.json 阈值）───

def test_category_precision_at_threshold_passes(monkeypatch, tmp_path):
    # 4/5 = 0.80 == category_precision_min → 不触发（严格 <）→ PASS
    cases = [_case(i) for i in range(5)]
    preds = {f"{i}.jpg": _pred() for i in range(4)}
    preds["4.jpg"] = _pred(category="financial")
    out = _run(monkeypatch, tmp_path, cases, preds)
    assert out["aggregate"]["category_precision"] == 0.8
    assert out["verdict"] == "PASS"
    assert out["verdict_reasons"] == []
    assert out["benchmark"] == "accuracy" and out["model"] == "stub-vlm"


def test_category_precision_below_threshold_fails(monkeypatch, tmp_path):
    cases = [_case(i) for i in range(5)]
    preds = {f"{i}.jpg": _pred() for i in range(3)}
    preds.update({"3.jpg": _pred(category="financial"),
                  "4.jpg": _pred(category="form")})
    out = _run(monkeypatch, tmp_path, cases, preds)
    assert out["verdict"] == "FAIL"
    assert "FAIL: 分类 precision 60.0% < 80%" in out["verdict_reasons"]


def test_entity_recall_at_threshold_passes(monkeypatch, tmp_path):
    # 3/5 = 0.60 == entity_recall_min → PASS（严格 <）
    cases = [_case(0, entities=("alpha", "bravo", "charlie", "delta", "echo"))]
    out = _run(monkeypatch, tmp_path, cases,
               {"0.jpg": _pred(desc="alpha bravo charlie")})
    assert out["aggregate"]["entity_recall"] == 0.6
    assert out["verdict"] == "PASS"


def test_entity_recall_below_threshold_fails(monkeypatch, tmp_path):
    cases = [_case(0, entities=("alpha", "bravo", "charlie", "delta", "echo"))]
    out = _run(monkeypatch, tmp_path, cases, {"0.jpg": _pred(desc="alpha bravo")})
    assert out["aggregate"]["entity_recall"] == 0.4
    assert out["verdict"] == "FAIL"
    assert "FAIL: 实体 recall 40.0% < 60%" in out["verdict_reasons"]


def test_no_entities_anywhere_means_zero_recall_fail(monkeypatch, tmp_path):
    """current behavior：golden 全无实体 → entity_total 强制 1 → recall
    0.0 < 0.6 → 必 FAIL（`or 1` 让 N/A 分支永不可达）。真实 golden 带 3
    个实体所以不踩；记录以防回归到更糟。"""
    out = _run(monkeypatch, tmp_path, [_case(0, entities=())], {"0.jpg": _pred()})
    assert out["aggregate"]["entity_recall"] == 0.0
    assert out["verdict"] == "FAIL"
    assert any("实体 recall" in r for r in out["verdict_reasons"])


def test_error_rate_at_threshold_passes(monkeypatch, tmp_path):
    # 1/20 = 0.05 == error_rate_max → 不触发（严格 >）→ PASS
    cases = [_case(i) for i in range(20)]
    preds = {f"{i}.jpg": _pred() for i in range(19)}
    preds["19.jpg"] = _pred(ok=False, latency_ms=0.0,
                            input_tokens=0, output_tokens=0)
    out = _run(monkeypatch, tmp_path, cases, preds)
    assert out["aggregate"]["error_rate"] == 0.05
    assert out["verdict"] == "PASS"
    # token / latency 统计只算 ok 的 19 个 case
    assert out["aggregate"]["latency_stats_ms"]["count"] == 19
    assert out["aggregate"]["input_tokens_stats"]["min"] == 900


def test_error_rate_above_threshold_fails(monkeypatch, tmp_path):
    cases = [_case(i) for i in range(20)]
    preds = {f"{i}.jpg": _pred() for i in range(18)}
    preds.update({"18.jpg": _pred(ok=False), "19.jpg": _pred(ok=False)})
    out = _run(monkeypatch, tmp_path, cases, preds)
    assert out["aggregate"]["error_rate"] == 0.1
    assert out["verdict"] == "FAIL"
    assert "FAIL: 错误率 10.0% 超标" in out["verdict_reasons"]


def test_truncation_rate_at_threshold_passes(monkeypatch, tmp_path):
    # 1/10 = 0.10 == output_token_truncation_rate_max → 不触发（严格 >）
    cases = [_case(i) for i in range(10)]
    preds = {f"{i}.jpg": _pred() for i in range(9)}
    preds["9.jpg"] = _pred(finish_reason="length")
    out = _run(monkeypatch, tmp_path, cases, preds)
    assert out["aggregate"]["truncation_rate"] == 0.1
    assert out["verdict"] == "PASS"


def test_truncation_rate_above_threshold_warns(monkeypatch, tmp_path):
    # 2/10 = 0.20 > 0.10 → 仅 WARN（L161 规则），不升级为 FAIL
    cases = [_case(i) for i in range(10)]
    preds = {f"{i}.jpg": _pred() for i in range(8)}
    preds.update({"8.jpg": _pred(finish_reason="length"),
                  "9.jpg": _pred(output_tokens=800)})
    out = _run(monkeypatch, tmp_path, cases, preds)
    assert out["aggregate"]["truncation_rate"] == 0.2
    assert out["verdict"] == "WARN"
    assert out["verdict_reasons"] == ["WARN: 输出截断率 20.0%"]


def test_must_not_say_single_violation_is_red_line_fail(monkeypatch, tmp_path):
    cases = [_case(0, must_not=("holdharmless",))]
    out = _run(monkeypatch, tmp_path, cases,
               {"0.jpg": _pred(desc="alice holdharmless")})
    assert out["aggregate"]["must_not_say_violations"] == 1
    assert out["verdict"] == "FAIL"
    assert "FAIL: must_not_say 违反 1 次（红线）" in out["verdict_reasons"]


def test_fail_beats_warn_and_reasons_accumulate(monkeypatch, tmp_path):
    # 低 precision (FAIL) + 高截断率 (WARN) 同时出现 → verdict 取 FAIL
    cases = [_case(i) for i in range(5)]
    preds = {f"{i}.jpg": _pred(category="financial", finish_reason="length")
             for i in range(5)}
    out = _run(monkeypatch, tmp_path, cases, preds)
    assert out["verdict"] == "FAIL"
    assert any(r.startswith("FAIL: 分类 precision") for r in out["verdict_reasons"])
    assert "WARN: 输出截断率 100.0%" in out["verdict_reasons"]


def test_missing_acceptance_criteria_still_returns_fail(monkeypatch, tmp_path):
    cases = [_case(i) for i in range(5)]
    preds = {f"{i}.jpg": _pred(category="financial") for i in range(5)}
    out = _run(monkeypatch, tmp_path, cases, preds, criteria=None)
    assert out["verdict"] == "FAIL"
