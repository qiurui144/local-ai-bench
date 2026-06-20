"""场景维度通用 runner:加载 cases → 被测推理 → L1 → L2(judge)→ verdict。

verdict 语义(与主 harness「空跑不得 PASS」原则一致):
- cases 缺失 → BLOCKED(维度级记 WARN)
- error_rate>0.2 或 L1 低于阈值 → FAIL
- judge 不可用 / 未校准 / judge==被测模型 / 全 synthetic 数据 → 封顶 WARN
"""
from __future__ import annotations

import logging
import statistics
from collections import Counter
from pathlib import Path

from common import ModelConfig, infer_sync

from benchmark.registry import cap_warn as _cap_warn, worst_verdict as _worst

from . import SCENARIOS
from . import judge as judge_mod
from .base import load_cases

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _calibration_anchors() -> list[dict]:
    """锚定题的真 rubric 按 scenario 字段从注册表注入(文件里的是人读注记)。

    全量锚定题(好/坏配对)都参与校准:judge 必须能区分两极,否则
    "永远打满分"的退化 judge 也能通过 — 那校准就没有意义了。
    """
    anchors = []
    for a in judge_mod.load_anchors():
        spec = SCENARIOS.get(a.get("scenario"))
        if spec is None:
            continue
        anchors.append(dict(a, rubric=spec.judge_rubric))
    return anchors


def _primary_metric(spec) -> str | None:
    """Derive primary L1 metric key.

    Prefers the first threshold key (stripped of _min) when it appears directly
    in per-case l1_score output; falls back to None if default_thresholds is
    empty.  Callers must check whether the returned key is actually present in
    the score dict before using it — the threshold metric may be an aggregate
    alias (e.g. ``intent_accuracy`` computed over ``intent_hit``).
    """
    for key in spec.default_thresholds:
        return key.removesuffix("_min")
    return None


def run_scenarios(model_cfg: ModelConfig, *, judge_cfg: ModelConfig | None,
                  cfg: dict, consistency_runs: int = 1) -> dict:
    out: dict = {"benchmark": "scenarios", "model": model_cfg.name,
                 "judge_model": getattr(judge_cfg, "name", None),
                 "judge_calibration": None, "scenarios": {},
                 "verdict": "PASS", "verdict_reasons": []}

    judge_ok = judge_cfg is not None and judge_cfg.name != model_cfg.name
    if not judge_ok:
        out["verdict_reasons"].append(
            "L2 judge unavailable or judge==model-under-test — L1-only, capped at WARN")
    else:
        out["judge_calibration"] = judge_mod.calibrate(judge_cfg, _calibration_anchors())
        if not out["judge_calibration"]["passed"]:
            judge_ok = False
            out["verdict_reasons"].append(
                "judge anchor calibration failed — L1-only, capped at WARN")

    num_cases = cfg.get("num_cases")
    thr_cfg = (cfg.get("thresholds") or {})
    for name, spec in SCENARIOS.items():
        block = _run_one_scenario(model_cfg, spec,
                                  judge_cfg if judge_ok else None, num_cases,
                                  thr_cfg.get(spec.name) or {},
                                  consistency_runs=consistency_runs)
        out["scenarios"][name] = block
        for r in block.get("verdict_reasons", []):
            out["verdict_reasons"].append(f"[{name}] {r}")

    out["verdict"] = _worst(b["verdict"] for b in out["scenarios"].values())
    if not judge_ok and out["verdict"] == "PASS":
        out["verdict"] = "WARN"
    return out


def _safe_l1_null(spec, case) -> dict:
    """l1_score(case, None, "") 自身也可能因 payload 畸形而抛 — 兜底空 dict。"""
    try:
        return spec.l1_score(case, None, "")
    except Exception:
        return {}


def _run_one_scenario(model_cfg, spec, judge_cfg, num_cases,
                      threshold_override=None, consistency_runs: int = 1) -> dict:
    if spec.requires_vlm and not model_cfg.is_vlm:
        return {"verdict": "SKIPPED", "reason": "requires VLM"}

    cases = load_cases(ROOT / spec.cases_path, num_samples=num_cases)
    if cases is None:
        logger.warning("scenario %s: cases file missing (%s) — BLOCKED",
                       spec.name, spec.cases_path)
        return {"verdict": "BLOCKED", "reason": "cases file missing",
                "verdict_reasons": ["cases file missing — build the dataset"]}

    per_case_l1, per_case_scores_k, errors = [], [], 0
    judge_means, judge_attempted, judge_unscored = [], 0, 0
    for case in cases:
        # 单 case 异常(截图缺失/payload 畸形/编码失败)不允许炸掉整个
        # 多小时 run — 记 error 继续。
        case_k_scores = []  # K primary-metric scores for this case
        last_r = None
        for _run_i in range(max(1, consistency_runs)):
            try:
                prompt, image = spec.build_prompt(case)
                r = infer_sync(model_cfg, prompt=prompt,
                               image_path=(ROOT / image) if image else None,
                               max_tokens=800)
                last_r = r
            except Exception as exc:
                errors += 1
                logger.warning("scenario %s case %s run %d: infer failed: %s",
                               spec.name, case.id, _run_i, exc)
                per_case_l1.append(_safe_l1_null(spec, case))
                case_k_scores = []  # invalidate consistency for this case
                break
            if not r.ok:
                errors += 1
                per_case_l1.append(_safe_l1_null(spec, case))
                case_k_scores = []
                break
            try:
                s = spec.l1_score(case, r.parsed_json, r.content)
                if _run_i == 0:
                    per_case_l1.append(s)
                # track primary metric for consistency;
                # threshold name may be an aggregate alias, so fall back to
                # first numeric key in the per-case score dict.
                primary_key = _primary_metric(spec)
                if primary_key and primary_key in s:
                    case_k_scores.append(s[primary_key])
                elif s:
                    case_k_scores.append(next(iter(s.values())))
            except Exception as exc:
                errors += 1
                logger.warning("scenario %s case %s run %d: l1_score failed: %s",
                               spec.name, case.id, _run_i, exc)
                if _run_i == 0:
                    per_case_l1.append(_safe_l1_null(spec, case))
                case_k_scores = []
                break

        per_case_scores_k.append(case_k_scores)

        # Judge only on last successful run (avoid K × judge calls)
        if last_r is not None and last_r.ok and judge_cfg is not None:
            judge_attempted += 1
            summary = f"场景 {spec.name},case {case.id},期望: {case.payload}"
            try:
                j = judge_mod.judge_case(judge_cfg, spec.judge_rubric,
                                         summary, last_r.content)
            except Exception as exc:
                logger.warning("scenario %s case %s: judge failed: %s",
                               spec.name, case.id, exc)
                judge_unscored += 1
                continue
            # 多 seed 诚信:≥2/3 seed 有效才算有分,否则该 case 计 unscored
            if j["unscored"] <= 1:
                judge_means.append(j["mean"])
            else:
                judge_unscored += 1

    n = len(cases)
    l1 = spec.aggregate_l1(per_case_l1)
    unscored_rate = judge_unscored / judge_attempted if judge_attempted else 0.0
    block: dict = {
        "n_cases": n,
        "provenance": dict(Counter(c.provenance for c in cases)),
        "error_rate": errors / n if n else 0.0,
        "l1": l1,
        "l2_judge": None,
        "judge_unscored_rate": unscored_rate,
        "verdict_reasons": [],
    }

    verdict = "PASS"
    if block["error_rate"] > 0.2:
        verdict = "FAIL"
        block["verdict_reasons"].append(f"error_rate {block['error_rate']:.0%} > 20%")
    thresholds = {**spec.default_thresholds, **(threshold_override or {})}
    for key, floor in thresholds.items():
        metric = key.removesuffix("_min")
        if l1.get(metric, 0.0) < floor:
            verdict = "FAIL"
            block["verdict_reasons"].append(
                f"L1 {metric} {l1.get(metric, 0):.2f} < {floor}")

    if judge_cfg is not None and judge_attempted and unscored_rate > 0.10:
        # judge 在真实输出上大面积失效 — L2 不可信,整段作废并封顶 WARN
        block["l2_judge"] = None
        verdict = _cap_warn(verdict)
        block["verdict_reasons"].append(
            f"L2 unscored rate {unscored_rate:.0%} > 10% — "
            f"judge unusable on real outputs, L1-only")
    elif judge_cfg is not None and judge_means:
        mean = statistics.mean(judge_means)
        block["l2_judge"] = {
            "mean": mean,
            "std": statistics.stdev(judge_means) if len(judge_means) > 1 else 0.0,
            "seeds": len(judge_mod.SEEDS),
            "unscored_rate": unscored_rate,
        }
        if verdict == "PASS" and mean < 2.5:
            verdict = "FAIL"
            block["verdict_reasons"].append(f"L2 judge mean {mean:.2f} < 2.5")
        elif verdict == "PASS" and mean < 3.5:
            verdict = "WARN"
            block["verdict_reasons"].append(f"L2 judge mean {mean:.2f} < 3.5")

    if set(block["provenance"]) == {"synthetic"}:
        verdict = _cap_warn(verdict)
        block["verdict_reasons"].append(
            "all cases synthetic — capped at WARN (not real-scenario evidence)")

    # Consistency stats (only when consistency_runs > 1)
    if consistency_runs > 1 and per_case_scores_k:
        valid = [ks for ks in per_case_scores_k if len(ks) == consistency_runs]
        if valid:
            consistent = sum(1 for ks in valid if max(ks) - min(ks) <= 0.10)
            stds = [statistics.stdev(ks) for ks in valid if len(ks) > 1]
            block["consistency_runs"] = consistency_runs
            block["consistency_rate"] = consistent / len(valid)
            block["l1_mean_std"] = sum(stds) / len(stds) if stds else 0.0
            block["consistency_eligible_cases"] = len(valid)

    block["verdict"] = verdict
    return block
