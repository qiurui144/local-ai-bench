"""D2 replaceability:两模型已存报告离线对比,2σ 纪律硬编码不可绕。

verdict: REPLACEABLE(0) / INCONCLUSIVE(1) / NOT_REPLACEABLE(2)。
单 seed → 硬性封顶 INCONCLUSIVE;harness_version / condition 不一致 →
INCONCLUSIVE;hardware_profile 不一致 → 性能侧强制 INCONCLUSIVE(质量仍比)。
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import re
from pathlib import Path
from typing import Optional

from benchmark.registry import SCHEMA_VERSION, collect_quality_leaves
from benchmark.rigor.multi_seed_runner import SeedRun, aggregate, two_sigma_significant

EXIT_BY_VERDICT = {"REPLACEABLE": 0, "INCONCLUSIVE": 1, "NOT_REPLACEABLE": 2}
LOWER_IS_BETTER_TOKENS = ("cer", "wer", "rtf", "error_rate", "_ms", "latency",
                          "violations", "truncation", "drop", "unscored", "drift")


def _higher_is_better(path: str) -> bool:
    p = path.lower()
    return not any(t in p for t in LOWER_IS_BETTER_TOKENS)


def _quality_dims() -> tuple:
    import run_benchmark as rb   # 延迟 import 防环
    return rb.QUALITY_DIMS


def load_group(reports_dir: Path, model: str) -> Optional[dict]:
    """模型最新 merged 报告 + 其 seed 归档(凭文件名约定 {model}_{ts}[_seedK])。

    文件名必须精确匹配 {model}_<YYYYMMDD>_<HHMMSS>(防 'qwen3' 吞掉
    qwen3_mini_*.json 的前缀碰撞),且报告 model 字段必须等于请求名 —
    任一不符视为无报告(上游报 INCONCLUSIVE),绝不拿错模型报告比。"""
    stem_re = re.compile(rf"^{re.escape(model)}_\d{{8}}_\d{{6}}$")
    merged = sorted(p for p in Path(reports_dir).glob(f"{model}_*.json")
                    if stem_re.match(p.stem))
    if not merged:
        return None
    latest = merged[-1]
    report = json.loads(latest.read_text(encoding="utf-8"))
    if report.get("model") != model:
        return None
    seeds = sorted(Path(reports_dir).glob(f"{latest.stem}_seed*.json"))
    return {"merged": report,
            "seeds": [json.loads(p.read_text(encoding="utf-8")) for p in seeds],
            "path": str(latest)}


# (dim, threshold key, metric getter, op) — 阈值键/取值路径对齐 models.yaml 与 runner 输出
_PERF_CHECKS = (
    ("ttft", "p95_ttft_ms_max",
     lambda b: b.get("ttft_ms_stats", {}).get("p95", 0), "le"),
    ("throughput", "tps_min", lambda b: b.get("aggregate_tps", 0), "ge"),
    ("prefill_decode", "tg_tps_min",
     lambda b: b.get("decode", {}).get("tok_per_sec", {}).get("p50", 0), "ge"),
    ("concurrency", "success_rate_min",
     lambda b: min((s.get("success_rate", 0) for s in b.get("steps") or []), default=0), "ge"),
    ("stability", "latency_drift_ratio_max",
     lambda b: b.get("latency_drift_ratio", 0), "le"),
)


def _deep_merge(base: dict | None, override: dict | None) -> dict:
    merged = copy.deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _bench_cfg_for_report(report: dict, fallback: dict) -> dict:
    return _deep_merge(fallback, report.get("benchmark_config") or {})


def check_candidate_performance(report: dict, bench_cfg: dict) -> tuple[str, list[str]]:
    """候选自身性能阈值(models.yaml)检查;不与 baseline 比相对值(Q2)。"""
    bench_cfg = _bench_cfg_for_report(report, bench_cfg)
    bm = report.get("benchmarks") or {}
    checks, reasons = [], []
    for dim, key, getter, op in _PERF_CHECKS:
        thr = ((bench_cfg.get(dim) or {}).get("thresholds") or {}).get(key)
        if dim not in bm or thr is None:
            continue
        val = getter(bm[dim])
        ok = (val <= thr) if op == "le" else (val >= thr)
        checks.append(ok)
        if not ok:
            reasons.append(f"{dim}: {val} violates {key}={thr}")
    if not checks:
        return "UNKNOWN", ["no performance dimensions present in candidate report"]
    return ("PASS" if all(checks) else "FAIL"), reasons


def compare_reports(base: dict, cand: dict, bench_cfg: dict) -> dict:
    bm, cm = base["merged"], cand["merged"]
    out = {"baseline": bm.get("model"), "candidate": cm.get("model"),
           "verdict": "INCONCLUSIVE", "quality": {}, "performance": {}, "reasons": []}

    for r, side in ((bm, "baseline"), (cm, "candidate")):
        if r.get("schema_version") != SCHEMA_VERSION:
            out["reasons"].append(
                f"{side} report has no schema_version={SCHEMA_VERSION} (legacy) — not comparable")
            return out
    if bm.get("harness_version") != cm.get("harness_version"):
        out["reasons"].append("harness_version mismatch — methodology changes make scores incomparable")
        return out
    if bm.get("condition") != cm.get("condition"):
        out["reasons"].append("condition mismatch — only same-condition reports are comparable")
        return out

    perf_verdict, perf_reasons = check_candidate_performance(cm, bench_cfg)
    if bm.get("hardware_profile") != cm.get("hardware_profile"):
        perf_verdict = "INCONCLUSIVE"
        perf_reasons.append("hardware_profile differs — performance side forced INCONCLUSIVE (Q6)")
    out["performance"] = {"candidate_thresholds": perf_verdict, "reasons": perf_reasons}

    qd = _quality_dims()
    base_runs = [collect_quality_leaves(s, qd) for s in (base["seeds"] or [bm])]
    cand_runs = [collect_quality_leaves(s, qd) for s in (cand["seeds"] or [cm])]
    shared = set.intersection(*(set(r) for r in base_runs + cand_runs))
    if not shared:
        out["reasons"].append("no shared quality metrics between the two reports")
        return out

    multi_seed = len(base["seeds"]) >= 2 and len(cand["seeds"]) >= 2
    b_aggs = aggregate([SeedRun(seed=i, metrics=r, duration_s=0.0)
                        for i, r in enumerate(base_runs)])
    c_aggs = aggregate([SeedRun(seed=i, metrics=r, duration_s=0.0)
                        for i, r in enumerate(cand_runs)])
    regression = False
    for path in sorted(shared):
        a, c = b_aggs[path], c_aggs[path]
        delta = round(c.mean - a.mean, 6)
        sigma = round(max(a.std, c.std), 6)
        significant = bool(multi_seed and two_sigma_significant(a, c))
        if delta == 0:
            direction = "equal"
        else:
            better = (delta > 0) == _higher_is_better(path)
            direction = "better" if better else "worse"
        out["quality"][path] = {"delta": delta, "sigma": sigma,
                                "significant": significant, "direction": direction}
        if significant and direction == "worse":
            regression = True
            out["reasons"].append(f"{path}: Δ={delta} ≥ 2σ ({sigma}) regression")

    base_perf, base_perf_reasons = check_candidate_performance(bm, bench_cfg)
    if base_perf == "FAIL":
        out["reasons"].append("note: baseline itself fails its performance thresholds: "
                              + "; ".join(base_perf_reasons))

    if not multi_seed:
        out["verdict"] = "INCONCLUSIVE"
        out["reasons"].append("single-seed data — rankings are noise; rerun with --seeds 3 (hard cap, not configurable)")
    elif regression:
        out["verdict"] = "NOT_REPLACEABLE"
    elif perf_verdict == "PASS":
        out["verdict"] = "REPLACEABLE"
        out["reasons"].append("all shared quality dims within 2σ + candidate performance thresholds PASS")
    elif perf_verdict == "FAIL":
        out["verdict"] = "NOT_REPLACEABLE"
        out["reasons"].append("quality equivalent but candidate fails its own performance thresholds")
    else:
        out["reasons"].append("performance side UNKNOWN/INCONCLUSIVE — cannot conclude replaceability")
    return out


def render_compare_markdown(out: dict) -> str:
    lines = [f"# Compare: {out['candidate']} vs baseline {out['baseline']}", "",
             f"**Verdict: {out['verdict']}**", "",
             f"- performance(candidate thresholds): {out['performance'].get('candidate_thresholds')}",
             "", "| metric | Δ (cand−base) | σ | significant | direction |", "|---|---|---|---|---|"]
    for path, q in out["quality"].items():
        lines.append(f"| {path} | {q['delta']:+.4f} | {q['sigma']:.4f} "
                     f"| {q['significant']} | {q['direction']} |")
    lines += ["", "## Reasons"] + [f"- {r}" for r in out["reasons"]]
    return "\n".join(lines)


def run_compare(baseline: str, candidate: str, reports_dir: Path, bench_cfg: dict) -> int:
    reports_dir = Path(reports_dir)
    base = load_group(reports_dir, baseline)
    cand = load_group(reports_dir, candidate)
    if base is None or cand is None:
        missing = baseline if base is None else candidate
        out = {"baseline": baseline, "candidate": candidate, "verdict": "INCONCLUSIVE",
               "quality": {}, "performance": {},
               "reasons": [f"no report found for {missing!r} in {reports_dir}"]}
    else:
        out = compare_reports(base, cand, bench_cfg)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"compare_{baseline}_vs_{candidate}_{ts}"
    (reports_dir / f"{stem}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / f"{stem}.md").write_text(render_compare_markdown(out), encoding="utf-8")
    from benchmark.report.html_report import generate_html
    html_report = dict(out, mode="compare", final_verdict=out.get("verdict"),
                       baseline_report=base["merged"] if base else {},
                       candidate_report=cand["merged"] if cand else {})
    (reports_dir / f"{stem}.html").write_text(
        generate_html(html_report), encoding="utf-8")
    print(render_compare_markdown(out))
    return EXIT_BY_VERDICT[out["verdict"]]


def main(argv=None) -> int:
    from common import load_benchmarks_config
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Replaceability compare of two saved report sets")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--reports-dir", default=str(root / "output" / "reports"))
    args = ap.parse_args(argv)
    return run_compare(args.baseline, args.candidate, Path(args.reports_dir),
                       load_benchmarks_config(root / "models.yaml"))
