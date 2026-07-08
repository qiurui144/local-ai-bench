"""主 benchmark 入口 —— 对 models.yaml 中声明的模型跑完整测试套件

使用：
  python run_benchmark.py --model qwen3-vl-8b-instruct
  python run_benchmark.py --model all              # 跑所有已启动的模型
  python run_benchmark.py --model qwen3-vl-8b-instruct \\
      --skip stability                             # 跳过 30 分钟稳定性
  python run_benchmark.py --model qwen3-vl-8b-instruct \\
      --seeds 3                                    # 完整重跑 3 次,报 mean±std

输出：
  output/reports/{model}_{timestamp}.json       # 机器可读
  output/reports/{model}_{timestamp}.md          # 人类可读
  output/reports/{model}_{timestamp}_seed{k}.json  # --seeds N>1 时每 seed raw 结果
  output/reports/matrix_{timestamp}.md           # 所有模型对比表（all 模式）

退出码：
  0 全部 PASS
  1 有 WARN
  2 任一模型 FAIL
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import httpx

from common import (
    ModelConfig,
    _is_chat_capable,
    get_vram_info,
    load_benchmarks_config,
    load_models,
    wait_model_ready,
)

sys.path.insert(0, str(Path(__file__).parent / "benchmark"))
from benchmark.accuracy import run_accuracy
from benchmark.performance import (
    run_concurrency,
    run_prefill_decode,
    run_stability,
    run_throughput,
    run_ttft,
)
from benchmark.translation.dimension import run_translation_dimension
from benchmark.embedding.dimension import run_embedding_dimension
from benchmark.rerank.dimension import run_rerank_dimension
from benchmark.asr.dimension import run_asr_dimension
from benchmark.ocr.dimension import run_ocr_dimension
from benchmark.conditioned.runner import run_conditioned
from benchmark.long_context.runner import run_long_context
try:
    from benchmark.general_ability.runner import run_general_ability
    _GENERAL_ABILITY_AVAILABLE = True
except ImportError as _e:
    _GENERAL_ABILITY_AVAILABLE = False
    _ga_import_error = str(_e)

    def run_general_ability(model_cfg, cfg):  # type: ignore[misc]
        return {"verdict": "BLOCKED", "reason": f"general_ability deps missing: {_ga_import_error}",
                "tasks": {}}

from benchmark.registry import (
    SCHEMA_VERSION,
    DimensionSpec,
    RunContext,
    collect_quality_leaves,
)
from benchmark.report import sections
from benchmark.rigor.multi_seed_runner import SeedRun, aggregate as aggregate_seed_runs
from benchmark.scenarios.runner import run_scenarios
from benchmark.scenarios.judge import select_judge_model as _select_judge_model
from benchmark.conversation_drift.runner import run_conversation_drift

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
GOLDEN = ROOT / "golden" / "expectations.json"
REPORTS = ROOT / "output" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def harness_version() -> str:
    """git short SHA；非 git 环境回退 env HARNESS_VERSION，再回退 'unknown'。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    import os
    return os.environ.get("HARNESS_VERSION") or "unknown"


def get_hardware_profile(model_cfg=None) -> dict:
    from benchmark.probe.hardware import HardwareProbe
    from common import load_targets
    target_name = getattr(model_cfg, "target", "local") or "local"
    targets = load_targets()
    target_cfg = targets.get(target_name)
    probe = HardwareProbe.for_target(target_cfg)
    profile = probe.collect()
    # vLLM version probe（保留现有逻辑）
    if model_cfg is not None and getattr(model_cfg, "port", 0):
        try:
            r = httpx.get(f"http://localhost:{model_cfg.port}/version", timeout=3.0)
            if r.status_code == 200:
                profile["vllm"] = str(r.json().get("version", "unknown"))
        except Exception:
            pass
    return profile


def _default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    # numpy scalar types from embedding/similarity computations
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


_YAML_CACHE: dict = {}


def _deep_merge(base: dict | None, override: dict | None) -> dict:
    """Recursively merge model-specific benchmark overrides onto global config."""
    merged = copy.deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _effective_bench_cfg(model_cfg: ModelConfig, bench_cfg: dict | None) -> dict:
    return _deep_merge(bench_cfg or {}, getattr(model_cfg, "benchmarks", None) or {})


def _effective_skip(model_cfg: ModelConfig, skip: set[str], bench_cfg: dict) -> set[str]:
    configured = bench_cfg.get("skip") or []
    if isinstance(configured, str):
        configured = [configured]
    return set(skip) | {str(item) for item in configured}


def _model_hint(model_cfg: ModelConfig, key: str) -> bool:
    """能力查询：优先 ModelConfig.capabilities（load_models 加载时派生，零 IO）；
    无 capabilities 的 stub/legacy 对象走缓存 yaml hint（一个 minor 后移除）。"""
    caps = getattr(model_cfg, "capabilities", None)
    if caps:
        from common import _HINT_TO_CAP
        return _HINT_TO_CAP.get(key, key) in caps
    try:
        import yaml

        path = ROOT / "models.yaml"
        if path not in _YAML_CACHE:
            _YAML_CACHE[path] = yaml.safe_load(path.read_text(encoding="utf-8"))
        for m in _YAML_CACHE[path].get("models", []):
            if m.get("name") == model_cfg.name:
                return bool(m.get(key, False))
    except Exception:
        pass
    return False


def _is_translation_capable(model_cfg: ModelConfig) -> bool:
    """读 models.yaml 的 translation_capable hint。"""
    return _model_hint(model_cfg, "translation_capable")


# Dimension wrappers:模块级 def + 体内引用全局名(call 时查 globals),
# 保证测试对 rb.run_ttft / rb._resolve_judge 等 seam 的 monkeypatch 仍生效。
def _run_accuracy_dim(m, c, ctx):
    return run_accuracy(m, ctx.golden, ctx.fixtures)


def _run_ttft_dim(m, c, ctx):
    return run_ttft(m, ctx.fixtures, samples=int(c.get("samples", 5)))


def _run_throughput_dim(m, c, ctx):
    return run_throughput(m, ctx.fixtures, duration_s=float(c.get("duration_s", 60.0)))


def _run_prefill_decode_dim(m, c, ctx):
    return run_prefill_decode(
        m,
        ctx.fixtures,
        samples=int(c.get("samples", 5)),
        decode_tokens=int(c.get("decode_tokens", 128)),
    )


def _run_concurrency_dim(m, c, ctx):
    return run_concurrency(
        m,
        ctx.fixtures,
        concurrencies=[int(v) for v in c.get("concurrencies", [1, 5, 10, 30, 50])],
        duration_s=float(c.get("duration_s", 60.0)),
    )


def _run_stability_dim(m, c, ctx):
    return run_stability(
        m,
        ctx.fixtures,
        duration_s=float(c.get("duration_s", 1800.0)),
        sample_interval_s=float(c.get("sample_interval_s", 5.0)),
    )


def _run_scenarios_dim(m, c, ctx):
    judge_cfg = _resolve_judge(c.get("judge_model"), m)
    return run_scenarios(m, judge_cfg=judge_cfg, cfg=c)


def _run_translation_dim(m, c, ctx):
    return run_translation_dimension(m, c, ctx.root)


def _run_embedding_dim(m, c, ctx):
    return run_embedding_dimension(m, c, ctx.root)


def _run_rerank_dim(m, c, ctx):
    return run_rerank_dimension(m, c, ctx.root)


def _run_asr_dim(m, c, ctx):
    return run_asr_dimension(m, c, ctx.root)


def _run_ocr_dim(m, c, ctx):
    return run_ocr_dimension(m, c, ctx.root)


def _run_general_ability_dim(m, c, ctx):
    return run_general_ability(m, c)


def _run_conditioned_dim(m, c, ctx):
    return run_conditioned(m, c, ctx.root)


def _run_long_context_dim(m, c, ctx):
    return run_long_context(m, c, ctx.root)


def _run_conversation_drift_dim(m, c, ctx):
    return run_conversation_drift(m, cfg=c)


def _parameter_size_b(model_cfg: ModelConfig) -> float:
    explicit = getattr(model_cfg, "parameter_size_b", None)
    if explicit is not None:
        return float(explicit)
    haystack = " ".join(
        str(x or "")
        for x in (
            getattr(model_cfg, "name", ""),
            getattr(model_cfg, "hf_repo", ""),
            getattr(model_cfg, "model_id", ""),
            getattr(model_cfg, "notes", ""),
        )
    )
    matches = re.findall(r"(?i)(\d+(?:\.\d+)?)\s*b\b", haystack)
    return max((float(m) for m in matches), default=0.0)


def _is_long_context_required(model_cfg: ModelConfig) -> bool:
    if not _is_chat_capable(model_cfg):
        return False
    benchmarks = getattr(model_cfg, "benchmarks", None) or {}
    dim_cfg = benchmarks.get("long_context") or {}
    if dim_cfg.get("required") is True:
        return True
    if dim_cfg.get("required") is False:
        return False
    return _parameter_size_b(model_cfg) >= 20.0


# 注册顺序 = dispatch 顺序 = 报告节顺序,不可乱。
DIMENSIONS: dict[str, DimensionSpec] = {
    "accuracy": DimensionSpec("accuracy", quality=True, run=_run_accuracy_dim,
                              render=sections.render_accuracy),
    "ttft": DimensionSpec("ttft", quality=False, run=_run_ttft_dim,
                          gate=_is_chat_capable, render=sections.render_ttft),
    "throughput": DimensionSpec("throughput", quality=False, run=_run_throughput_dim,
                                gate=_is_chat_capable, render=sections.render_throughput),
    "prefill_decode": DimensionSpec("prefill_decode", quality=False, run=_run_prefill_decode_dim,
                                    gate=_is_chat_capable, render=sections.render_prefill_decode),
    "concurrency": DimensionSpec("concurrency", quality=False, run=_run_concurrency_dim,
                                 gate=_is_chat_capable, render=sections.render_concurrency),
    "stability": DimensionSpec("stability", quality=False, run=_run_stability_dim,
                               gate=_is_chat_capable, render=sections.render_stability),
    "translation": DimensionSpec("translation", quality=True, run=_run_translation_dim,
                                 gate=_is_translation_capable,
                                 render=sections.render_translation),
    "embedding": DimensionSpec("embedding", quality=True, run=_run_embedding_dim,
                               gate=lambda m: _model_hint(m, "embedding_capable"),
                               render=sections.render_embedding),
    "rerank": DimensionSpec("rerank", quality=True, run=_run_rerank_dim,
                            gate=lambda m: _model_hint(m, "rerank_capable"),
                            render=sections.render_rerank),
    "asr": DimensionSpec("asr", quality=True, run=_run_asr_dim,
                         gate=lambda m: _model_hint(m, "asr_capable"),
                         render=sections.render_asr),
    "ocr": DimensionSpec("ocr", quality=True, run=_run_ocr_dim,
                         gate=lambda m: _model_hint(m, "ocr_capable"),
                         render=sections.render_ocr),
    "general_ability": DimensionSpec("general_ability", quality=True,
                                     run=_run_general_ability_dim, gate=_is_chat_capable,
                                     render=sections.render_general_ability),
    "conditioned": DimensionSpec("conditioned", quality=True,
                                 run=_run_conditioned_dim, gate=_is_chat_capable,
                                 render=sections.render_conditioned),
    "long_context": DimensionSpec("long_context", quality=True,
                                  run=_run_long_context_dim,
                                  gate=_is_long_context_required,
                                  render=sections.render_long_context),
    "scenarios": DimensionSpec(
        "scenarios", quality=True, run=_run_scenarios_dim, gate=_is_chat_capable,
        render=sections.render_scenarios),
    "conversation_drift": DimensionSpec(
        "conversation_drift", quality=True, run=_run_conversation_drift_dim,
        gate=_is_chat_capable, render=sections.render_conversation_drift),
}

# 质量维度（verdict 进 exit code；--seeds N 时也只对这些做 multi_seed 聚合）
QUALITY_DIMS = tuple(n for n, s in DIMENSIONS.items() if s.quality)


def run_all_for_model(
    model_cfg: ModelConfig,
    golden: dict,
    skip: set[str],
    bench_cfg: dict | None = None,
) -> dict:
    """对单个模型跑全量 benchmark"""
    global_bench_cfg = (
        bench_cfg if bench_cfg is not None else load_benchmarks_config(ROOT / "models.yaml")
    )
    bench_cfg = _effective_bench_cfg(model_cfg, global_bench_cfg)
    skip = _effective_skip(model_cfg, skip, bench_cfg)
    results: dict = {
        "model": model_cfg.name,
        "schema_version": SCHEMA_VERSION,
        "harness_version": harness_version(),
        "benchmark_config": bench_cfg,
        "hardware_profile": get_hardware_profile(model_cfg),
        "condition": {"context_tokens": None, "cache_mode": None},
        "hf_repo": model_cfg.hf_repo,
        "quantization": model_cfg.quantization,
        "hardware_min": model_cfg.hardware_min,
        "timestamp": datetime.datetime.now().isoformat(),
        "vram_snapshot": get_vram_info(),
        "benchmarks": {},
    }

    # 预检：端点是否就绪（port=0 = 本地 ONNX 推理，无 HTTP 端点，跳过轮询）
    if model_cfg.port != 0:
        logger.info("检查 %s (port %d) 就绪...", model_cfg.name, model_cfg.port)
        if not wait_model_ready(model_cfg, timeout_s=60.0):
            logger.error("  %s 未就绪，跳过", model_cfg.name)
            results["error"] = "model_not_ready"
            return results
    else:
        logger.info("检查 %s — ONNX 本地推理，跳过端点探测", model_cfg.name)

    ctx = RunContext(root=ROOT, fixtures=FIXTURES, golden=golden, bench_cfg=bench_cfg)
    for name, spec in DIMENSIONS.items():
        if name in skip or not spec.gate(model_cfg):
            continue
        if spec.requires:
            skip_dim = False
            for req in spec.requires:
                # Only block if the required dim actually ran and produced a bad verdict.
                # If req was in the user's skip set or gated out (not in benchmarks), allow.
                req_verdict = results["benchmarks"].get(req, {}).get("verdict")
                if req_verdict in ("FAIL", "BLOCKED"):
                    results["benchmarks"][name] = {
                        "verdict": "SKIPPED",
                        "reason": f"requires {req}",
                    }
                    skip_dim = True
                    break
            if skip_dim:
                continue
        logger.info("▶ %s", name)
        block = spec.run(model_cfg, bench_cfg.get(name, {}) or {}, ctx)
        if block is not None:
            results["benchmarks"][name] = block

    results["vram_after"] = get_vram_info()
    return results


def _resolve_judge(judge_name: str | None, model_cfg: ModelConfig):
    """按名取 judge 模型;未配置则取被测之外最大的 text 模型(排除
    embedding/rerank/asr/无端点)。选中后做 liveness 探测,端点未就绪 →
    None(runner 降级 L1-only),避免对死端点白打 18 次校准 call。"""
    models = load_models(ROOT / "models.yaml")
    if judge_name:
        judge = next((m for m in models if m.name == judge_name), None)
        if judge is None:
            logger.warning(
                "scenarios judge_model %r not in models.yaml — L2 disabled",
                judge_name)
            return None
    else:
        pool = [
            m for m in models
            if m.name != model_cfg.name and m.port and _is_chat_capable(m)
        ]
        target = getattr(model_cfg, "target", None)
        if target:
            target_pool = [m for m in pool if getattr(m, "target", None) == target]
            if target_pool:
                pool = target_pool
        ready_pool = [m for m in pool if wait_model_ready(m, timeout_s=2.0)]
        if not ready_pool:
            logger.warning("no ready scenarios judge candidates — L2 disabled")
            return None
        if not pool:
            return None
        # Use priority-based selection: 7B→14B→3B→1.5B→0.6B→first-available
        try:
            judge = _select_judge_model(ready_pool)
        except RuntimeError:
            return None
    if not wait_model_ready(judge, timeout_s=10.0):
        logger.warning(
            "scenarios judge %s (port %s) not ready — L2 disabled",
            judge.name, judge.port)
        return None
    return judge


def aggregate_multi_seed(
    seed_runs: list[dict], durations: list[float] | None = None
) -> dict:
    """跨 N 个 run_all_for_model 结果，对质量维度的数值指标做 mean/std/ci95。

    只聚合在所有 N 个 run 里都出现的 dotted path（缺席任一 run 的指标
    不可比，丢弃）。verdict 等非数值字段不进来 — verdict 永远取最差，
    不取平均（见 main()）。"""
    per_run: list[dict[str, float]] = [
        collect_quality_leaves(r, QUALITY_DIMS) for r in seed_runs
    ]
    common = set(per_run[0])
    for lv in per_run[1:]:
        common &= set(lv)
    # 计数器启发式：n_cases / violations / requests 等计数器是整数值，
    # 跨 seed 求 mean/std 没有意义，还会把真正的质量指标 (BLEU/chrF/recall
    # — 分数值) 挤出 markdown 的 15 行上限。只丢「每个 seed 都是精确整数
    # 且 max>1」的叶子 — 稳定 1.0 的 recall/accuracy 是 [0,1] 质量指标，
    # 必须存活（bool 已在 _numeric_leaves 排除）。
    common = {k for k in common
              if not (all(lv[k].is_integer() for lv in per_run)
                      and max(lv[k] for lv in per_run) > 1)}
    runs = [
        SeedRun(
            seed=i,
            metrics={k: lv[k] for k in common},
            duration_s=durations[i] if durations else 0.0,
        )
        for i, lv in enumerate(per_run)
    ]
    aggs = aggregate_seed_runs(runs)
    out = {
        "n_seeds": len(seed_runs),
        "metrics": {
            path: {
                "mean": a.mean,
                "std": a.std,
                "ci95_lower": a.ci95_lower,
                "ci95_upper": a.ci95_upper,
            }
            for path, a in sorted(aggs.items())
        },
        # 证据保全（§6.3）：merged 报告必须记录是哪个 seed 出了什么
        # verdict / error，否则 seed 1..N-1 的 FAIL 在归档里不可见。
        "per_seed": [
            {
                "seed": i,
                "verdict_summary": {
                    dim: block.get("verdict")
                    for dim in QUALITY_DIMS
                    if isinstance(
                        block := (r.get("benchmarks") or {}).get(dim), dict
                    )
                },
                "error": r.get("error"),
                "duration_s": durations[i] if durations else 0.0,
            }
            for i, r in enumerate(seed_runs)
        ],
    }
    if not out["metrics"] or any(not lv for lv in per_run):
        out["warning"] = (
            "metric intersection empty — no fractional quality metric appeared "
            "in all N seed runs (a seed produced no measurements, or the shared "
            "leaves were all integer counters)"
        )
    return out


def render_markdown(result: dict) -> str:
    """单模型报告 Markdown(头部 + 注册表 render 钩子 + multi-seed 节)。"""
    m = result["model"]
    bm = result.get("benchmarks", {})
    vram = result.get("vram_after", {})

    lines = [
        f"# {m} Benchmark",
        "",
        f"- HF repo: `{result.get('hf_repo','')}`",
        f"- Quantization: `{result.get('quantization')}`",
        f"- Hardware min: {result.get('hardware_min','')}",
        f"- Time: {result.get('timestamp','')}",
        f"- VRAM after run: {vram.get('used_mb','?')}MB / {vram.get('total_mb','?')}MB"
        if vram else "- VRAM: 未采集",
        "",
    ]
    for name, spec in DIMENSIONS.items():
        if spec.render is not None:
            lines += spec.render(bm.get(name) or {})

    ms = result.get("multi_seed") or {}
    if ms:
        metrics = ms.get("metrics", {})
        lines += ["", f"## Multi-seed (N={ms.get('n_seeds')})", "",
                  "质量指标跨 run mean ± std（verdict 取最差，不取平均）:", ""]
        if ms.get("warning"):
            lines.append(f"- ⚠ {ms['warning']}")
        for i, (path, st) in enumerate(metrics.items()):
            if i >= 15:
                lines.append(f"- …其余 {len(metrics) - 15} 项见 JSON 报告")
                break
            lines.append(f"- {path}: {st['mean']:.4f} ± {st['std']:.4f} "
                         f"(CI95 [{st['ci95_lower']:.4f}, {st['ci95_upper']:.4f}])")
    return "\n".join(lines)


def render_matrix(all_results: list[dict]) -> str:
    """多模型横向对比"""
    lines = [
        "# 模型矩阵对比",
        "",
        "| Model | 分类 precision | 实体 recall | TTFT P95 | TPS | 并发 50 成功率 | 稳定性漂移 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        name = r["model"]
        bm = r.get("benchmarks", {})
        acc = bm.get("accuracy", {}).get("aggregate", {}) if bm.get("accuracy") else {}
        ttft_p95 = bm.get("ttft", {}).get("ttft_ms_stats", {}).get("p95", 0)
        tps = bm.get("throughput", {}).get("aggregate_tps", 0)
        con_steps = bm.get("concurrency", {}).get("steps", [])
        con_50 = next((s for s in con_steps if s.get("concurrency") == 50), {})
        drift = bm.get("stability", {}).get("latency_drift_ratio", 0)

        lines.append(
            f"| {name} "
            f"| {acc.get('category_precision',0)*100:.1f}% "
            f"| {acc.get('entity_recall',0)*100:.1f}% "
            f"| {ttft_p95:.0f}ms "
            f"| {tps:.1f} "
            f"| {con_50.get('success_rate',0)*100:.1f}% "
            f"| {drift:.2f}× |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="模型名（见 models.yaml）或 all")
    parser.add_argument("--skip", default="",
                        help="逗号分隔跳过的 benchmark: accuracy,ttft,throughput,"
                             "prefill_decode,concurrency,stability,translation,"
                             "embedding,rerank,asr,general_ability,conditioned,"
                             "long_context,scenarios")
    parser.add_argument("--golden", default=str(GOLDEN))
    parser.add_argument("--seeds", type=int, default=1,
                        help="每模型完整重跑次数 N（默认 1 = 单次，行为不变）。"
                             "N>1 时报告含 multi_seed mean/std/ci95，"
                             "verdict 取 N 次中最差（per §2.3 多 seed 纪律）")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="离线对比两模型已存报告,输出 "
                             "REPLACEABLE/NOT_REPLACEABLE/INCONCLUSIVE")
    parser.add_argument("--native", action="store_true",
                        help="Use C++ native tools (llama-bench/wrk) for perf dims when available")
    parser.add_argument("--target", default="local",
                        help="Target ID from targets.yaml (default: local). "
                             "Use 'amd-win-x86', 'rk3588-linux', etc.")
    parser.add_argument("--local-only", action="store_true",
                        help="Internal flag: run only local dims without re-dispatching "
                             "(set by RemoteExecutor)")
    parser.add_argument("--install-first", action="store_true",
                        help="For remote --target runs, install Python requirements on "
                             "the target before running benchmarks")
    args = parser.parse_args()

    # Remote dispatch: if target is non-local and --local-only not set, hand off to RemoteExecutor.
    if not args.local_only:
        from common import load_targets
        from benchmark.executor.remote import RemoteExecutor
        targets = load_targets()
        target_name = args.target or "local"
        target_cfg = targets.get(target_name)
        if target_cfg and not target_cfg.is_local():
            def _extra_args(a) -> list[str]:
                extra: list[str] = ["--target", target_name]
                if a.skip:
                    extra += ["--skip", a.skip]
                if a.seeds > 1:
                    extra += ["--seeds", str(a.seeds)]
                if a.native:
                    extra.append("--native")
                return extra
            executor = RemoteExecutor(target_cfg)
            if args.model == "all":
                selected = [
                    m for m in load_models(ROOT / "models.yaml")
                    if (getattr(m, "target", None) or "local") == target_name
                ]
                if not selected:
                    logger.error("target %s 没有匹配模型", target_name)
                    return 2
                had_error = False
                for model in selected:
                    try:
                        executor.run_benchmark(
                            model.name,
                            _extra_args(args),
                            install_first=args.install_first,
                            raise_on_error=False,
                        )
                    except Exception as exc:
                        had_error = True
                        logger.error("%s remote benchmark failed: %s", model.name, exc)
                    last_error = getattr(executor, "last_error", None)
                    if last_error:
                        had_error = True
                        logger.error(
                            "%s remote benchmark failed: %s",
                            model.name,
                            last_error,
                        )
                if had_error:
                    return 2
            else:
                executor.run_benchmark(
                    args.model, _extra_args(args), install_first=args.install_first)
            return 0

    if args.compare:
        from benchmark.compare import run_compare
        bench_cfg = load_benchmarks_config(ROOT / "models.yaml")
        return run_compare(args.compare[0], args.compare[1], REPORTS, bench_cfg)
    if args.seeds < 1:
        logger.error("--seeds 必须 ≥ 1，得到 %d", args.seeds)
        return 2

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    unknown = skip - set(DIMENSIONS)
    if unknown:
        logger.error("--skip 含未知维度: %s。可选: %s",
                      ", ".join(sorted(unknown)), ", ".join(DIMENSIONS))
        return 2

    bench_cfg = load_benchmarks_config(ROOT / "models.yaml")
    models = load_models(ROOT / "models.yaml")
    if args.model == "all" and (args.target or "local") != "local":
        target_name = args.target or "local"
        models = [
            m for m in models
            if (getattr(m, "target", None) or "local") == target_name
        ]
        if not models:
            logger.error("target %s 没有匹配模型", target_name)
            return 2
    elif args.model != "all":
        models = [m for m in models if m.name == args.model]
        if not models:
            logger.error("未知模型: %s。可选: %s", args.model,
                         [m.name for m in load_models(ROOT / 'models.yaml')])
            return 2

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results: list[dict] = []
    has_fail = False
    has_warn = False
    measured_any = False

    for m in models:
        # --seeds N: 完整重跑 N 次（不传 per-call seed — run 间方差来源是
        # 采样温度噪声）。报告主体 = 第 1 次 run + multi_seed 聚合。
        seed_runs: list[dict] = []
        durations: list[float] = []
        for run_idx in range(args.seeds):
            if args.seeds > 1:
                logger.info("═══ %s (run %d/%d) ═══", m.name, run_idx + 1, args.seeds)
            else:
                logger.info("═══ %s ═══", m.name)
            t0 = time.monotonic()
            seed_runs.append(run_all_for_model(m, golden, skip, bench_cfg))
            durations.append(time.monotonic() - t0)
        stem = f"{m.name}_{timestamp}"
        if args.seeds > 1:
            # 证据保全（§6.3）：seed 1..N-1 的 raw 结果不能只活在内存里 —
            # 每个 seed 单独归档（在 multi_seed 注入前写，保持 raw 形态）。
            # {stem}.json/.md 仍是 seed-0 + multi_seed 的 merged 报告。
            for k, r in enumerate(seed_runs):
                (REPORTS / f"{stem}_seed{k}.json").write_text(
                    json.dumps(r, indent=2, ensure_ascii=False, default=_default),
                    encoding="utf-8",
                )
        result = seed_runs[0]
        if args.seeds > 1:
            result["multi_seed"] = aggregate_multi_seed(seed_runs, durations)
        all_results.append(result)

        # 判定（质量维度任一 FAIL 即整体 FAIL）。
        # --model all 的契约是「跑所有已启动的模型」:未启动 = 跳过不算 FAIL
        # (models.yaml 里 port=0 的非 HTTP 模型永远探活失败)。但用户点名的
        # 模型出 error 必须 FAIL;以及整轮零实测时绝不允许 exit 0 假装全过。
        # 多 seed 时取 N 次中最差 — verdict 不取平均,任一 run FAIL 即 FAIL。
        for r in seed_runs:
            if r.get("benchmarks"):
                measured_any = True
            if r.get("error") and args.model != "all":
                has_fail = True
            for dim in QUALITY_DIMS:
                verdict = r.get("benchmarks", {}).get(dim, {}).get("verdict")
                if verdict == "FAIL":
                    has_fail = True
                elif verdict in ("WARN", "BLOCKED"):
                    has_warn = True

        # 保存单模型报告
        (REPORTS / f"{stem}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=_default),
            encoding="utf-8",
        )
        (REPORTS / f"{stem}.md").write_text(render_markdown(result), encoding="utf-8")
        from benchmark.report.html_report import generate_html
        (REPORTS / f"{stem}.html").write_text(generate_html(result), encoding="utf-8")
        logger.info("报告保存: %s", REPORTS / f"{stem}.md")

    # 矩阵报告
    if len(all_results) > 1:
        matrix_md = render_matrix(all_results)
        (REPORTS / f"matrix_{timestamp}.md").write_text(matrix_md, encoding="utf-8")
        print("\n" + matrix_md)

    if not measured_any:
        logger.error("没有任何模型产出实测数据 — 空跑不允许报成功")
        has_fail = True
    return 2 if has_fail else (1 if has_warn else 0)


if __name__ == "__main__":
    sys.exit(main())
