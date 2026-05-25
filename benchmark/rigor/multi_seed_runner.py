"""Multi-seed run orchestration with rank-stability detection.

Why this exists
---------------
Single-seed benchmark comparisons are statistical theater: the noise of
sampling, randomized batching, dropout, retrieval ties, and judge wobble
each contribute variance that can flip the ranking between two systems.
The CLAUDE.md "调研/算法项目工作纪律" §7 rule is uncompromising:

    "任何被选作 SOTA / 胜出 / 新方向的候选必须用 >=3 seed 复跑;
     报告 mean +- std, 不报单一数字; 改进 < 2 sigma 不算改进;
     seed 间排名翻转 -> 撤回原 SOTA claim"

This module enforces that contract. A caller registers a `run_fn(seed)`
which returns a metric dict; we run it N times under controlled seeds,
aggregate, and flag rank flips.

The runner also writes a structured manifest (run id, seed list, mean,
std, CI, rank order) that can be archived under reports/runs/<ts>/ per
the project's Baseline SOP.

References
----------
- Henderson, P. et al. (2018). Deep Reinforcement Learning that Matters.
  AAAI. Famous illustration of seed-fragility.
- Bouthillier, X. et al. (2021). Accounting for Variance in Machine
  Learning Benchmarks. MLSys.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SeedRun:
    seed: int
    metrics: Dict[str, float]
    duration_s: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateResult:
    metric: str
    values: List[float]
    mean: float
    std: float
    median: float
    p25: float
    p75: float
    ci95_lower: float
    ci95_upper: float
    n_seeds: int

    def as_report_line(self) -> str:
        return (
            f"{self.metric}: mean={self.mean:.4f} std={self.std:.4f} "
            f"median={self.median:.4f} CI95=[{self.ci95_lower:.4f}, "
            f"{self.ci95_upper:.4f}] (n={self.n_seeds})"
        )


@dataclass
class RankFlipReport:
    """Detection of seed-induced rank instability across systems."""

    metric: str
    per_seed_ranking: List[List[str]]
    canonical_ranking: List[str]
    flips_observed: int
    fraction_flips: float
    stable: bool  # True iff every seed produced the same ordering

    def summary(self) -> str:
        return (
            f"metric={self.metric} stable={self.stable} "
            f"flips={self.flips_observed}/{len(self.per_seed_ranking)} "
            f"canonical={self.canonical_ranking}"
        )


# ---------------------------------------------------------------------------
# Seed-pinning helper used by `run`.
# Sets python's random, numpy, PYTHONHASHSEED.
# Callers using torch/tf should hook those into their own run_fn.
# ---------------------------------------------------------------------------


def pin_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _ci95_from_samples(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        v = float(values[0]) if values else 0.0
        return v, v
    # Use t-based interval since n is small.
    from scipy import stats as _stats

    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    if sem == 0:
        return mean, mean
    t_val = float(_stats.t.ppf(0.975, df=arr.size - 1))
    return mean - t_val * sem, mean + t_val * sem


# ---------------------------------------------------------------------------
# Single-system multi-seed runner
# ---------------------------------------------------------------------------


def run_multi_seed(
    run_fn: Callable[[int], Dict[str, float]],
    seeds: Sequence[int] = (0, 1, 2),
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[SeedRun]:
    """Invoke `run_fn` once per seed, returning the raw seed-by-seed records.

    `run_fn` must accept a seed integer, set its own framework seeds (in
    addition to what pin_seeds covers), and return a flat dict of
    {metric_name: float}.
    """
    runs: List[SeedRun] = []
    n = len(seeds)
    for i, seed in enumerate(seeds):
        pin_seeds(seed)
        t0 = time.perf_counter()
        metrics = run_fn(seed)
        t1 = time.perf_counter()
        if not isinstance(metrics, dict):
            raise TypeError(
                f"run_fn must return Dict[str, float], got {type(metrics).__name__}"
            )
        runs.append(SeedRun(seed=int(seed), metrics=dict(metrics), duration_s=t1 - t0))
        if progress_cb:
            progress_cb(i + 1, n)
    return runs


def aggregate(runs: Sequence[SeedRun]) -> Dict[str, AggregateResult]:
    """Per-metric aggregation across seeds."""
    if not runs:
        return {}
    keys = sorted({k for r in runs for k in r.metrics.keys()})
    out: Dict[str, AggregateResult] = {}
    for k in keys:
        vals = [r.metrics[k] for r in runs if k in r.metrics]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        lo, hi = _ci95_from_samples(vals)
        out[k] = AggregateResult(
            metric=k,
            values=[float(v) for v in vals],
            mean=float(arr.mean()),
            std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            median=float(np.median(arr)),
            p25=float(np.quantile(arr, 0.25)),
            p75=float(np.quantile(arr, 0.75)),
            ci95_lower=float(lo),
            ci95_upper=float(hi),
            n_seeds=arr.size,
        )
    return out


def two_sigma_significant(
    agg_a: AggregateResult,
    agg_b: AggregateResult,
) -> bool:
    """Per CLAUDE.md rule: improvement < 2 sigma doesn't count.

    We define sigma as the pooled std across seeds. Returns True iff the
    mean gap is at least 2 * typical_std.
    """
    typical_std = max(agg_a.std, agg_b.std)
    if typical_std == 0:
        return agg_a.mean != agg_b.mean
    return abs(agg_a.mean - agg_b.mean) >= 2 * typical_std


# ---------------------------------------------------------------------------
# Multi-system comparison + rank-flip detection
# ---------------------------------------------------------------------------


def detect_rank_flips(
    system_runs: Dict[str, Sequence[SeedRun]],
    metric: str,
    higher_is_better: bool = True,
) -> RankFlipReport:
    """Cross-system rank stability check.

    `system_runs` maps system_id -> list of SeedRun (one per seed). All
    systems must share the same seed set, otherwise the comparison is
    apples-to-oranges. Returns a report flagging any seed at which the
    ranking diverged from the seed-mean canonical ordering.
    """
    if not system_runs:
        raise ValueError("system_runs is empty")
    # Sanity: same seeds across all systems.
    seed_sets = {tuple(sorted(r.seed for r in runs)) for runs in system_runs.values()}
    if len(seed_sets) != 1:
        raise ValueError(
            "all systems must share identical seed sets; got: " + repr(seed_sets)
        )

    systems = list(system_runs.keys())
    seeds = sorted(next(iter(system_runs.values())), key=lambda r: r.seed)
    n_seeds = len(seeds)

    # Per-seed ranking.
    per_seed_ranking: List[List[str]] = []
    seed_idx_map = {r.seed: i for i, r in enumerate(seeds)}
    matrix = np.zeros((len(systems), n_seeds), dtype=float)
    for s_idx, sys in enumerate(systems):
        for r in system_runs[sys]:
            if metric not in r.metrics:
                raise KeyError(f"system {sys} seed {r.seed} missing metric {metric}")
            matrix[s_idx, seed_idx_map[r.seed]] = r.metrics[metric]

    for col in range(n_seeds):
        scores = list(zip(systems, matrix[:, col]))
        scores.sort(key=lambda x: x[1], reverse=higher_is_better)
        per_seed_ranking.append([s for s, _ in scores])

    # Canonical: by mean across seeds.
    means = [(sys, float(matrix[i].mean())) for i, sys in enumerate(systems)]
    means.sort(key=lambda x: x[1], reverse=higher_is_better)
    canonical = [s for s, _ in means]

    flips = sum(1 for r in per_seed_ranking if r != canonical)
    return RankFlipReport(
        metric=metric,
        per_seed_ranking=per_seed_ranking,
        canonical_ranking=canonical,
        flips_observed=flips,
        fraction_flips=flips / n_seeds,
        stable=(flips == 0),
    )


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


def write_manifest(
    output_dir: Path,
    runs: Sequence[SeedRun],
    aggregates: Dict[str, AggregateResult],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a JSON manifest of the multi-seed run.

    Layout (under reports/runs/<ts>/):
        manifest.json   # everything machine-readable
        report.txt      # human-readable summary

    Returns the manifest path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "timestamp_unix": time.time(),
        "n_seeds": len(runs),
        "seeds": [r.seed for r in runs],
        "runs": [asdict(r) for r in runs],
        "aggregates": {k: asdict(v) for k, v in aggregates.items()},
        "extra": extra or {},
    }
    p = output_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [f"# Multi-seed run manifest", f"seeds = {[r.seed for r in runs]}"]
    for agg in aggregates.values():
        report_lines.append(agg.as_report_line())
    (output_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    return p
