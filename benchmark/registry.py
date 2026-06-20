"""DimensionSpec 注册表 + RunContext + 共享 verdict 库。

把 SCENARIOS(ScenarioSpec dict)的注册表 idiom 推广到维度层:每个维度 =
name + quality 标记 + gate(能力门)+ run(执行)+ render(markdown 节)。
verdict 语义全仓单源:PASS/SKIPPED < WARN/BLOCKED < FAIL,worst-of 合并。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# 报告 schema 版本单源(run_benchmark 写入 / compare 校验共用;放 registry
# 避免 compare → run_benchmark 的 import 环)。
SCHEMA_VERSION = 1

_RANK = {"PASS": 0, "SKIPPED": 0, "WARN": 1, "BLOCKED": 1, "FAIL": 2}


def worst_verdict(verdicts) -> str:
    worst = max((_RANK[v] for v in verdicts), default=0)
    return {0: "PASS", 1: "WARN", 2: "FAIL"}[worst]


def cap_warn(verdict: str) -> str:
    return "WARN" if verdict == "PASS" else verdict


def _gate_open(model_cfg) -> bool:
    return True


def _numeric_leaves(node, prefix: str = "") -> dict[str, float]:
    """递归收集 dict 树上的数值叶子（dotted path → float）。

    只下钻 dict（list 内容样本数不定，跨 run 不可对齐，跳过）；bool 是
    int 子类但不是 metric，排除。"""
    out: dict[str, float] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.update(_numeric_leaves(v, path))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out[prefix] = float(node)
    return out


def collect_quality_leaves(result: dict, quality_dims) -> dict[str, float]:
    """一份 run_all_for_model 结果 → 质量维度数值叶子(dotted path → float)。"""
    leaves: dict[str, float] = {}
    for dim in quality_dims:
        block = (result.get("benchmarks") or {}).get(dim)
        if isinstance(block, dict):
            leaves.update(_numeric_leaves(block, dim))
    return leaves


@dataclass(frozen=True)
class RunContext:
    """一次 run_all_for_model 的环境(arch review P0.4 的 v0.3 最小型)。"""

    root: Path
    fixtures: Path
    golden: dict
    bench_cfg: dict


@dataclass(frozen=True)
class DimensionSpec:
    """一个 benchmark 维度。run(model_cfg, dim_cfg, ctx) -> block dict | None。"""

    name: str
    quality: bool                                   # verdict 进 exit code + multi-seed 聚合
    run: Callable
    gate: Callable = field(default=_gate_open)      # False → 不 dispatch(无 block)
    render: Optional[Callable] = None               # (block) -> list[str] markdown 行
    requires: tuple = ()                            # 必须先通过的维度 key（SKIPPED/FAIL/BLOCKED 则跳过本维度）
