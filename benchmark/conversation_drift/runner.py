"""Conversation drift dimension: measures quality stability as conversation grows.

Protocol:
  For each registered scenario and a subset of its cases:
    1. Run the case with 0 prior turns (baseline)
    2. Run the case with 5 prior turns (filler conversation)
    3. Run the case with 10 prior turns
    4. Run the case with 20 prior turns

  Primary metric per position: same L1 metric as the scenario uses.
  Drift verdict:
    - STABLE: max quality drop across positions ≤ 0.05
    - WARN: max quality drop 0.05–0.15
    - DRIFT: max quality drop > 0.15 (not suitable for long-session deployment)
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Optional

from common import ModelConfig, infer_sync, InferResult

from benchmark.scenarios import SCENARIOS
from benchmark.scenarios.base import load_cases

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FILLER_PATH = ROOT / "datasets/conversation_drift/filler_turns.jsonl"

POSITIONS = [0, 5, 10, 20]  # number of prior Q&A turns


def _primary_metric(spec) -> str | None:
    """Derive primary L1 metric key from a ScenarioSpec's default_thresholds."""
    for key in spec.default_thresholds:
        return key.removesuffix("_min")
    return None
MAX_CASES_PER_SCENARIO = 5   # keep drift run fast; 5 × 4 positions × N scenarios


def _load_filler() -> list[dict]:
    """Load neutral filler Q&A pairs for building conversation history."""
    if not FILLER_PATH.exists():
        return []
    turns = []
    for line in FILLER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            turns.append(json.loads(line))
    return turns


def _build_prior_messages(filler: list[dict], n_turns: int) -> list[dict]:
    """Build n_turns of prior conversation messages from filler corpus, cycling if needed."""
    if not filler or n_turns <= 0:
        return []
    msgs = []
    for i in range(n_turns):
        turn = filler[i % len(filler)]
        msgs.append({"role": "user", "content": turn["q"]})
        msgs.append({"role": "assistant", "content": turn["a"]})
    return msgs


def run_conversation_drift(model_cfg: ModelConfig, *, cfg: dict) -> dict:
    """Run conversation drift evaluation and return result block."""
    filler = _load_filler()
    if not filler:
        return {
            "verdict": "BLOCKED",
            "verdict_reasons": [f"filler corpus missing: {FILLER_PATH}"],
        }

    num_cases = cfg.get("num_cases") or MAX_CASES_PER_SCENARIO
    results: dict[str, dict] = {}

    for scenario_name, spec in SCENARIOS.items():
        # Skip VLM scenarios (multi-turn + image is complex; defer to v.next)
        if spec.requires_vlm:
            results[scenario_name] = {"verdict": "SKIPPED", "reason": "VLM scenario — skip drift"}
            continue
        # Skip scenarios with no quality metric
        primary_key = _primary_metric(spec)
        if primary_key is None:
            results[scenario_name] = {"verdict": "SKIPPED", "reason": "no primary metric"}
            continue

        cases_path = ROOT / spec.cases_path
        if not cases_path.exists():
            results[scenario_name] = {"verdict": "BLOCKED", "reason": f"cases missing: {cases_path}"}
            continue

        # load_cases returns None when file is missing; num_samples limits count
        cases = load_cases(cases_path, num_samples=min(num_cases, MAX_CASES_PER_SCENARIO))
        if cases is None:
            results[scenario_name] = {"verdict": "BLOCKED", "reason": "no cases loaded"}
            continue

        quality_by_position: dict[int, list[float]] = {p: [] for p in POSITIONS}
        errors = 0

        for case in cases:
            for n_turns in POSITIONS:
                prior_msgs = _build_prior_messages(filler, n_turns)
                try:
                    prompt, image = spec.build_prompt(case)
                    r: InferResult = infer_sync(
                        model_cfg,
                        prompt=prompt,
                        prior_messages=prior_msgs,
                        max_tokens=800,
                    )
                except Exception as exc:
                    logger.warning("drift %s case %s pos %d: %s", scenario_name, case.id, n_turns, exc)
                    errors += 1
                    continue

                if not r.ok:
                    errors += 1
                    continue

                try:
                    score = spec.l1_score(case, r.parsed_json, r.content)
                    if primary_key in score:
                        quality_by_position[n_turns].append(score[primary_key])
                except Exception as exc:
                    logger.warning("drift %s case %s pos %d: l1_score error: %s",
                                   scenario_name, case.id, n_turns, exc)
                    errors += 1

        # Compute per-position means
        pos_means: dict[int, Optional[float]] = {}
        for pos, vals in quality_by_position.items():
            pos_means[pos] = statistics.mean(vals) if vals else None

        # Compute drift: quality drop from position 0 baseline
        baseline = pos_means.get(0)
        max_drop = 0.0
        if baseline is not None:
            for pos in POSITIONS[1:]:
                if pos_means.get(pos) is not None:
                    drop = baseline - pos_means[pos]  # positive = degraded
                    max_drop = max(max_drop, drop)

        # Drift verdict per scenario
        if baseline is None:
            s_verdict = "BLOCKED"
            s_reasons = ["no baseline measurement at position 0"]
        elif max_drop > 0.15:
            s_verdict = "DRIFT"
            s_reasons = [f"quality drops {max_drop:.2%} over long conversation (threshold 15%)"]
        elif max_drop > 0.05:
            s_verdict = "WARN"
            s_reasons = [f"quality drops {max_drop:.2%} — monitor in production"]
        else:
            s_verdict = "STABLE"
            s_reasons = []

        results[scenario_name] = {
            "verdict": s_verdict,
            "verdict_reasons": s_reasons,
            "quality_by_position": {str(p): round(v, 4) if v is not None else None
                                     for p, v in pos_means.items()},
            "max_quality_drop": round(max_drop, 4),
            "primary_metric": primary_key,
            "cases_tested": len(cases),
            "errors": errors,
        }

    # Overall verdict: worst across scenarios (DRIFT > WARN > STABLE)
    _order = {"DRIFT": 3, "WARN": 2, "STABLE": 1, "BLOCKED": 2, "SKIPPED": 0}
    overall = max(results.values(), key=lambda v: _order.get(v.get("verdict", "SKIPPED"), 0))
    overall_verdict = overall.get("verdict", "STABLE")
    # Normalize to standard verdict vocabulary
    verdict_map = {"DRIFT": "FAIL", "WARN": "WARN", "STABLE": "PASS",
                   "BLOCKED": "WARN", "SKIPPED": "PASS"}

    return {
        "benchmark": "conversation_drift",
        "model": model_cfg.name,
        "positions_tested": POSITIONS,
        "per_scenario": results,
        "overall_drift_verdict": overall_verdict,
        "verdict": verdict_map.get(overall_verdict, "WARN"),
        "verdict_reasons": [r for v in results.values()
                            for r in v.get("verdict_reasons", [])],
    }
