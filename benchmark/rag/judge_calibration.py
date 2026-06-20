"""LLM-as-judge calibration (PDF Chapter 8).

Even a thoughtfully-prompted judge has biases:
- Position bias: prefers the first answer shown in pairwise comparison.
- Verbosity bias: prefers longer answers regardless of quality.
- Self-preference: prefers answers from its own family (Zheng 2023).
- Sycophancy: agrees with prior reasoning steps shown in CoT.

Calibration buys evidence that the judge is fit for purpose by:
1. Replaying known-good and known-bad answer pairs and confirming
   high-quality rate >>> low-quality rate.
2. Replaying the same items multiple times and confirming stability.
3. Running adversarial controls (swap A/B, vary length, vary tokens that
   indicate provenance) and checking that the verdict does not flip.

This module orchestrates those experiments and reports the diagnostics.
Statistics live in ../rigor/; this module is the calibration *experiment
runner*.

References
----------
- Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena. NeurIPS. (position / verbosity / self-preference biases)
- Wang, P. et al. (2024). Large Language Models are not Fair Evaluators.
- Saito, K. et al. (2023). Verbosity Bias in Preference Labeling by LLMs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..rigor.calibration import expected_calibration_error


@dataclass(frozen=True)
class GoldPair:
    """A known-good / known-bad pair for calibration."""

    pair_id: str
    question: str
    good_answer: str
    bad_answer: str
    evidence: List[Dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrationReport:
    n_pairs: int
    n_runs_per_pair: int
    accuracy: float
    consistency: float  # fraction of pairs where all N runs agree
    position_bias: float  # |P(picks A | A=good) - P(picks B | B=good)| signed
    verbosity_correlation: float  # spearman between answer-length-delta and verdict
    self_preference: Optional[float]
    parse_failure_rate: float


# ---------------------------------------------------------------------------
# Pairwise replay with A/B swap
# ---------------------------------------------------------------------------


def replay_calibration_pairs(
    pairs: Sequence[GoldPair],
    judge_fn: Callable[[GoldPair, bool], List[Dict[str, Any]]],
    n_runs_per_pair: int = 3,
    swap_order: bool = True,
) -> List[Dict[str, Any]]:
    """Run the judge on each pair `n_runs_per_pair` times, optionally also
    with A/B swapped (so we get 2 * n_runs records per pair).

    `judge_fn(pair, good_is_A) -> [run_output_dict, ...]` is the caller's
    bridge to whatever invoke_fn they're using. Each run_output should
    contain at minimum a "winner" key in {"A", "B", "tie"}.
    """
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        # Original order: A = good, B = bad.
        for out in judge_fn(pair, True):
            rows.append({
                "pair_id": pair.pair_id,
                "good_was_A": True,
                "verdict": out.get("winner"),
                "parse_error": "_parse_error" in out,
                "raw": out,
            })
        if swap_order:
            for out in judge_fn(pair, False):
                rows.append({
                    "pair_id": pair.pair_id,
                    "good_was_A": False,
                    "verdict": out.get("winner"),
                    "parse_error": "_parse_error" in out,
                    "raw": out,
                })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def calibration_report(
    rows: Sequence[Dict[str, Any]],
    pairs: Sequence[GoldPair],
    n_runs_per_pair: int,
    self_label: Optional[str] = None,
) -> CalibrationReport:
    """Compute headline diagnostics from the replay rows.

    `self_label` is the judge's own family tag (e.g. "gpt-4"); if any
    pair has a "produced_by" field on its bad answer, we measure how
    often the judge votes for its own family even when bad.
    """
    n_total = len(rows)
    if n_total == 0:
        raise ValueError("no rows to evaluate")
    parse_fail = sum(1 for r in rows if r["parse_error"]) / n_total

    # Accuracy: judge picks the good answer regardless of position.
    correct = 0
    n_decided = 0
    for r in rows:
        if r["parse_error"] or r["verdict"] is None:
            continue
        n_decided += 1
        good_label = "A" if r["good_was_A"] else "B"
        if r["verdict"] == good_label:
            correct += 1
    accuracy = correct / n_decided if n_decided else 0.0

    # Consistency: per-pair, did all N runs (both orderings) agree on winner?
    by_pair: Dict[str, List[Optional[str]]] = {}
    for r in rows:
        if r["parse_error"]:
            continue
        # Normalize verdict: did it pick the good answer?
        good_label = "A" if r["good_was_A"] else "B"
        picked_good = r["verdict"] == good_label
        by_pair.setdefault(r["pair_id"], []).append(picked_good)
    consistent_pairs = sum(
        1 for pid, lst in by_pair.items() if lst and all(v == lst[0] for v in lst)
    )
    consistency = consistent_pairs / max(1, len(by_pair))

    # Position bias: P(picks A) - 0.5 across all rows where the verdict was decisive.
    picks_a = sum(1 for r in rows if not r["parse_error"] and r["verdict"] == "A")
    picks_b = sum(1 for r in rows if not r["parse_error"] and r["verdict"] == "B")
    decided = picks_a + picks_b
    position_bias = (picks_a / decided - 0.5) * 2 if decided else 0.0

    # Verbosity correlation: spearman between len(good)-len(bad) and verdict
    # encoded as +1 if picked good, -1 if picked bad.
    pair_map = {p.pair_id: p for p in pairs}
    deltas: List[float] = []
    verdicts: List[float] = []
    for r in rows:
        if r["parse_error"] or r["verdict"] not in {"A", "B"}:
            continue
        pair = pair_map.get(r["pair_id"])
        if not pair:
            continue
        if r["good_was_A"]:
            verdict_val = 1 if r["verdict"] == "A" else -1
        else:
            verdict_val = 1 if r["verdict"] == "B" else -1
        deltas.append(len(pair.good_answer) - len(pair.bad_answer))
        verdicts.append(verdict_val)
    verb_corr = _spearman(deltas, verdicts) if len(deltas) >= 3 else 0.0

    # Self-preference: only meaningful if pair.bad_answer has provenance.
    self_pref: Optional[float] = None
    if self_label is not None:
        n_self_present = 0
        n_self_picked = 0
        for r in rows:
            if r["parse_error"] or r["verdict"] not in {"A", "B"}:
                continue
            pair = pair_map.get(r["pair_id"])
            if not pair:
                continue
            bad_provenance = (pair.evidence[0].get("produced_by") if pair.evidence else "")
            if bad_provenance == self_label:
                n_self_present += 1
                if r["good_was_A"] and r["verdict"] == "B":
                    n_self_picked += 1
                elif (not r["good_was_A"]) and r["verdict"] == "A":
                    n_self_picked += 1
        self_pref = n_self_picked / n_self_present if n_self_present else None

    return CalibrationReport(
        n_pairs=len(pair_map),
        n_runs_per_pair=n_runs_per_pair,
        accuracy=float(accuracy),
        consistency=float(consistency),
        position_bias=float(position_bias),
        verbosity_correlation=float(verb_corr),
        self_preference=self_pref,
        parse_failure_rate=float(parse_fail),
    )


# ---------------------------------------------------------------------------
# Probabilistic judges: ECE / Brier wrappers
# ---------------------------------------------------------------------------


def probabilistic_judge_ece(
    confidences: Sequence[float],
    correct: Sequence[int],
    n_bins: int = 10,
) -> Dict[str, float]:
    """When a judge emits a "confidence" field, calibrate it.

    Returns ECE / Brier / MCE so dashboard can show a calibration curve.
    """
    rpt = expected_calibration_error(confidences, correct, n_bins=n_bins)
    return {"ece": rpt.ece, "brier": rpt.brier, "mce": rpt.mce}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    n = len(xs)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    denx = (sum((r - mean_x) ** 2 for r in rx)) ** 0.5
    deny = (sum((r - mean_y) ** 2 for r in ry)) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def _rankdata(xs: Sequence[float]) -> List[float]:
    """Average-rank ties."""
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    n = len(xs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks
