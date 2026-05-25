"""Lab 6: Regression CI end-to-end on a tempdir.

Creates a tiny golden set, runs it once to seed a baseline, then runs
again with one item intentionally regressed to demonstrate the detection
flow and the flake controller.

Run:
    python -m benchmark.rag.labs.lab6_regression_ci
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from ..regression_ci import (
    GoldenItem,
    detect_regressions,
    flake_recheck,
    load_baseline,
    run_against_golden,
    write_snapshot,
)


def main() -> None:
    items = [
        GoldenItem(item_id=f"q{i}", query=f"question {i}", expected={"answer": f"ans{i}"})
        for i in range(10)
    ]

    def stable_runner(item: GoldenItem):
        return {"ndcg": 0.85, "grounded": 0.92}

    def regressing_runner(item: GoldenItem):
        rng = random.Random(item.item_id)
        # Item q3 regresses by 0.10 on ndcg; q7 fluctuates randomly (flake)
        if item.item_id == "q3":
            return {"ndcg": 0.70, "grounded": 0.92}
        if item.item_id == "q7":
            return {"ndcg": 0.85 + rng.uniform(-0.05, 0.05), "grounded": 0.92}
        return {"ndcg": 0.85, "grounded": 0.92}

    with tempfile.TemporaryDirectory() as tmp:
        baseline_path = Path(tmp) / "baseline.json"
        # Seed baseline.
        baseline_rows = run_against_golden(items, stable_runner)
        write_snapshot(baseline_rows, baseline_path)

        # Re-run with regressions.
        current_rows = run_against_golden(items, regressing_runner)
        baseline = load_baseline(baseline_path)
        report = detect_regressions(current_rows, baseline, tolerance=0.02)
        print("# Lab 6: regression CI")
        print("-" * 60)
        print(f"detected regressions: {len(report.regressions)}")
        for r in report.regressions:
            print(f"  {r.item_id}.{r.metric}: {r.baseline:.3f} -> {r.current:.3f} (delta={r.delta:+.3f})")
        print(f"aggregate deltas: {report.aggregate_deltas}")

        # Run flake controller (3 retries on each flagged).
        retries = flake_recheck(
            report.regressions,
            lambda iid: regressing_runner(next(it for it in items if it.item_id == iid)),
            n_retries=3,
        )
        print(f"\nAfter flake controller: {len(retries)} surviving regressions")
        for r in retries:
            print(f"  {r.item_id}.{r.metric}: {r.baseline:.3f} -> {r.current:.3f}")


if __name__ == "__main__":
    main()
