"""Lab 5: LLM-judge calibration with simulated runs.

Replays a 5-pair calibration set against a synthetic judge with known
bias (slight position preference + verbosity preference) so you can see
how the calibration report surfaces both.

Run:
    python -m benchmark.rag.labs.lab5_judge_calibration
"""
from __future__ import annotations

import random

from ..judge_calibration import GoldPair, calibration_report, replay_calibration_pairs


def main() -> None:
    rng = random.Random(0)

    pairs = [
        GoldPair(
            pair_id=f"p{i}",
            question="What is the boiling point of water at sea level?",
            good_answer="At sea level, water boils at 100 degrees Celsius.",
            bad_answer="Water boils when it's hot. Very hot. Around two hundred degrees in many places.",
        )
        for i in range(5)
    ]

    def simulated_judge(pair: GoldPair, good_is_A: bool):
        verdicts = []
        for _ in range(3):
            # Real judge accuracy ~ 85%; small position bias toward A.
            picks_correct = rng.random() < 0.85
            if picks_correct:
                # Bias: 10% of the time picks longer answer instead of correct.
                if len(pair.bad_answer) > len(pair.good_answer) and rng.random() < 0.10:
                    picks_correct = False
            picked_label = "A" if (good_is_A and picks_correct) or (not good_is_A and not picks_correct) else "B"
            # Position bias: occasionally just picks A regardless.
            if rng.random() < 0.05:
                picked_label = "A"
            verdicts.append({"winner": picked_label, "rationale": "simulated"})
        return verdicts

    rows = replay_calibration_pairs(pairs, simulated_judge, n_runs_per_pair=3, swap_order=True)
    report = calibration_report(rows, pairs, n_runs_per_pair=3)
    print("# Lab 5: judge calibration")
    print("-" * 60)
    print(f"  pairs:                {report.n_pairs}")
    print(f"  runs per pair:        {report.n_runs_per_pair}")
    print(f"  judge accuracy:       {report.accuracy:.3f}")
    print(f"  consistency:          {report.consistency:.3f}")
    print(f"  position bias:        {report.position_bias:+.3f}  (0 = no bias, +/- = pro A / pro B)")
    print(f"  verbosity correlation:{report.verbosity_correlation:+.3f}")
    print(f"  parse failure rate:   {report.parse_failure_rate:.3f}")


if __name__ == "__main__":
    main()
