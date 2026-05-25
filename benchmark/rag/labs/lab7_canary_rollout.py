"""Lab 7: Canary rollout with rollback policy.

Simulates a 10% canary that starts healthy, then degrades over time.
Watch the rollback policy decide based on consecutive breaches.

Run:
    python -m benchmark.rag.labs.lab7_canary_rollout
"""
from __future__ import annotations

import random

from ..canary import (
    CanaryGate,
    CanaryThreshold,
    RollbackPolicy,
    TrafficSplitter,
)


def main() -> None:
    splitter = TrafficSplitter(fraction_to_candidate=0.10, salt="lab7")
    gate = CanaryGate(
        thresholds=[
            CanaryThreshold(metric="groundedness", min_value=0.85),
            CanaryThreshold(metric="latency_ms", max_value=2500),
        ],
        window_size=100,
        min_samples=50,
    )
    policy = RollbackPolicy(consecutive_breaches=2, cooldown_seconds=10)

    rng = random.Random(0)
    print("# Lab 7: canary rollout")
    print("-" * 60)
    print("Routing 500 requests; canary gradually degrades after step 200")

    for step in range(500):
        arm = splitter.route(f"req-{step}")
        if arm != "candidate":
            continue
        # Healthy mean 0.9, gradually degrading after step 200 to 0.7.
        degrade = max(0, (step - 200) / 300)
        groundedness = max(0.0, rng.gauss(0.9 - 0.2 * degrade, 0.05))
        latency = rng.gauss(1500 + 1500 * degrade, 200)
        gate.record("groundedness", groundedness)
        gate.record("latency_ms", latency)
        if step % 50 == 0 and step >= 100:
            status = gate.evaluate()
            decision = policy.step(status)
            print(
                f"step={step:>4} arm=canary groundedness_mean="
                f"{status.metric_summary['groundedness'].get('mean', float('nan')):.3f} "
                f"decision={status.decision} policy={decision['action']}"
            )
            if decision["action"] == "rollback_now":
                print(f"  --> ROLLBACK triggered at step {step}; violations={status.violations}")
                break


if __name__ == "__main__":
    main()
