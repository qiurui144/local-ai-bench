"""Lab 8: Drift detection with PSI, KS, and temporal cohorts.

Builds synthetic reference and observed distributions, computes drift
signals, and demonstrates auto-curation of low-confidence cases.

Run:
    python -m benchmark.rag.labs.lab8_drift_detection
"""
from __future__ import annotations

import random
import time

import numpy as np

from ..drift_detection import (
    assess_drift,
    mine_high_disagreement_candidates,
    mine_low_confidence_candidates,
)


def main() -> None:
    rng = np.random.default_rng(0)

    # Reference: query lengths around 25 chars, scores around 0.7
    ref_queries = [f"reference query {i:02d}" for i in range(200)]
    ref_scores = rng.normal(0.7, 0.10, size=300).tolist()
    ref_embeddings = rng.normal(0, 1, size=(200, 64))

    # Observed: longer queries, scores shifted down
    obs_queries = [f"a slightly longer observed query about topic {i}" for i in range(200)]
    obs_scores = rng.normal(0.55, 0.15, size=300).tolist()
    obs_embeddings = rng.normal(0.2, 1, size=(200, 64))

    # Temporal performance: degrades over time.
    now = time.time()
    timestamps = [now - (300 - i) * 3600 for i in range(300)]
    metric_values = [0.9 - 0.001 * i + random.Random(i).gauss(0, 0.02) for i in range(300)]

    report = assess_drift(
        reference={
            "queries": ref_queries,
            "embeddings": ref_embeddings,
            "scores": ref_scores,
        },
        observed={
            "queries": obs_queries,
            "embeddings": obs_embeddings,
            "scores": obs_scores,
            "timestamps": timestamps,
            "metric_values": metric_values,
        },
    )
    print("# Lab 8: drift detection")
    print("-" * 60)
    print(f"recommendation: {report.recommendation}")
    print("\ndrift signals:")
    for s in report.signals:
        flag = "TRIGGER" if s.triggered else "OK"
        print(f"  [{flag}] {s.metric}: value={s.value:.4f} threshold={s.threshold:.4f}")

    print("\nAuto-curation: mining low-confidence cases")
    items = [
        {"item_id": f"i{i}", "query": f"q{i}", "confidence": random.random()}
        for i in range(50)
    ]
    cands = mine_low_confidence_candidates(items, threshold=0.3, limit=5)
    for c in cands:
        print(f"  candidate {c.item_id}: confidence={c.confidence:.3f}")


if __name__ == "__main__":
    main()
