"""Lab 3: Reranker win-rate evaluation with latency budget.

Builds two ranked-list versions of the same retrieval and uses
rerank_win_rate to confirm that the "reranker" gained quality without
blowing the latency budget.

Run:
    python -m benchmark.rag.labs.lab3_reranker_winrate
"""
from __future__ import annotations

import random

from ..reranker import (
    LatencyBudgetReport,
    latency_budget,
    rerank_win_rate,
)
from ..retrieval_metrics import RetrievalQueryResult


def main() -> None:
    rng = random.Random(0)
    base_results = []
    rerank_results = []
    rerank_latencies = []
    for q in range(30):
        rel = "d_gold"
        # Baseline retrieval: relevant doc somewhere in top-5 at random rank
        rank_pos = rng.randint(1, 5)
        base_ranked = ["d_noise"] * (rank_pos - 1) + [rel] + ["d_noise"] * (5 - rank_pos)
        base_results.append(
            RetrievalQueryResult(
                query_id=f"q{q}",
                ranked_doc_ids=base_ranked,
                relevant_doc_ids=[rel],
                relevance_grades={d: (3.0 if d == rel else 0.0) for d in base_ranked},
            )
        )
        # Reranked: relevant doc moved to rank 1 with 80% probability
        if rng.random() < 0.8:
            rer_ranked = [rel] + ["d_noise"] * 4
        else:
            rer_ranked = base_ranked  # reranker fails sometimes
        rerank_results.append(
            RetrievalQueryResult(
                query_id=f"q{q}",
                ranked_doc_ids=rer_ranked,
                relevant_doc_ids=[rel],
                relevance_grades={d: (3.0 if d == rel else 0.0) for d in rer_ranked},
            )
        )
        rerank_latencies.append(rng.gauss(80, 25))  # rerank ms overhead

    report = rerank_win_rate(
        base_results,
        rerank_results,
        metric="ndcg",
        k=5,
        latency_overhead_ms=rerank_latencies,
    )
    print("# Lab 3: reranker win-rate")
    print("-" * 60)
    print(f"queries:    {report.n_queries}")
    print(f"wins:       {report.wins}")
    print(f"ties:       {report.ties}")
    print(f"losses:     {report.losses}")
    print(f"win_rate:   {report.win_rate:.3f}")
    print(f"mean delta: {report.mean_metric_delta:+.4f}")
    print(f"P95 latency overhead: {report.p95_latency_overhead_ms:.1f} ms")

    lat = latency_budget(rerank_latencies, p50_budget=100, p95_budget=200)
    print(
        f"\nLatency budget check (p50<=100, p95<=200): "
        f"fits={lat.fits_budget}  P50={lat.p50_ms:.0f} P95={lat.p95_ms:.0f} P99={lat.p99_ms:.0f}"
    )


if __name__ == "__main__":
    main()
