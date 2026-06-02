"""Standalone reranker quality evaluation (nDCG@10 / MRR + per-pair latency).

Distinct from the RAG-internal reranker in ``benchmark/rag/reranker.py``: this
is a **standalone benchmark** of a dedicated reranker deployment — score every
(query, candidate-doc) pair, re-rank by score, and measure nDCG@10 / MRR plus
single-pair latency P50. Reuses the embedding dimension's retrieval datasets +
ranking metrics so embedding and rerank are scored on the same gold relevance.

Methodology source: K23 edge eval Qwen3-Reranker section
(``2026-06-01_embedding_reranker_eval.md`` §4) — generative reranker, per-pair
relevance, nDCG/MRR, and an explicit per-pair-latency reality check (a 2.9 s/pair
reranker is offline-only).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from common import ModelConfig, summarize_latencies

from benchmark.embedding.datasets import RetrievalQuery
from benchmark.embedding.metrics import ndcg_at_k, reciprocal_rank, aggregate_retrieval

from .scorer import score_pair

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = {
    "ndcg_at_10_min": 0.75,
    "mrr_min": 0.70,
}


def _rank_by_scores(scores: Sequence[float]) -> list[int]:
    """Indices sorted by descending score (stable → deterministic ties)."""
    return [i for i, _ in sorted(enumerate(scores), key=lambda t: (-t[1], t[0]))]


def run_rerank(
    model_cfg: ModelConfig,
    queries: Sequence[RetrievalQuery],
    *,
    thresholds: Optional[dict] = None,
) -> dict:
    """Score + re-rank candidate docs per query; nDCG@10 / MRR + pair latency."""
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    if not queries:
        return {"benchmark": "rerank", "model": model_cfg.name,
                "skipped": True, "reason": "no queries"}

    rankings: list[list[int]] = []
    relevants: list[set] = []
    per_query: list[dict] = []
    pair_latencies: list[float] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    pair_errors = total_pairs = 0

    for q in queries:
        scores: list[float] = []
        for j, doc in enumerate(q.candidates):
            total_pairs += 1
            s, ok, lat = score_pair(model_cfg, q.query, doc)
            scores.append(s)
            if ok:
                pair_latencies.append(lat)
                (pos_scores if j in q.relevant else neg_scores).append(s)
            else:
                pair_errors += 1
        ranked = _rank_by_scores(scores)
        rankings.append(ranked)
        relevants.append(q.relevant)
        per_query.append({
            "qid": q.qid,
            "source": q.source,
            "rr": reciprocal_rank(ranked, q.relevant),
            "ndcg@10": ndcg_at_k(ranked, q.relevant, 10),
        })

    agg = aggregate_retrieval(rankings, relevants)
    agg["data_source"] = queries[0].source
    agg["num_pairs"] = total_pairs
    agg["pair_error_rate"] = pair_errors / total_pairs if total_pairs else 0.0
    agg["single_pair_latency_ms_stats"] = summarize_latencies(pair_latencies)
    # Score separation sanity (anti random / collapsed reranker).
    agg["score_separation"] = {
        "pos_mean": (sum(pos_scores) / len(pos_scores)) if pos_scores else 0.0,
        "neg_mean": (sum(neg_scores) / len(neg_scores)) if neg_scores else 0.0,
    }

    reasons: list[str] = []
    if pair_errors == total_pairs and total_pairs > 0:
        reasons.append("FAIL: all rerank calls failed (endpoint unavailable?)")
    if agg["ndcg@10"] < thresholds.get("ndcg_at_10_min", 0):
        reasons.append(f"FAIL: nDCG@10 {agg['ndcg@10']:.3f} < {thresholds['ndcg_at_10_min']}")
    if agg["mrr"] < thresholds.get("mrr_min", 0):
        reasons.append(f"FAIL: MRR {agg['mrr']:.3f} < {thresholds['mrr_min']}")
    sep = agg["score_separation"]
    if sep["pos_mean"] <= sep["neg_mean"]:
        reasons.append(
            f"WARN: score separation weak (pos {sep['pos_mean']:.2f} "
            f"<= neg {sep['neg_mean']:.2f})"
        )

    verdict = "FAIL" if any(r.startswith("FAIL") for r in reasons) else (
        "WARN" if reasons else "PASS"
    )

    return {
        "benchmark": "rerank",
        "model": model_cfg.name,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "aggregate": agg,
        "per_query": per_query,
    }
