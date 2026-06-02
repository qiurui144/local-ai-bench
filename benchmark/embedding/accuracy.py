"""Embedding retrieval-quality evaluation.

Calls the served (vLLM / OpenAI-compatible) ``/v1/embeddings`` endpoint to embed
each query and its candidate docs, ranks candidates by cosine similarity, and
scores recall@k / MRR / nDCG@10 against the gold relevance — plus a numerical
validation gate so a silently broken embedder (zero / NaN vectors) surfaces as a
FAIL instead of a plausible-looking number.

Output JSON shape mirrors ``benchmark/accuracy.py`` / the translation module:
  aggregate : corpus recall@k / MRR / nDCG@10 + validation summary
  verdict   : PASS / WARN / FAIL against thresholds
  per_query : per-query rank-of-first-hit provenance
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from common import ModelConfig, infer_embedding

from .datasets import RetrievalQuery
from .metrics import (
    aggregate_retrieval,
    cosine_topk,
    ndcg_at_k,
    reciprocal_rank,
    validate_embeddings,
)

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = {
    "recall_at_1_min": 0.60,
    "recall_at_10_min": 0.90,
    "mrr_min": 0.70,
    "ndcg_at_10_min": 0.75,
}


def _embed_texts(model_cfg: ModelConfig, texts: Sequence[str]) -> tuple[list, bool, str]:
    """Embed ``texts`` in one call; return (vectors, ok, error)."""
    res = infer_embedding(model_cfg, list(texts))
    if not res.ok:
        return [], False, res.error
    if len(res.embeddings) != len(texts):
        return res.embeddings, False, (
            f"embedding count {len(res.embeddings)} != inputs {len(texts)}"
        )
    return res.embeddings, True, ""


def run_embedding(
    model_cfg: ModelConfig,
    queries: Sequence[RetrievalQuery],
    *,
    thresholds: Optional[dict] = None,
) -> dict:
    """Embed + rank + score ``queries``; return a benchmark result dict."""
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    if not queries:
        return {"benchmark": "embedding", "model": model_cfg.name,
                "skipped": True, "reason": "no queries"}

    rankings: list[list[int]] = []
    relevants: list[set] = []
    per_query: list[dict] = []
    all_vectors: list = []          # for one corpus-wide numerical validation
    call_errors = 0

    for q in queries:
        q_vecs, ok_q, err_q = _embed_texts(model_cfg, [q.query])
        d_vecs, ok_d, err_d = _embed_texts(model_cfg, q.candidates)
        if not (ok_q and ok_d) or not q_vecs:
            call_errors += 1
            per_query.append({"qid": q.qid, "ok": False,
                              "error": err_q or err_d, "source": q.source})
            # Empty ranking → contributes 0 to every metric (honest penalty).
            rankings.append([])
            relevants.append(q.relevant)
            continue

        all_vectors.append(q_vecs[0])
        all_vectors.extend(d_vecs)

        ranked = cosine_topk(q_vecs[0], d_vecs)
        rankings.append(ranked)
        relevants.append(q.relevant)
        per_query.append({
            "qid": q.qid,
            "ok": True,
            "source": q.source,
            "rr": reciprocal_rank(ranked, q.relevant),
            "ndcg@10": ndcg_at_k(ranked, q.relevant, 10),
            "top1": ranked[0] if ranked else None,
        })

    agg = aggregate_retrieval(rankings, relevants)
    validation = validate_embeddings(all_vectors)
    agg["validation"] = validation
    agg["call_error_rate"] = call_errors / len(queries)
    agg["data_source"] = queries[0].source

    reasons: list[str] = []
    if not validation["ok"]:
        reasons.append(
            "FAIL: numerical validation "
            f"(zero={validation['zero_vectors']} nan={validation['nan_vectors']} "
            f"inf={validation['inf_vectors']} dim_mismatch={validation['dim_mismatch']})"
        )
    if call_errors == len(queries):
        reasons.append("FAIL: all embedding calls failed (endpoint unavailable?)")
    if agg["recall@1"] < thresholds.get("recall_at_1_min", 0):
        reasons.append(f"FAIL: recall@1 {agg['recall@1']:.3f} < {thresholds['recall_at_1_min']}")
    if agg["recall@10"] < thresholds.get("recall_at_10_min", 0):
        reasons.append(f"FAIL: recall@10 {agg['recall@10']:.3f} < {thresholds['recall_at_10_min']}")
    if agg["mrr"] < thresholds.get("mrr_min", 0):
        reasons.append(f"FAIL: MRR {agg['mrr']:.3f} < {thresholds['mrr_min']}")
    if agg["ndcg@10"] < thresholds.get("ndcg_at_10_min", 0):
        reasons.append(f"FAIL: nDCG@10 {agg['ndcg@10']:.3f} < {thresholds['ndcg_at_10_min']}")

    verdict = "FAIL" if any(r.startswith("FAIL") for r in reasons) else (
        "WARN" if reasons else "PASS"
    )

    return {
        "benchmark": "embedding",
        "model": model_cfg.name,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "aggregate": agg,
        "per_query": per_query,
    }
