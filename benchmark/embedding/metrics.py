"""Retrieval metrics + numerical validation for the embedding dimension.

Pure NumPy, CPU-only, deterministic — every function here is unit-testable
without a vLLM / GPU endpoint.

Methodology mirrors the K23 edge embedding eval
(``rv-achievements/reports/2026-06-01_embedding_reranker_eval.md``):

- **Recall@k / MRR / nDCG@10** — for each query, score its candidate docs by
  cosine similarity against the query vector, rank, and measure where the
  relevant doc(s) land. Cosine is computed with plain NumPy (``q·d / |q||d|``)
  so results are independent of any vector DB / faiss build.
- **Numerical validation** — every embedding is checked for NaN / Inf / all-zero
  / dimension drift. A zero vector is a **FAIL** (it silently collapses cosine
  to 0 and produces a plausible-looking-but-wrong ranking — the classic "fast
  but wrong output" trap).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Cosine + ranking primitives
# ---------------------------------------------------------------------------
def cosine_topk(query_vec: Sequence[float], doc_vecs: Sequence[Sequence[float]]) -> list[int]:
    """Return doc indices sorted by descending cosine similarity to ``query_vec``.

    Zero-norm vectors (query or doc) yield a similarity of 0 rather than NaN, so
    a degenerate embedding ranks last instead of crashing the sort.
    """
    q = np.asarray(query_vec, dtype=np.float64)
    d = np.asarray(doc_vecs, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] == 0:
        return []
    qn = np.linalg.norm(q)
    dn = np.linalg.norm(d, axis=1)
    denom = dn * qn
    sims = np.where(denom > 0, (d @ q) / np.where(denom > 0, denom, 1.0), 0.0)
    # Stable descending sort (ties broken by original index for determinism).
    return list(np.argsort(-sims, kind="stable"))


def _dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant docs retrieved within the top-k.

    With a single relevant doc this is the standard hit@k (0 or 1).
    """
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return len(top & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Whether at least one relevant doc appears within the top-k."""
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return 1.0 if top & relevant else 0.0


def reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    """1 / rank of the first relevant doc (0 if none retrieved)."""
    for i, idx in enumerate(ranked, start=1):
        if idx in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """Binary-relevance nDCG@k in [0, 1]."""
    if not relevant:
        return 0.0
    gains = [1.0 if idx in relevant else 0.0 for idx in ranked[:k]]
    dcg = _dcg(gains)
    ideal = _dcg([1.0] * min(len(relevant), k))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------
def aggregate_retrieval(
    rankings: Sequence[Sequence[int]],
    relevants: Sequence[set],
    *,
    ks: Sequence[int] = (1, 5, 10),
    ndcg_k: int = 10,
) -> dict:
    """Mean recall@k/hit@k (each k) + MRR + nDCG@ndcg_k across all queries."""
    n = len(rankings)
    if n == 0:
        return {"num_queries": 0, "mrr": 0.0, f"ndcg@{ndcg_k}": 0.0,
                **{f"recall@{k}": 0.0 for k in ks},
                **{f"hit@{k}": 0.0 for k in ks}}
    out: dict = {"num_queries": n}
    for k in ks:
        out[f"recall@{k}"] = sum(
            recall_at_k(r, rel, k) for r, rel in zip(rankings, relevants)
        ) / n
        out[f"hit@{k}"] = sum(
            hit_at_k(r, rel, k) for r, rel in zip(rankings, relevants)
        ) / n
    out["mrr"] = sum(
        reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants)
    ) / n
    out[f"ndcg@{ndcg_k}"] = sum(
        ndcg_at_k(r, rel, ndcg_k) for r, rel in zip(rankings, relevants)
    ) / n
    return out


# ---------------------------------------------------------------------------
# Numerical validation (anti "fast but wrong")
# ---------------------------------------------------------------------------
def validate_embeddings(
    vectors: Sequence[Sequence[float]],
    *,
    expected_dim: int | None = None,
) -> dict:
    """Check a batch of embeddings for NaN / Inf / zero-vector / dim drift.

    Returns a dict with ``ok`` (all checks pass) plus per-issue counts. A single
    zero / NaN / Inf vector flips ``ok`` to False — the caller treats that as a
    FAIL rather than reporting a fast-but-meaningless score.
    """
    total = len(vectors)
    nan = inf = zero = dim_mismatch = empty = 0
    dim_seen = expected_dim
    for v in vectors:
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == 0:
            empty += 1
            continue
        if dim_seen is None:
            dim_seen = arr.size
        elif arr.size != dim_seen:
            dim_mismatch += 1
        if np.isnan(arr).any():
            nan += 1
        if np.isinf(arr).any():
            inf += 1
        if float(np.linalg.norm(arr)) == 0.0:
            zero += 1
    ok = total > 0 and nan == 0 and inf == 0 and zero == 0 and empty == 0 and dim_mismatch == 0
    return {
        "ok": ok,
        "total": total,
        "dim": dim_seen or 0,
        "nan_vectors": nan,
        "inf_vectors": inf,
        "zero_vectors": zero,
        "empty_vectors": empty,
        "dim_mismatch": dim_mismatch,
    }
