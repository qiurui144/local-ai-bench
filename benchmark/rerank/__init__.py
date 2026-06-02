"""Standalone reranker benchmark dimension (nDCG@10 / MRR + per-pair latency).

Benchmarks a dedicated reranker deployment as the second stage of a
retrieve-then-rerank pipeline — distinct from the RAG-internal reranker in
``benchmark/rag/reranker.py``. Reuses ``benchmark.embedding``'s retrieval
datasets + ranking metrics so embedding and rerank share one gold relevance.

Methodology source: K23 edge eval §4 (generative Qwen3-Reranker; per-pair
relevance; nDCG/MRR; per-pair latency reality check).

- ``scorer``   : served-endpoint relevance scoring (yes/no proxy, logprob when
  available).
- ``accuracy`` : ``run_rerank`` — score + re-rank + nDCG/MRR + latency + score
  separation sanity, with PASS/WARN/FAIL verdict.
"""

from __future__ import annotations

__all__ = [
    "accuracy",
    "scorer",
]
