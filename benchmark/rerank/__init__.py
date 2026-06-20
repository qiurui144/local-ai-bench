"""Standalone reranker benchmark dimension (nDCG@10 / MRR + per-pair latency).

Benchmarks a dedicated reranker deployment as the second stage of a
retrieve-then-rerank pipeline — distinct from the RAG-internal reranker in
``benchmark/rag/reranker.py``. Reuses ``benchmark.embedding``'s retrieval
datasets + ranking metrics so embedding and rerank share one gold relevance.

Two scoring paths share the same metrics:

- **native** (``rerank_native: true``) — a BERT cross-encoder (bge-reranker)
  served by llama.cpp ``--reranking`` / vLLM via the native ``/v1/rerank``
  endpoint: one single-pass relevance score per doc, batched per query.
  Real-time-capable; GGUF carries its own tokenizer (no Python deps on host).
- **generative proxy** (default) — yes/no relevance from a chat endpoint
  (Qwen3-Reranker); offline-grade latency.

Methodology source: K23 edge eval §4 (generative Qwen3-Reranker) +
``2026-06-02_realtime_reranker_plan.md`` (native bge cross-encoder); both score
per-pair relevance → nDCG/MRR with an explicit latency reality check.

- ``scorer``   : relevance scoring — ``score_pair`` (routes native/generative)
  and ``score_query_native`` (whole candidate list in one ``/v1/rerank`` call).
- ``accuracy`` : ``run_rerank`` — score + re-rank + nDCG/MRR + latency + score
  separation sanity, with PASS/WARN/FAIL verdict.
"""

from __future__ import annotations

__all__ = [
    "accuracy",
    "scorer",
]
