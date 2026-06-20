"""Embedding retrieval-quality + latency/memory benchmark dimension.

Synced from the K23 edge AI-box embedding eval methodology, adapted to a served
OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM / sglang / llama.cpp server
/ Ollama). The K3 X100 eval drove the metric/path choices:
``rv-achievements/reports/2026-06-01_embedding_reranker_eval.md``.

- ``metrics``     : pure-NumPy recall@k / MRR / nDCG@10 + numerical validation
  (zero / NaN / Inf vectors → FAIL). CPU-only, fully unit-testable.
- ``datasets``    : retrieval-set loaders (JSONL custom corpus + a built-in
  synthetic Chinese fallback for offline / unit-test runs).
- ``accuracy``    : ``run_embedding`` — embed queries + candidates, cosine top-k
  rank, score, PASS/WARN/FAIL verdict.
- ``performance`` : ``run_embedding_performance`` — single-query embed latency
  P50 (resident model) + RSS dual distinction (batch vs resident-query).
"""

from __future__ import annotations

__all__ = [
    "accuracy",
    "datasets",
    "metrics",
    "performance",
]
