"""Embedding latency + memory characterisation.

Two product-relevant numbers, following the K23 edge eval methodology
(``2026-06-01_embedding_reranker_eval.md``):

- **Single-query embed latency P50 (resident)** — the served model stays
  resident (vLLM / llama-server), and we time single-text ``/v1/embeddings``
  calls. This is the real conversational-query path; a per-process CLI embedder
  would fold in model load time and overstate latency, so we never measure that
  way here.
- **RSS dual distinction** — *batch RSS* (embedding a large batch in one call;
  inflated by logical-batch KV allocation, NOT the query-path memory) vs
  *resident-query RSS* (the server's peak RSS after a single short query;
  ≈ weights + small KV — the real chat-query memory). RSS is only readable when
  the served process is local; otherwise it is reported ``available: False``.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from common import ModelConfig, infer_embedding, proc_rss_mb, summarize_latencies

from .datasets import RetrievalQuery

logger = logging.getLogger(__name__)


def run_embedding_latency(
    model_cfg: ModelConfig,
    texts: Sequence[str],
    *,
    samples: int = 12,
    warmup: int = 2,
) -> dict:
    """Single-text embed latency over ``samples`` resident-model calls."""
    if not texts:
        return {"benchmark": "embedding_latency", "model": model_cfg.name, "skipped": True}

    # Warmup (resident model — exclude first-call cache effects).
    for i in range(warmup):
        infer_embedding(model_cfg, texts[i % len(texts)])

    latencies: list[float] = []
    errors = 0
    for i in range(samples):
        res = infer_embedding(model_cfg, texts[i % len(texts)])
        if res.ok and res.latency_ms > 0:
            latencies.append(res.latency_ms)
        else:
            errors += 1
        logger.info("  [embed-lat %d/%d] %s %.1fms",
                    i + 1, samples, model_cfg.name, res.latency_ms)

    return {
        "benchmark": "embedding_latency",
        "model": model_cfg.name,
        "path": "resident-model single-query",
        "samples": samples,
        "single_query_latency_ms_stats": summarize_latencies(latencies),
        "errors": errors,
        "error_rate": errors / samples if samples else 0,
    }


def measure_rss_dual(
    model_cfg: ModelConfig,
    texts: Sequence[str],
    *,
    server_pid: Optional[int] = None,
    batch_size: int = 256,
) -> dict:
    """Batch RSS vs resident-query RSS (VmHWM), read from /proc/<pid>/status.

    ``server_pid`` must be the locally-served model process. When it is None or
    unreadable (remote endpoint / non-Linux), RSS is reported unavailable
    instead of fabricating a number.
    """
    if not server_pid:
        return {"available": False, "reason": "server pid unknown (remote endpoint?)"}

    # Resident-query RSS: a single short query, then peak RSS.
    infer_embedding(model_cfg, texts[0] if texts else "查询")
    resident_rss = proc_rss_mb(server_pid)

    # Batch RSS: embed a large batch in one call, then peak RSS.
    batch = list(texts)[:batch_size] or ["填充文本"] * batch_size
    infer_embedding(model_cfg, batch)
    batch_rss = proc_rss_mb(server_pid)

    if resident_rss == 0.0 and batch_rss == 0.0:
        return {"available": False, "reason": f"/proc/{server_pid}/status unreadable"}

    return {
        "available": True,
        "server_pid": server_pid,
        "resident_query_rss_mb": resident_rss,   # product-relevant chat memory
        "batch_rss_mb": batch_rss,               # inflated by logical-batch KV
        "note": "resident_query_rss is the real chat-query memory; "
                "batch_rss is inflated by logical-batch KV allocation",
    }


def run_embedding_performance(
    model_cfg: ModelConfig,
    queries: Sequence[RetrievalQuery],
    *,
    samples: int = 12,
    server_pid: Optional[int] = None,
) -> dict:
    """Latency P50 (resident) + RSS dual distinction for the embedding model."""
    texts = [q.query for q in queries] or ["查询"]
    out: dict = {"benchmark": "embedding_performance", "model": model_cfg.name}
    out["latency"] = run_embedding_latency(model_cfg, texts, samples=samples)
    out["memory"] = measure_rss_dual(model_cfg, texts, server_pid=server_pid)
    return out
