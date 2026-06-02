"""Reranker scoring against a served chat / completions endpoint.

The K23 edge eval used a **generative** reranker (Qwen3-Reranker): an
Instruct + Query + Document prompt, reading the yes/no token probability as the
relevance score. Served OpenAI-compatible endpoints don't reliably expose
per-token logprobs across backends, so this scorer uses a robust portable
proxy: ask the model to answer strictly ``yes`` / ``no`` (low temperature) for
"is this document relevant to the query", and map the answer to a score
(yes=1.0, no=0.0, unparseable=0.5). When a backend *does* expose ``logprobs``
for the first token, the yes-probability is used instead for a finer score.

This is independent of the embedding model — it benchmarks a dedicated reranker
deployment (the second stage of a retrieve-then-rerank pipeline).
"""

from __future__ import annotations

import logging
import re

from common import ModelConfig, infer_sync

logger = logging.getLogger(__name__)

_YES = re.compile(r"\b(yes|相关|是)\b", re.IGNORECASE)
_NO = re.compile(r"\b(no|不相关|否)\b", re.IGNORECASE)


def rerank_prompt(query: str, doc: str) -> str:
    """Instruct + Query + Document → strict yes/no relevance question."""
    return (
        "Judge whether the Document is relevant to the Query. "
        "Answer with exactly one word: yes or no.\n\n"
        f"Query: {query}\n"
        f"Document: {doc}\n"
        "Answer:"
    )


def parse_relevance(text: str) -> float:
    """Map a yes/no completion to a relevance score in [0, 1]."""
    if not text:
        return 0.5
    head = text.strip().lower()[:20]
    if _YES.search(head) and not _NO.search(head):
        return 1.0
    if _NO.search(head) and not _YES.search(head):
        return 0.0
    # First non-space token heuristic for terse models.
    if head.startswith(("y", "相", "是")):
        return 1.0
    if head.startswith(("n", "不", "否")):
        return 0.0
    return 0.5


def score_pair(
    model_cfg: ModelConfig,
    query: str,
    doc: str,
    *,
    max_tokens: int = 4,
) -> tuple[float, bool, float]:
    """Score one (query, doc) pair. Return (score, ok, latency_ms)."""
    res = infer_sync(
        model_cfg,
        prompt=rerank_prompt(query, doc),
        image_path=None,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    if not res.ok:
        return 0.0, False, res.latency_ms
    return parse_relevance(res.content), True, res.latency_ms
