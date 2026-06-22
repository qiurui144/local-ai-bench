"""Reranker scoring against a served endpoint — two paths.

**Native (real-time) path** — ``rerank_native: true``. A BERT cross-encoder
(e.g. bge-reranker-v2-m3) served by llama.cpp ``--reranking`` (``--pooling
rank``) / vLLM, scoring ``[CLS] query [SEP] doc [SEP]`` in a single encoder pass
via the native ``/v1/rerank`` endpoint. No generation, no KV cache → ~20-100×
faster than the generative path, the difference between offline-only and
real-time reranking. The GGUF model carries its own tokenizer, so the serving
host needs no Python ``transformers`` (unblocks tokenizer-less edge devices —
the same path the embedding GGUFs already run on).

**Generative (proxy) path** — the default. The K23 edge eval used a generative
reranker (Qwen3-Reranker): an Instruct + Query + Document prompt, reading the
yes/no token probability as the relevance score. Served OpenAI-compatible
endpoints don't reliably expose per-token logprobs across backends, so this
scorer uses a robust portable proxy: ask the model to answer strictly ``yes`` /
``no`` (low temperature), and map the answer to a score (yes=1.0, no=0.0,
unparseable=0.5).

Both benchmark a dedicated reranker deployment (the second stage of a
retrieve-then-rerank pipeline), independent of the embedding model.
"""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from typing import Sequence

from common import ModelConfig, infer_rerank, infer_sync

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


@lru_cache(maxsize=4)
def _load_cross_encoder(model_id: str):
    from sentence_transformers import CrossEncoder  # type: ignore
    return CrossEncoder(model_id)


@lru_cache(maxsize=2)
def _load_ov_reranker(model_dir: str, device: str = "GPU"):
    """OVModelForSequenceClassification cross-encoder on iGPU.

    Confirmed working on Intel Arc (bge-reranker-base INT8): 36.4 ms warm.
    """
    from optimum.intel import OVModelForSequenceClassification  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = OVModelForSequenceClassification.from_pretrained(model_dir, device=device)
    return tok, model


def _is_local_reranker(model_cfg: ModelConfig) -> bool:
    return getattr(model_cfg, "provider", "") == "local_reranker"


def _is_ov_reranker(model_cfg: ModelConfig) -> bool:
    """True when model uses in-process OV GPU cross-encoder scoring."""
    return (
        getattr(model_cfg, "provider", "") == "local_reranker"
        and getattr(model_cfg, "rerank_backend", "") == "openvino_gpu"
    )


def _ov_reranker_pairs(model_cfg: ModelConfig, pairs: list[tuple[str, str]]) -> tuple[list[float], bool, float]:
    """Score a batch of (query, doc) pairs via OVModelForSequenceClassification."""
    t0 = time.monotonic()
    try:
        import torch  # type: ignore
        model_dir = getattr(model_cfg, "ov_model_dir", model_cfg.effective_model_id)
        device = getattr(model_cfg, "ov_device", "GPU")
        tok, model = _load_ov_reranker(model_dir, device)
        qs, ds = zip(*pairs)
        inputs = tok(list(qs), list(ds), return_tensors="pt", padding=True,
                     truncation=True, max_length=512)
        out = model(**inputs)
        scores = torch.sigmoid(out.logits[:, 0]).tolist()
        return scores, True, (time.monotonic() - t0) * 1000
    except Exception as exc:
        logger.warning("OV GPU reranker failed: %s", exc)
        return [0.0] * len(pairs), False, (time.monotonic() - t0) * 1000


def score_pair_local(
    model_cfg: ModelConfig,
    query: str,
    doc: str,
) -> tuple[float, bool, float]:
    """Score one pair with an in-process dedicated cross-encoder reranker.

    Routes to OV GPU (iGPU) when rerank_backend=openvino_gpu is set.
    """
    if _is_ov_reranker(model_cfg):
        scores, ok, ms = _ov_reranker_pairs(model_cfg, [(query, doc)])
        return scores[0], ok, ms
    t0 = time.monotonic()
    try:
        model = _load_cross_encoder(model_cfg.effective_model_id)
        score = model.predict([(query, doc)], convert_to_numpy=True)[0]
        return float(score), True, (time.monotonic() - t0) * 1000
    except Exception as exc:
        logger.warning("local reranker failed: %s", exc)
        return 0.0, False, (time.monotonic() - t0) * 1000


def score_pair(
    model_cfg: ModelConfig,
    query: str,
    doc: str,
    *,
    max_tokens: int = 4,
) -> tuple[float, bool, float]:
    """Score one (query, doc) pair. Return (score, ok, latency_ms).

    Routes to the native ``/v1/rerank`` endpoint when the model is
    ``rerank_native`` (single-doc request → one relevance score), otherwise the
    generative yes/no proxy. For native models prefer :func:`score_query_native`,
    which batches the whole candidate list into one request.
    """
    if _is_local_reranker(model_cfg):
        return score_pair_local(model_cfg, query, doc)

    if getattr(model_cfg, "rerank_native", False):
        res = infer_rerank(model_cfg, query, [doc])
        if not res.ok or not res.scores:
            return 0.0, False, res.latency_ms
        return float(res.scores[0]), True, res.latency_ms

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


def score_query_native(
    model_cfg: ModelConfig,
    query: str,
    docs: Sequence[str],
) -> tuple[list[float], bool, float]:
    """Score a whole candidate list in one native ``/v1/rerank`` request.

    Returns (scores aligned to ``docs``, ok, latency_ms). One request per query
    (instead of one per pair) amortises HTTP + model-resident overhead and lets
    the backend batch the candidates internally — closer to how a real-time
    reranker is actually called. Only meaningful for ``rerank_native`` models.
    """
    if _is_local_reranker(model_cfg):
        if _is_ov_reranker(model_cfg):
            return _ov_reranker_pairs(model_cfg, [(query, d) for d in docs])
        t0 = time.monotonic()
        try:
            model = _load_cross_encoder(model_cfg.effective_model_id)
            scores = model.predict([(query, doc) for doc in docs], convert_to_numpy=True)
            return [float(s) for s in scores], True, (time.monotonic() - t0) * 1000
        except Exception as exc:
            logger.warning("local reranker batch failed: %s", exc)
            return [0.0] * len(docs), False, (time.monotonic() - t0) * 1000

    res = infer_rerank(model_cfg, query, list(docs))
    if not res.ok:
        return [0.0] * len(docs), False, res.latency_ms
    return [float(s) for s in res.scores], True, res.latency_ms
