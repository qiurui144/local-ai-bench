"""rerank 维度编排(重排质量 + 单 pair 延迟),自 run_benchmark 下沉。"""
from __future__ import annotations

from pathlib import Path

from benchmark.embedding.datasets import load_retrieval
from benchmark.rerank.accuracy import run_rerank


def run_rerank_dimension(model_cfg, rr_cfg: dict, root: Path) -> dict:
    """Reranker 重排质量 + 单 pair 延迟。复用同一检索集。"""
    corpus = rr_cfg.get("corpus", "datasets/retrieval/cmteb_zh_subset.jsonl")
    corpus_path = root / corpus
    num_samples = rr_cfg.get("num_samples")
    thresholds = rr_cfg.get("thresholds")
    queries = load_retrieval(corpus_path if corpus_path.exists() else None,
                             num_samples=num_samples)
    return run_rerank(model_cfg, queries, thresholds=thresholds)
