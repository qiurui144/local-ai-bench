"""embedding 维度编排(检索质量 + 延迟/内存),自 run_benchmark 下沉。"""
from __future__ import annotations

from pathlib import Path

from benchmark.embedding.accuracy import run_embedding
from benchmark.embedding.datasets import load_retrieval
from benchmark.embedding.performance import run_embedding_performance


def run_embedding_dimension(model_cfg, emb_cfg: dict, root: Path) -> dict:
    """Embedding 检索质量 + 延迟/内存。数据集缺失时回退内置合成检索集。"""
    corpus = emb_cfg.get("corpus", "datasets/retrieval/cmteb_zh_subset.jsonl")
    corpus_path = root / corpus
    num_samples = emb_cfg.get("num_samples")
    thresholds = emb_cfg.get("thresholds")
    queries = load_retrieval(corpus_path if corpus_path.exists() else None,
                             num_samples=num_samples)
    out = run_embedding(model_cfg, queries, thresholds=thresholds)
    out["performance"] = run_embedding_performance(
        model_cfg, queries, samples=emb_cfg.get("latency_samples", 12)
    )
    return out
