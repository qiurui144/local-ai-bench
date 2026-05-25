"""Embedding 检索任务：NDCG@10，对齐 MTEB Retrieval 标准。"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    TaskResult,
)
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


def run_retrieval(
    backend,
    config: BenchmarkTaskConfig,
    model_name: str,
    dataset_name: str = "NQ",
) -> TaskResult:
    """执行向量检索 benchmark，NDCG@10 指标。"""
    start_time = time.time()

    # 优先通过 MTEB 标准评测
    try:
        return _run_mteb_retrieval(backend, config, model_name, dataset_name, start_time)
    except ImportError:
        logger.warning("mteb 未安装，使用内置检索测试集")
        return _run_simple_retrieval(backend, config, model_name, start_time)
    except Exception as e:
        logger.warning(f"MTEB 检索失败: {e}，回退到内置测试集")
        return _run_simple_retrieval(backend, config, model_name, start_time)


def _run_mteb_retrieval(
    backend, config, model_name, dataset_name, start_time
) -> TaskResult:
    """使用 MTEB 框架评测向量检索 NDCG@10。"""
    import mteb

    class _Encoder:
        """适配 MTEB encode 接口的包装器。"""

        def encode(self, sentences: List[str], batch_size: int = 32, **kwargs):
            return np.array(backend.embed(sentences))

        def encode_queries(self, queries: List[str], batch_size: int = 32, **kwargs):
            return self.encode(queries, batch_size, **kwargs)

        def encode_corpus(self, corpus: List[Dict], batch_size: int = 32, **kwargs):
            texts = [
                (doc.get("title", "") + " " + doc.get("text", "")).strip()
                for doc in corpus
            ]
            return self.encode(texts, batch_size, **kwargs)

    task = mteb.get_task(dataset_name)
    results = mteb.MTEB(tasks=[task]).run(_Encoder(), output_folder=None, verbosity=0)

    ndcg_10 = 0.0
    if results and results[0].scores:
        ndcg_10 = results[0].scores.get("test", {}).get("ndcg_at_10", 0.0)

    threshold = config.thresholds.get("ndcg_at_10", ThresholdConfig())
    status = BenchmarkStatus.PASS if threshold.check(ndcg_10) else BenchmarkStatus.FAIL

    return TaskResult(
        task_name=f"retrieval_{dataset_name}",
        model_name=model_name,
        metrics=[
            MetricResult(
                name="ndcg_at_10",
                value=round(ndcg_10, 4),
                higher_is_better=True,
                threshold=threshold.min_value,
                status=status,
            )
        ],
        num_samples=config.num_samples or 0,
        duration_seconds=time.time() - start_time,
        status=status,
        metadata={"dataset": dataset_name, "mode": "mteb"},
    )


def _run_simple_retrieval(
    backend, config, model_name, start_time
) -> TaskResult:
    """内置小型检索测试集（无需外部数据集）。

    5 条 query，10 篇文档，标注相关性 [0,1,2]（非相关/弱相关/强相关）。
    """
    queries = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "Capital city of France",
        "Python programming language features",
        "Climate change effects on oceans",
    ]
    corpus = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with many layers to learn representations.",
        "Photosynthesis converts sunlight into chemical energy in plant cells.",
        "Plants use carbon dioxide and water to produce glucose via photosynthesis.",
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe with rich cultural heritage.",
        "Python is a high-level, interpreted programming language known for readability.",
        "Python supports multiple programming paradigms including OOP and functional.",
        "Ocean temperatures are rising due to global warming and climate change.",
        "Climate change is causing more frequent extreme weather events worldwide.",
    ]
    # qrels[query_idx][doc_idx] = relevance
    qrels = [
        {0: 2, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 2, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 2},
    ]

    # 限制样本数
    num_samples = min(config.num_samples or len(queries), len(queries))
    queries = queries[:num_samples]
    qrels = qrels[:num_samples]

    # 编码
    query_embs = np.array(backend.embed(queries))
    corpus_embs = np.array(backend.embed(corpus))

    # L2 归一化后计算余弦相似度
    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
    c_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-8)
    scores_matrix = q_norm @ c_norm.T  # (num_queries, num_corpus)

    # 计算 NDCG@10
    from benchmark.llama_benchmark.metrics.ranking import NDCGMetric
    ndcg_metric = NDCGMetric(k=10)

    all_relevance: List[List[float]] = []
    for qi, row_scores in enumerate(scores_matrix):
        ranked_indices = np.argsort(-row_scores)
        relevance = [float(qrels[qi].get(int(di), 0)) for di in ranked_indices]
        all_relevance.append(relevance)

    ndcg_10 = ndcg_metric.compute(all_relevance, [])

    threshold = config.thresholds.get("ndcg_at_10", ThresholdConfig())
    status = BenchmarkStatus.PASS if threshold.check(ndcg_10) else BenchmarkStatus.FAIL

    return TaskResult(
        task_name="retrieval_builtin",
        model_name=model_name,
        metrics=[
            MetricResult(
                name="ndcg_at_10",
                value=round(ndcg_10, 4),
                higher_is_better=True,
                threshold=threshold.min_value,
                status=status,
            )
        ],
        num_samples=num_samples,
        duration_seconds=time.time() - start_time,
        status=status,
        metadata={"dataset": "builtin", "mode": "simple"},
    )
