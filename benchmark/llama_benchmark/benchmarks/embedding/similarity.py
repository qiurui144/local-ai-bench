"""Embedding 语义相似度任务：Spearman 相关系数，对齐 MTEB STS 标准。"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
from scipy.stats import spearmanr

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    TaskResult,
)
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


def run_similarity(
    backend,
    config: BenchmarkTaskConfig,
    model_name: str,
) -> TaskResult:
    """执行语义文本相似度 benchmark。"""
    start_time = time.time()

    # 尝试 MTEB STS 数据集
    try:
        return _run_mteb_sts(backend, config, model_name, start_time)
    except ImportError:
        logger.warning("mteb 未安装，使用内置 STS 测试集")
        return _run_simple_sts(backend, config, model_name, start_time)
    except Exception as e:
        return TaskResult(
            task_name="similarity",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=0.0,
            status=BenchmarkStatus.ERROR,
            error_message=str(e),
        )


def _run_mteb_sts(backend, config, model_name, start_time):
    import mteb

    class _Encoder:
        def encode(self, sentences, batch_size=32, **kwargs):
            return backend.embed(sentences)

    task = mteb.get_task("STSBenchmark")
    results = mteb.MTEB(tasks=[task]).run(_Encoder(), output_folder=None, verbosity=0)

    spearman = 0.0
    if results and results[0].scores:
        spearman = results[0].scores.get("test", {}).get("spearman_cosine", 0.0)

    threshold = config.thresholds.get("spearman_correlation", ThresholdConfig())
    status = BenchmarkStatus.PASS if threshold.check(spearman) else BenchmarkStatus.FAIL

    return TaskResult(
        task_name="similarity",
        model_name=model_name,
        metrics=[
            MetricResult(
                name="spearman_cosine",
                value=round(spearman, 4),
                higher_is_better=True,
                threshold=threshold.min_value,
                status=status,
            )
        ],
        num_samples=config.num_samples or 0,
        duration_seconds=time.time() - start_time,
        status=status,
    )


def _run_simple_sts(backend, config, model_name, start_time):
    """内置 STS 对（句对 + 相似度标签 0-5 归一化到 0-1）。"""
    pairs: List[Tuple[str, str, float]] = [
        ("A dog is running in the park.", "A dog runs in a park.", 0.9),
        ("The cat sat on the mat.", "A feline rested on a rug.", 0.7),
        ("I love programming.", "I hate coding.", 0.1),
        ("The weather is sunny today.", "It is a beautiful day.", 0.8),
        ("Stock prices fell sharply.", "The economy is doing great.", 0.1),
    ]

    sentences_a = [p[0] for p in pairs]
    sentences_b = [p[1] for p in pairs]
    gold_scores = [p[2] for p in pairs]

    embs_a = backend.embed(sentences_a)
    embs_b = backend.embed(sentences_b)

    # 计算余弦相似度
    norms_a = embs_a / (np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-8)
    norms_b = embs_b / (np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-8)
    pred_scores = (norms_a * norms_b).sum(axis=1).tolist()

    spearman_val, _ = spearmanr(pred_scores, gold_scores)
    spearman_val = float(spearman_val) if not np.isnan(spearman_val) else 0.0

    threshold = config.thresholds.get("spearman_correlation", ThresholdConfig())
    status = BenchmarkStatus.PASS if threshold.check(spearman_val) else BenchmarkStatus.FAIL

    return TaskResult(
        task_name="similarity",
        model_name=model_name,
        metrics=[
            MetricResult(
                name="spearman_cosine",
                value=round(spearman_val, 4),
                higher_is_better=True,
                status=status,
            )
        ],
        num_samples=len(pairs),
        duration_seconds=time.time() - start_time,
        status=status,
        metadata={"mode": "builtin_simple"},
    )
