"""排序指标：NDCG / MAP / MRR。"""

from __future__ import annotations

import math
from typing import List

from benchmark.llama_benchmark.core.base_metric import AbstractMetric


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """计算 NDCG@k。

    Args:
        relevance_scores: 按预测排序后的相关性标签列表（第 0 位为排名第 1 的结果）
        k: 截断位置

    Returns:
        NDCG@k ∈ [0, 1]
    """
    scores = relevance_scores[:k]
    dcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(scores)
    )
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal)
    )
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(relevance_scores: List[float]) -> float:
    """计算单个查询的 Average Precision。"""
    hits = 0
    ap = 0.0
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            hits += 1
            ap += hits / (i + 1)
    total_relevant = sum(1 for r in relevance_scores if r > 0)
    return ap / total_relevant if total_relevant > 0 else 0.0


def reciprocal_rank(relevance_scores: List[float]) -> float:
    """计算单个查询的 Reciprocal Rank（第一个相关结果的倒数排名）。"""
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


class NDCGMetric(AbstractMetric):
    """NDCG@k 指标。"""

    higher_is_better = True

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.name = f"ndcg_at_{k}"
        self.unit = ""

    def compute(
        self,
        predictions: List[List[float]],
        references: List[List[float]],
        **kwargs,
    ) -> float:
        """计算多个查询的平均 NDCG@k。

        Args:
            predictions: 每个查询按预测排序后的相关性分数列表
            references: 未使用（相关性已包含在 predictions 中）
        """
        if not predictions:
            return 0.0
        scores = [ndcg_at_k(p, self.k) for p in predictions]
        return sum(scores) / len(scores)


class MAPMetric(AbstractMetric):
    """Mean Average Precision。"""

    name = "map"
    unit = ""
    higher_is_better = True

    def compute(
        self,
        predictions: List[List[float]],
        references: List[List[float]],
        **kwargs,
    ) -> float:
        if not predictions:
            return 0.0
        scores = [average_precision(p) for p in predictions]
        return sum(scores) / len(scores)


class MRRMetric(AbstractMetric):
    """Mean Reciprocal Rank。"""

    name = "mrr"
    unit = ""
    higher_is_better = True

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.name = f"mrr_at_{k}"

    def compute(
        self,
        predictions: List[List[float]],
        references: List[List[float]],
        **kwargs,
    ) -> float:
        if not predictions:
            return 0.0
        scores = [reciprocal_rank(p[: self.k]) for p in predictions]
        return sum(scores) / len(scores)
