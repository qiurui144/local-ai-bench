"""AbstractMetric：所有指标计算器的基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class AbstractMetric(ABC):
    """所有指标计算器的抽象基类。"""

    name: str = ""
    unit: str = ""
    higher_is_better: bool = True

    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> float:
        """计算整体指标值。"""

    def compute_per_sample(
        self, predictions: List[Any], references: List[Any]
    ) -> List[float]:
        """逐样本计算，默认实现（子类可优化）。"""
        return [self.compute([p], [r]) for p, r in zip(predictions, references)]
