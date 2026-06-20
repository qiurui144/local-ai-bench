"""数据集基类：定义统一的加载接口。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# 合成回退的统一警示文案：进 TaskResult.metadata["warning"]，机器可读 + 人可读
SYNTHETIC_FALLBACK_WARNING = "builtin synthetic samples — NOT a real benchmark score"


def synthetic_fallback_metadata(dataset: Any) -> Dict[str, Any]:
    """数据集触发合成回退时返回机器可读 metadata 标记，否则返回空 dict。

    供 benchmark 在构造 TaskResult 时合并进 metadata，确保合成样本得分
    不会与真实 benchmark 分数混淆。
    """
    if getattr(dataset, "synthetic_fallback", False):
        return {"synthetic_fallback": True, "warning": SYNTHETIC_FALLBACK_WARNING}
    return {}


class AbstractDataset(ABC):
    """所有数据集加载器的基类。

    子类必须实现 `load()` 方法，返回可迭代的样本列表。
    每个样本为一个字典，包含 benchmark runner 所需的字段。
    """

    def __init__(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        dataset_path: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.num_samples = num_samples
        self.dataset_path = dataset_path
        self.seed = seed
        self._samples: Optional[List[Dict[str, Any]]] = None
        # True = load() 回退到了内置合成样本（WARN 日志之外的机器可读标记）
        self.synthetic_fallback: bool = False

    @abstractmethod
    def _load_hf(self) -> List[Dict[str, Any]]:
        """从 HuggingFace datasets 加载数据。"""
        ...

    @abstractmethod
    def _load_builtin(self) -> List[Dict[str, Any]]:
        """返回内置小型样本集（无需网络，用于单元测试和离线场景）。"""
        ...

    def load(self) -> List[Dict[str, Any]]:
        """加载数据集，优先 HuggingFace，失败则回退到内置样本。"""
        if self._samples is not None:
            return self._samples

        if self.dataset_path is not None:
            samples = self._load_from_path(self.dataset_path)
        else:
            try:
                samples = self._load_hf()
            except Exception as e:
                # 回退到内置合成样本必须"响亮":内置样本仅用于单元测试/离线冒烟，
                # 不能伪装成真实 benchmark 数据。
                samples = self._load_builtin()
                self.synthetic_fallback = True
                logger.warning(
                    "%s: HuggingFace 加载失败 (%s: %s)；回退到 %d 条内置合成样本 "
                    "— 该结果不是真实 benchmark 分数",
                    type(self).__name__,
                    type(e).__name__,
                    e,
                    len(samples),
                )

        if self.num_samples is not None:
            samples = samples[: self.num_samples]

        self._samples = samples
        return self._samples

    def _load_from_path(self, path: Path) -> List[Dict[str, Any]]:
        """从本地路径加载（子类可覆盖）。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持从本地路径加载")

    def __len__(self) -> int:
        return len(self.load())

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.load())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.load()[idx]
