"""数据集基类：定义统一的加载接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


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
            except (ImportError, Exception):
                samples = self._load_builtin()

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
