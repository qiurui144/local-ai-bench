"""AbstractModelBackend：所有模型后端的基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.config import ModelConfig


class AbstractModelBackend(ABC):
    """所有模型后端的抽象基类。

    各类型方法（transcribe / get_text / embed / rerank_score / generate）
    的默认实现抛出 NotImplementedError，子类按实际功能覆盖即可。
    这样 runner 可在运行前通过 hasattr 或 try/except 做能力检测，
    也让 IDE 和 mypy 知道接口的存在。
    """

    def __init__(self, config: "ModelConfig") -> None:
        self.config = config
        self._model: Any = None

    @abstractmethod
    def load(self) -> None:
        """加载模型到内存/GPU。"""

    @abstractmethod
    def unload(self) -> None:
        """释放模型资源。"""

    @contextmanager
    def managed(self) -> Generator["AbstractModelBackend", None, None]:
        """上下文管理器：自动管理模型生命周期。"""
        try:
            self.load()
            yield self
        finally:
            self.unload()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型元信息（量化方式、参数量等）。子类实现具体逻辑。"""
        return {
            "name": self.config.name,
            "type": self.config.type.value,
            "backend": self.config.backend.value,
        }

    # ── 能力接口（默认不支持，子类按需覆盖）─────────────────────────────────

    def transcribe(self, audio: "np.ndarray", language: Optional[str] = None, **kwargs):
        """ASR 转录。返回 (text, latency_ms) 或 (text, enc_ms, total_ms)。"""
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 ASR 转录（transcribe），"
            f"请使用 ASR 类型后端（如 whisper_onnx / sensevoice_onnx / funasr）"
        )

    def get_text(self, image: "np.ndarray", **kwargs):
        """OCR 识别，返回 (text, latency_ms)。"""
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 OCR 识别（get_text），"
            f"请使用 OCR 类型后端（如 rapidocr）"
        )

    def recognize(self, image: "np.ndarray", **kwargs):
        """OCR 识别，返回 (results, latency_ms)，results 为 [(box, text, conf), ...]。"""
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 OCR 识别（recognize），"
            f"请使用 OCR 类型后端（如 rapidocr）"
        )

    def embed(self, texts: List[str]) -> "np.ndarray":
        """Embedding，返回 shape=(N, dim) 的 numpy 数组。"""
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 Embedding，"
            f"请使用 Embedding 类型后端（如 sentence_transformers / ollama）"
        )

    def rerank_score(self, query: str, documents: List[str]) -> List[float]:
        """Rerank 评分，返回每个文档对 query 的相关性分数列表。"""
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 Rerank，"
            f"请使用 Rerank 类型后端（如 sentence_transformers / ollama）"
        )

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """LLM 文本生成。"""
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 LLM 生成（generate），"
            f"请使用 LLM 类型后端（如 ollama / llama_cpp / openai_compatible）"
        )

    def _require_loaded(self) -> None:
        """断言后端已加载，否则抛出明确错误。"""
        if not self.is_loaded:
            raise RuntimeError(
                f"{type(self).__name__} 尚未加载（is_loaded=False），请先调用 load()"
            )
