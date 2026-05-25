"""SentenceTransformersBackend：直连 HuggingFace 模型进行 Embedding / Rerank。

不需要额外服务进程，直接加载模型权重在当前进程中推理。
适合作为 Embedding 和 Rerank 精度基线，或 CPU/GPU 吞吐对比。

安装::

    pip install sentence-transformers

配置示例（models.yaml）::

    - name: "bge-m3-direct"
      type: embedding
      backend: sentence_transformers
      path: "/data/models/bge-m3"         # 本地路径
      # 或不填 path，直接用 HuggingFace 模型 ID：
      extra:
        model_name_or_path: "BAAI/bge-m3"
        device: "cpu"                      # "cpu" / "cuda" / "mps"
        batch_size: 32

    - name: "bge-reranker-v2-m3-direct"
      type: rerank
      backend: sentence_transformers
      extra:
        model_name_or_path: "BAAI/bge-reranker-v2-m3"
        device: "cuda"
        use_cross_encoder: true
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)


@register_backend(BackendType.SENTENCE_TRANSFORMERS.value)
class SentenceTransformersBackend(AbstractModelBackend):
    """sentence-transformers 直连推理后端。

    extra.use_cross_encoder=true 时使用 CrossEncoder（用于 Rerank）。
    否则使用 SentenceTransformer（用于 Embedding）。
    """

    def load(self) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        model_path = (
            str(self.config.path)
            if self.config.path
            else self.config.extra.get("model_name_or_path", self.config.name)
        )
        device = self.config.extra.get("device", "cpu")
        use_cross_encoder = self.config.extra.get("use_cross_encoder", False)

        logger.info(
            f"加载 sentence-transformers 模型: {model_path} "
            f"(device={device}, cross_encoder={use_cross_encoder})"
        )

        if use_cross_encoder:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_path, device=device)
            self._is_cross_encoder = True
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_path, device=device)
            self._is_cross_encoder = False

        self._batch_size = int(self.config.extra.get("batch_size", 32))
        logger.info(f"sentence-transformers 模型加载完成: {self.config.name}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量 Embedding，返回 shape=(N, dim) 的 numpy 数组。"""
        self._require_loaded()
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)

    def rerank_score(self, query: str, documents: List[str]) -> List[float]:
        """Rerank 评分。

        CrossEncoder 模式：直接输出 relevance score。
        Bi-encoder 模式：cosine 相似度。
        """
        self._require_loaded()

        if self._is_cross_encoder:
            pairs = [[query, doc] for doc in documents]
            scores = self._model.predict(pairs, batch_size=self._batch_size)
            return scores.tolist()

        # Bi-encoder cosine 相似度
        query_emb = self.embed([query])[0]
        doc_embs = self.embed(documents)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        return (d_norm @ q_norm).tolist()

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "device": self.config.extra.get("device", "cpu"),
            "use_cross_encoder": self.config.extra.get("use_cross_encoder", False),
        })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("SentenceTransformersBackend 未初始化，请先调用 load()")
