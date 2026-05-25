"""MarkerBackend：Marker PDF→Markdown 解析后端（VikParuchuri）。

Marker 使用 surya 系列模型：
  - surya-detect（版面检测）
  - surya-ocr（OCR，支持 90+ 语言）
  - surya-order（阅读顺序）
  - surya-table（表格识别）

比 Nougat 快 10×，支持 CPU 和 GPU，适合通用文档（非学术 PDF 专用）。

安装::

    pip install marker-pdf

配置示例::

    - name: "marker-default"
      type: docling
      backend: marker
      extra:
        device: "cuda"      # "cuda" / "cpu" / "mps"（Apple Silicon）
        batch_multiplier: 2  # GPU 批大小乘数（越大越快但 VRAM 占用更多）
        langs: null         # null = 自动检测；["zh", "en"] = 强制语言
        ocr_all_pages: false # true = 强制所有页面 OCR（扫描件必须开启）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)


@register_backend(BackendType.MARKER.value)
class MarkerBackend(AbstractModelBackend):
    """Marker 文档解析后端（surya 模型驱动）。"""

    def load(self) -> None:
        try:
            import marker  # noqa: F401
        except ImportError:
            raise ImportError("请安装 marker: pip install marker-pdf")

        self._device = self.config.extra.get("device", "cpu")
        self._batch_multiplier = int(self.config.extra.get("batch_multiplier", 1))
        self._langs: Optional[List[str]] = self.config.extra.get("langs", None)
        self._ocr_all_pages = bool(self.config.extra.get("ocr_all_pages", False))

        logger.info(
            f"Marker 后端初始化: device={self._device}, "
            f"batch_multiplier={self._batch_multiplier}"
        )
        # Marker 在首次调用时懒加载 surya 模型，此处预热
        try:
            from marker.models import load_all_models
            self._models = load_all_models(device=self._device)
            logger.info("Marker surya 模型预加载完成")
        except Exception as e:
            logger.warning(f"Marker 模型预加载失败（将在首次调用时加载）: {e}")
            self._models = None

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, "_models") and self._models is not None:
            del self._models
            self._models = None

    def parse(self, document_path: str) -> Dict[str, Any]:
        """解析文档，返回结构化内容字典。"""
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models

        models = self._models or load_all_models(device=self._device)

        with measure_ms() as elapsed:
            full_text, images, out_meta = convert_single_pdf(
                document_path,
                models,
                batch_multiplier=self._batch_multiplier,
                langs=self._langs,
                ocr_all_pages=self._ocr_all_pages,
            )

        # 从 metadata 提取表格信息
        tables = []
        for block in out_meta.get("block_stats", {}).get("Table", []):
            tables.append({
                "cells": [],
                "page": block.get("page_idx", 0),
            })

        return {
            "text": full_text,
            "tables": tables,
            "metadata": {
                "num_pages": out_meta.get("pages", 0),
                "num_tables": len(tables),
                "languages": out_meta.get("languages", []),
                "backend": "marker",
            },
            "latency_ms": elapsed[0],
        }

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "device": self._device,
            "batch_multiplier": self._batch_multiplier,
        })
        return base
