"""UnstructuredBackend：Unstructured 文档解析后端。

Unstructured 是企业文档处理的主流选择：
  - fast 模式：纯规则，极快，适合 native PDF / DOCX
  - hi_res 模式：使用 Detectron2 + Tesseract，适合扫描件，精度高但慢
  - ocr_only 模式：仅 OCR，适合图片类文档

安装::

    pip install "unstructured[pdf,docx,pptx]"
    # hi_res 模式需要额外依赖：
    pip install "unstructured[pdf,docx,pptx,local-inference]"

配置示例::

    - name: "unstructured-fast"
      type: docling
      backend: unstructured
      extra:
        strategy: "fast"          # "fast" / "hi_res" / "ocr_only" / "auto"
        extract_images: false
        chunking_strategy: null   # null / "basic" / "by_title"

    - name: "unstructured-hi-res"
      type: docling
      backend: unstructured
      extra:
        strategy: "hi_res"
        hi_res_model_name: "yolox"  # "yolox" 或 "detectron2_onnx"
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


@register_backend(BackendType.UNSTRUCTURED.value)
class UnstructuredBackend(AbstractModelBackend):
    """Unstructured 文档解析后端（企业文档处理主流方案）。"""

    def load(self) -> None:
        try:
            import unstructured  # noqa: F401
        except ImportError:
            raise ImportError(
                '请安装 unstructured: pip install "unstructured[pdf,docx,pptx]"\n'
                'hi_res 模式需要: pip install "unstructured[pdf,docx,pptx,local-inference]"'
            )

        self._strategy = self.config.extra.get("strategy", "auto")
        self._extract_images = bool(self.config.extra.get("extract_images", False))
        self._chunking_strategy = self.config.extra.get("chunking_strategy", None)
        self._hi_res_model = self.config.extra.get("hi_res_model_name", "yolox")

        logger.info(f"Unstructured 后端初始化: strategy={self._strategy}")

    def unload(self) -> None:
        self._model = None

    def parse(self, document_path: str) -> Dict[str, Any]:
        """解析文档，返回结构化内容字典。"""
        from unstructured.partition.auto import partition

        doc_path = Path(document_path)
        suffix = doc_path.suffix.lower()

        kwargs: Dict[str, Any] = {
            "filename": document_path,
            "strategy": self._strategy,
            "extract_image_block_types": ["Image", "Table"] if self._extract_images else [],
        }
        if self._strategy == "hi_res":
            kwargs["hi_res_model_name"] = self._hi_res_model

        with measure_ms() as elapsed:
            elements = partition(**kwargs)

        # 分离表格和文本
        tables = []
        text_parts = []
        for elem in elements:
            elem_type = type(elem).__name__
            if elem_type == "Table":
                tables.append({
                    "cells": [],  # Unstructured 表格以 HTML 格式存储
                    "html": getattr(elem, "metadata", None) and
                            elem.metadata.text_as_html or "",
                    "text": str(elem),
                })
            else:
                text_parts.append(str(elem))

        return {
            "text": "\n\n".join(text_parts),
            "tables": tables,
            "metadata": {
                "num_elements": len(elements),
                "num_tables": len(tables),
                "strategy": self._strategy,
                "backend": "unstructured",
            },
            "latency_ms": elapsed[0],
        }

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({"strategy": self._strategy})
        return base
