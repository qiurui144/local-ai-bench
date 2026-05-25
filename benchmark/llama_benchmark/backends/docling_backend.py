"""DoclingBackend：直接调用 docling 库进行文档解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)


@register_backend(BackendType.DOCLING.value)
class DoclingBackend(AbstractModelBackend):
    """Docling 文档解析后端。

    docling 使用内置模型（TableFormer 等），无需指定外部模型路径。
    config.path = null 时使用默认配置。

    安装::

        pip install docling
    """

    def load(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError("请安装 docling: pip install docling")

        logger.info("初始化 Docling DocumentConverter...")
        self._model = DocumentConverter()
        logger.info("Docling 初始化完成")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    def parse(self, document_path: str) -> Dict[str, Any]:
        """解析文档，返回结构化内容字典。

        Returns:
            {
                "text": str,           # 全文文本
                "tables": list,        # 表格列表
                "metadata": dict,      # 文档元信息
                "latency_ms": float,   # 解析耗时
            }
        """
        self._require_loaded()
        with measure_ms() as elapsed:
            result = self._model.convert(document_path)
            doc = result.document

        tables = []
        for table in doc.tables:
            cells = []
            for cell in table.data.grid:
                for c in cell:
                    cells.append({
                        "row": c.start_row_offset_idx,
                        "col": c.start_col_offset_idx,
                        "text": c.text,
                    })
            tables.append({"cells": cells})

        return {
            "text": doc.export_to_markdown(),
            "tables": tables,
            "metadata": {
                "num_pages": len(doc.pages) if hasattr(doc, "pages") else 0,
                "num_tables": len(tables),
            },
            "latency_ms": elapsed[0],
        }

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "ocr_engine": self.config.extra.get("ocr_engine", "default"),
        })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("DoclingBackend 未初始化，请先调用 load()")
