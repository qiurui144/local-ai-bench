"""PyMuPDFBackend：规则解析基线（无模型，极快）。

PyMuPDF（fitz）是纯规则 PDF 解析器，无任何 AI 模型依赖：
  - 极快（100+ 页/秒，CPU 单核）
  - 无 GPU 依赖
  - 表格识别能力弱（仅能提取文字，无结构感知）

作为基线用途：
  1. 验证"引入 AI 模型的精度提升"是否值得性能代价
  2. 测量硬件 I/O 基线（排除模型推理的瓶颈）
  3. native PDF（无扫描件）场景快速文本提取

安装::

    pip install pymupdf

配置示例::

    - name: "pymupdf-baseline"
      type: docling
      backend: pymupdf
      extra:
        extract_tables: true    # 尝试启发式表格检测
        output_format: "text"   # "text" / "markdown" / "html"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)


@register_backend(BackendType.PYMUPDF.value)
class PyMuPDFBackend(AbstractModelBackend):
    """PyMuPDF 规则解析后端（基线，无模型依赖）。"""

    def load(self) -> None:
        try:
            import fitz  # noqa: F401
        except ImportError:
            raise ImportError("请安装 PyMuPDF: pip install pymupdf")

        self._extract_tables = bool(self.config.extra.get("extract_tables", True))
        self._output_format = self.config.extra.get("output_format", "markdown")
        logger.info(
            f"PyMuPDF 后端初始化: extract_tables={self._extract_tables}, "
            f"format={self._output_format}"
        )

    def unload(self) -> None:
        pass  # 无状态，无需释放

    def parse(self, document_path: str) -> Dict[str, Any]:
        """解析文档，返回结构化内容字典。"""

        doc_path = Path(document_path)
        suffix = doc_path.suffix.lower()

        with measure_ms() as elapsed:
            if suffix == ".pdf":
                result = self._parse_pdf(document_path)
            elif suffix in (".docx", ".doc"):
                result = self._parse_docx(document_path)
            else:
                # 尝试通用 fitz 打开
                result = self._parse_pdf(document_path)

        result["latency_ms"] = elapsed[0]
        return result

    def _parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """解析 PDF 文件。"""
        import fitz

        doc = fitz.open(pdf_path)
        text_parts = []
        tables: List[Dict[str, Any]] = []

        for page_num, page in enumerate(doc):
            if self._output_format == "markdown":
                text_parts.append(page.get_text("markdown"))
            elif self._output_format == "html":
                text_parts.append(page.get_text("html"))
            else:
                text_parts.append(page.get_text("text"))

            if self._extract_tables:
                # fitz 启发式表格检测（PyMuPDF 1.23+）
                try:
                    page_tables = page.find_tables()
                    for tbl in page_tables:
                        cells = []
                        for row_idx, row in enumerate(tbl.extract()):
                            for col_idx, cell_text in enumerate(row):
                                cells.append({
                                    "row": row_idx,
                                    "col": col_idx,
                                    "text": str(cell_text or ""),
                                })
                        tables.append({"cells": cells, "page": page_num})
                except AttributeError:
                    # PyMuPDF < 1.23 不支持 find_tables
                    pass

        doc.close()
        return {
            "text": "\n\n".join(text_parts),
            "tables": tables,
            "metadata": {
                "num_pages": len(doc),
                "num_tables": len(tables),
                "backend": "pymupdf",
            },
        }

    def _parse_docx(self, docx_path: str) -> Dict[str, Any]:
        """解析 DOCX（通过 fitz 转换后处理）。"""
        import fitz

        # fitz 支持直接打开 DOCX
        doc = fitz.open(docx_path)
        text = "\n\n".join(page.get_text("text") for page in doc)
        doc.close()
        return {
            "text": text,
            "tables": [],
            "metadata": {"num_pages": len(doc), "num_tables": 0, "backend": "pymupdf"},
        }

    def get_model_info(self) -> Dict[str, Any]:
        import fitz
        base = super().get_model_info()
        base.update({
            "pymupdf_version": fitz.__doc__,
            "extract_tables": self._extract_tables,
        })
        return base
