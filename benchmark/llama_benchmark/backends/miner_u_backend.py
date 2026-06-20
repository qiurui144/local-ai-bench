"""MinerUBackend：MinerU 文档解析后端（opendatalab）。

MinerU 专为学术 PDF 设计，使用：
  - PDF-Extract-Kit（版面检测 + 公式识别）
  - PaddleOCR（扫描件 OCR）
  - TableMaster（表格结构识别）

GPU 加速显著（建议 VRAM >= 8GB），CPU 也可运行但较慢。

安装::

    pip install magic-pdf[full]
    # 下载模型权重（首次使用自动下载，或手动指定）:
    # mineru download

配置示例::

    - name: "mineru-default"
      type: docling
      backend: miner_u
      extra:
        backend: "pipeline"      # "pipeline"（推荐）或 "vlm-transformers"
        device: "cuda"           # "cuda" / "cpu"
        lang: null               # null = 自动检测；"ch" = 中文强制

MinerU VLM 模式（更高精度，需更多 VRAM）::

    extra:
      backend: "vlm-transformers"
      model: "opendatalab/MinerU2-7B"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)


@register_backend(BackendType.MINER_U.value)
class MinerUBackend(AbstractModelBackend):
    """MinerU 文档解析后端。"""

    def load(self) -> None:
        try:
            import magic_pdf  # noqa: F401
        except ImportError:
            raise ImportError(
                "请安装 MinerU: pip install magic-pdf[full]\n"
                "模型下载: mineru download"
            )

        self._backend_type = self.config.extra.get("backend", "pipeline")
        self._device = self.config.extra.get("device", "cpu")
        self._lang = self.config.extra.get("lang", None)
        logger.info(
            f"MinerU 后端初始化: type={self._backend_type}, device={self._device}"
        )

    def unload(self) -> None:
        self._model = None

    def parse(self, document_path: str) -> Dict[str, Any]:
        """解析文档，返回结构化内容字典。"""

        doc_path = Path(document_path)
        output_dir = doc_path.parent / f"_mineru_tmp_{doc_path.stem}"
        output_dir.mkdir(exist_ok=True)

        with measure_ms() as elapsed:
            if self._backend_type == "pipeline":
                result = self._run_pipeline(document_path, str(output_dir))
            else:
                result = self._run_vlm(document_path, str(output_dir))

        # 清理临时目录
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

        result["latency_ms"] = elapsed[0]
        return result

    def _run_pipeline(self, doc_path: str, output_dir: str) -> Dict[str, Any]:
        """使用 Pipeline 模式（推荐，速度/精度平衡）。"""
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter

        with open(doc_path, "rb") as f:
            pdf_bytes = f.read()

        image_writer = FileBasedDataWriter(output_dir)
        jso_useful_key = {"_pdf_type": "", "model_list": []}

        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()

        md_content = pipe.pipe_mk_markdown(image_writer, drop_mode="none")

        tables = []
        for page_block in pipe.model_list:
            for block in page_block.get("layout_dets", []):
                if block.get("category_id") == 5:  # table category
                    tables.append({"cells": [], "bbox": block.get("bbox", [])})

        return {
            "text": md_content,
            "tables": tables,
            "metadata": {
                "num_pages": len(pipe.model_list),
                "num_tables": len(tables),
                "backend": "mineru_pipeline",
            },
        }

    def _run_vlm(self, doc_path: str, output_dir: str) -> Dict[str, Any]:
        """使用 VLM 模式（更高精度，需要更多 VRAM）。"""
        from magic_pdf.pipe.VLMPipe import VLMPipe
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter

        with open(doc_path, "rb") as f:
            pdf_bytes = f.read()

        image_writer = FileBasedDataWriter(output_dir)
        pipe = VLMPipe(
            pdf_bytes,
            image_writer,
            model_name=self.config.extra.get("model", "opendatalab/MinerU2-7B"),
        )
        pipe.pipe_parse()
        md_content = pipe.pipe_mk_markdown(image_writer, drop_mode="none")

        return {
            "text": md_content,
            "tables": [],
            "metadata": {"backend": "mineru_vlm"},
        }

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "backend_type": self.config.extra.get("backend", "pipeline"),
            "device": self.config.extra.get("device", "cpu"),
        })
        return base
