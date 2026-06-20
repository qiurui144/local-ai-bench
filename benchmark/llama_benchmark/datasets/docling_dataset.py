"""Docling 文档解析评测数据集加载器。

数据格式：每条样本包含 document_path（文档路径）和 expected（预期解析字段字典）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

DOCLING_BUILTIN_SAMPLES = [
    {
        "document_path": None,  # 内置样本无实际文档，测试框架跳过推理
        "document_type": "pdf",
        "expected": {
            "title": "Sample Report",
            "sections": ["Introduction", "Methods", "Results"],
            "tables": [{"rows": 3, "cols": 4}],
        },
    },
    {
        "document_path": None,
        "document_type": "docx",
        "expected": {
            "title": "Technical Specification",
            "sections": ["Overview", "Requirements", "Architecture"],
            "tables": [],
        },
    },
]


class DoclingDataset(AbstractDataset):
    """Docling 文档解析评测数据集。

    支持本地 PDF/DOCX/PPTX 文件目录扫描，每个文档需附带 .json 标注文件。
    标注格式：{"title": "...", "sections": [...], "tables": [...]}
    """

    def __init__(
        self,
        document_dir: Optional[Path] = None,
        document_types: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(num_samples=num_samples, dataset_path=document_dir, **kwargs)
        self.document_types = document_types or ["pdf", "docx", "pptx"]

    def _load_hf(self) -> List[Dict[str, Any]]:
        """Docling 评测通常使用本地文档，HuggingFace 路径不适用。"""
        raise ImportError("Docling 数据集需要本地文档目录，使用 document_dir 参数")

    def _load_from_path(self, path: Path) -> List[Dict[str, Any]]:
        import json

        samples = []
        for ext in self.document_types:
            for doc_file in sorted(path.glob(f"**/*.{ext}")):
                annotation_file = doc_file.with_suffix(".json")
                if annotation_file.exists():
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        expected = json.load(f)
                    samples.append(
                        {
                            "document_path": str(doc_file),
                            "document_type": ext,
                            "expected": expected,
                        }
                    )
        return samples

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return list(DOCLING_BUILTIN_SAMPLES)
