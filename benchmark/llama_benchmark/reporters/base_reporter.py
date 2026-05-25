"""AbstractReporter。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.result import BenchmarkSuiteResult


class AbstractReporter(ABC):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate(self, result: "BenchmarkSuiteResult") -> Path:
        """生成报告，返回输出文件路径。"""

    def _get_output_path(self, run_id: str, suffix: str) -> Path:
        return self.output_dir / f"report_{run_id}{suffix}"
