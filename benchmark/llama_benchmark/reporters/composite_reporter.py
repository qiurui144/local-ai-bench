"""CompositeReporter：同时驱动多个 reporter 输出多种格式。"""

from __future__ import annotations

from pathlib import Path
from typing import List, TYPE_CHECKING

from benchmark.llama_benchmark.reporters.base_reporter import AbstractReporter

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.result import BenchmarkSuiteResult


class CompositeReporter(AbstractReporter):
    def __init__(self, output_dir: Path, formats: List[str]) -> None:
        super().__init__(output_dir)
        self._reporters: List[AbstractReporter] = []
        self._build_reporters(formats)

    def _build_reporters(self, formats: List[str]) -> None:
        from benchmark.llama_benchmark.reporters.json_reporter import JsonReporter
        from benchmark.llama_benchmark.reporters.markdown_reporter import MarkdownReporter
        from benchmark.llama_benchmark.reporters.html_reporter import HtmlReporter

        factory = {
            "json": JsonReporter,
            "markdown": MarkdownReporter,
            "md": MarkdownReporter,
            "html": HtmlReporter,
        }
        for fmt in formats:
            cls = factory.get(fmt.lower())
            if cls:
                self._reporters.append(cls(self.output_dir))

    def generate(self, result: "BenchmarkSuiteResult") -> Path:
        paths = []
        for reporter in self._reporters:
            try:
                path = reporter.generate(result)
                paths.append(path)
            except Exception as e:
                from benchmark.llama_benchmark.utils.logging import get_logger
                get_logger(__name__).error(f"Reporter {reporter.__class__.__name__} 失败: {e}")
        return paths[0] if paths else self.output_dir
