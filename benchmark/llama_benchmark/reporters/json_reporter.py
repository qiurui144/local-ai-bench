"""JSON 格式报告生成。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.llama_benchmark.reporters.base_reporter import AbstractReporter

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.result import BenchmarkSuiteResult


class JsonReporter(AbstractReporter):
    def generate(self, result: "BenchmarkSuiteResult") -> Path:
        result.compute_summary()
        data = result.to_dict()
        output_path = self._get_output_path(result.run_id, ".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return output_path
