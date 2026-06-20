"""DoclingBenchmarkRunner。"""

from __future__ import annotations

from typing import List

from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
from benchmark.llama_benchmark.core.config import ModelType
from benchmark.llama_benchmark.core.registry import create_backend, register_runner
from benchmark.llama_benchmark.core.result import TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_runner(ModelType.DOCLING.value)
class DoclingBenchmarkRunner(AbstractBenchmarkRunner):
    """Docling 文档解析 benchmark runner。"""

    supported_model_types = [ModelType.DOCLING.value]

    def setup(self) -> None:
        self._backend = create_backend(self.model_config)
        self._backend.load()

    def run(self) -> List[TaskResult]:
        from benchmark.llama_benchmark.benchmarks.docling.accuracy import run_parse_accuracy
        from benchmark.llama_benchmark.benchmarks.docling.throughput import run_throughput

        docling_cfg = self.app_config.benchmarks.docling
        task_results: List[TaskResult] = []

        if docling_cfg.parse_accuracy.enabled:
            logger.info(f"[{self.model_config.name}] 开始文档解析准确率...")
            task_results.append(
                run_parse_accuracy(
                    self._backend,
                    docling_cfg.parse_accuracy,
                    self.model_config.name,
                )
            )

        if docling_cfg.throughput.enabled:
            logger.info(f"[{self.model_config.name}] 开始文档解析吞吐量测试...")
            task_results.append(
                run_throughput(
                    self._backend,
                    docling_cfg.throughput,
                    self.model_config.name,
                )
            )

        return task_results

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
