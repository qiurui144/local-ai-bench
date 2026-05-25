"""OCRBenchmarkRunner：多分辨率 OCR 精度-效率对比。

注册后可通过 model type=ocr 触发，或直接调用 run_ocr_accuracy()。
"""

from __future__ import annotations

from typing import List

from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
from benchmark.llama_benchmark.core.config import AppConfig, ModelConfig, ModelType
from benchmark.llama_benchmark.core.registry import create_backend, register_runner
from benchmark.llama_benchmark.core.result import TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_runner(ModelType.OCR.value)
class OCRBenchmarkRunner(AbstractBenchmarkRunner):
    """OCR 多分辨率评测 runner。"""

    supported_model_types = [ModelType.OCR.value]

    def setup(self) -> None:
        self._backend = create_backend(self.model_config)
        if hasattr(self._backend, "configure"):
            self._backend.configure(self.app_config.ollama.base_url)
        self._backend.load()

    def run(self) -> List[TaskResult]:
        from benchmark.llama_benchmark.benchmarks.ocr.accuracy import run_ocr_accuracy

        ocr_cfg = self.app_config.benchmarks.ocr
        results: List[TaskResult] = []

        if ocr_cfg.accuracy.enabled:
            logger.info(f"[{self.model_config.name}] 开始 OCR 精度-延迟评测 ...")
            scale_results = run_ocr_accuracy(
                backend=self._backend,
                input_scales=ocr_cfg.input_scales,
                num_warmup=ocr_cfg.num_warmup,
                num_runs=ocr_cfg.num_runs,
                model_name=self.model_config.name,
            )
            results.extend(scale_results)

        return results

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
