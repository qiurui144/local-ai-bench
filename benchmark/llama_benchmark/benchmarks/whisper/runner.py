"""WhisperBenchmarkRunner。"""

from __future__ import annotations

from typing import List

from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
from benchmark.llama_benchmark.core.config import AppConfig, ModelConfig, ModelType
from benchmark.llama_benchmark.core.registry import create_backend, register_runner
from benchmark.llama_benchmark.core.result import TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_runner(ModelType.WHISPER.value)
class WhisperBenchmarkRunner(AbstractBenchmarkRunner):
    """Whisper 模型 benchmark runner。"""

    supported_model_types = [ModelType.WHISPER.value]

    def setup(self) -> None:
        self._backend = create_backend(self.model_config)
        if hasattr(self._backend, "configure"):
            self._backend.configure(self.app_config.ollama.base_url)
        self._backend.load()

    def run(self) -> List[TaskResult]:
        from benchmark.llama_benchmark.benchmarks.whisper.wer_cer import run_wer_cer

        whisper_cfg = self.app_config.benchmarks.whisper
        results: List[TaskResult] = []

        if whisper_cfg.wer_cer.enabled:
            logger.info(f"[{self.model_config.name}] 开始 WER/CER...")
            results.append(
                run_wer_cer(
                    self._backend,
                    whisper_cfg.wer_cer,
                    self.model_config.name,
                    dataset_name=whisper_cfg.dataset,
                    language=whisper_cfg.language,
                    beam_size=whisper_cfg.beam_size,
                )
            )

        return results

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
