"""SpeakerBenchmarkRunner：说话人分离 / 确认 benchmark runner。"""

from __future__ import annotations

from typing import List

from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
from benchmark.llama_benchmark.core.config import ModelType
from benchmark.llama_benchmark.core.registry import create_backend, register_runner
from benchmark.llama_benchmark.core.result import TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_runner(ModelType.SPEAKER.value)
class SpeakerBenchmarkRunner(AbstractBenchmarkRunner):
    """说话人分析 benchmark runner（分离 + 确认）。

    支持后端：
    - wespeaker  — WeSpeaker (ECAPA-TDNN / ResNet / CAM++)
    - pyannote   — pyannote.audio 3.x
    - nemo_speaker — NVIDIA NeMo MSDD
    """

    supported_model_types = [ModelType.SPEAKER.value]

    def setup(self) -> None:
        self._backend = create_backend(self.model_config)
        self._backend.load()

    def run(self) -> List[TaskResult]:
        from benchmark.llama_benchmark.benchmarks.speaker.diarization import run_diarization
        from benchmark.llama_benchmark.benchmarks.speaker.verification import run_verification

        speaker_cfg = self.app_config.benchmarks.speaker
        results: List[TaskResult] = []

        if speaker_cfg.diarization.enabled:
            logger.info(f"[{self.model_config.name}] 开始说话人分离评测 (DER)...")
            results.append(
                run_diarization(
                    self._backend,
                    speaker_cfg.diarization,
                    self.model_config.name,
                    collar=speaker_cfg.collar,
                    skip_overlap=speaker_cfg.skip_overlap,
                    datasets=speaker_cfg.datasets,
                )
            )

        if speaker_cfg.verification.enabled:
            logger.info(f"[{self.model_config.name}] 开始说话人确认评测 (EER)...")
            results.append(
                run_verification(
                    self._backend,
                    speaker_cfg.verification,
                    self.model_config.name,
                    datasets=speaker_cfg.datasets,
                )
            )

        return results

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
