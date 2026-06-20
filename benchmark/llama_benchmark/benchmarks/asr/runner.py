"""ASRBenchmarkRunner：多后端 ASR 精度-效率全面对比。

注册后可通过 model type=asr 触发，或直接调用 compare_backends()。
支持：whisper_onnx (tiny/base/small), sensevoice_onnx, faster_whisper。
"""

from __future__ import annotations

from typing import List

from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
from benchmark.llama_benchmark.core.config import ModelType
from benchmark.llama_benchmark.core.registry import create_backend, register_runner
from benchmark.llama_benchmark.core.result import TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_runner(ModelType.ASR.value)
class ASRBenchmarkRunner(AbstractBenchmarkRunner):
    """ASR 多后端 RTF/延迟评测 runner。"""

    supported_model_types = [ModelType.ASR.value]

    def setup(self) -> None:
        self._backend = create_backend(self.model_config)
        if hasattr(self._backend, "configure"):
            self._backend.configure(self.app_config.ollama.base_url)
        self._backend.load()

    def run(self) -> List[TaskResult]:
        from benchmark.llama_benchmark.benchmarks.asr.rtf import run_rtf_benchmark

        asr_cfg = self.app_config.benchmarks.asr
        results: List[TaskResult] = []

        if asr_cfg.rtf.enabled:
            logger.info(f"[{self.model_config.name}] 开始 RTF 评测 ...")
            for lang in asr_cfg.languages:
                r = run_rtf_benchmark(
                    backend=self._backend,
                    audio_durations_s=asr_cfg.audio_durations_s,
                    num_warmup=asr_cfg.num_warmup,
                    num_runs=asr_cfg.num_runs,
                    language=lang,
                    model_name=self.model_config.name,
                )
                r.task_name = f"asr_rtf_{lang}"
                results.append(r)
                rtf = r.metrics.get("avg_rtf")
                logger.info(
                    f"[{self.model_config.name}] lang={lang} RTF={rtf:.3f}" if rtf else
                    f"[{self.model_config.name}] lang={lang} 评测失败"
                )

        return results

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
