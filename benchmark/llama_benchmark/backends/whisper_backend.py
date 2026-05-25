"""FasterWhisperBackend：faster-whisper 高精度 Whisper 推理后端（Fallback）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)


@register_backend(BackendType.FASTER_WHISPER.value)
class FasterWhisperBackend(AbstractModelBackend):
    """faster-whisper 推理后端（Whisper Fallback）。

    当 Ollama Whisper 模型 WER 不满足阈值时切换此后端。
    支持 CPU（int8）和 GPU（float16/int8_float16）。

    安装::

        pip install faster-whisper
    """

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("请安装 faster-whisper: pip install faster-whisper")

        model_path = str(self.config.path)
        compute_type = self.config.extra.get("compute_type", "int8")
        device = self.config.extra.get("device", "cpu")

        logger.info(f"加载 faster-whisper 模型: {model_path} (device={device})")
        self._model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )
        logger.info(f"faster-whisper 模型加载完成: {self.config.name}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info(f"faster-whisper 模型已释放: {self.config.name}")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        beam_size: int = 5,
    ) -> Tuple[str, float]:
        """转录音频文件，返回 (转录文本, 推理时间ms)。"""
        self._require_loaded()
        with measure_ms() as elapsed:
            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
            )
            text = " ".join(seg.text for seg in segments).strip()
        return text, elapsed[0]

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "path": str(self.config.path),
            "compute_type": self.config.extra.get("compute_type", "int8"),
            "device": self.config.extra.get("device", "cpu"),
        })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("FasterWhisperBackend 未初始化，请先调用 load()")
