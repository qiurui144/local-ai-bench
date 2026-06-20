"""WhisperCppBackend：whisper.cpp Python 绑定推理后端。

whisper.cpp 是 C 语言实现，内存占用极低，支持 CPU / CUDA / Metal / OpenCL。
适合资源受限场景（嵌入式、边缘设备）或对比 faster-whisper 的内存/速度差异。

安装::

    pip install pywhispercpp
    # 或使用 CUDA 加速：
    # CMAKE_ARGS="-DWHISPER_CUDA=ON" pip install pywhispercpp --no-cache-dir

配置示例::

    - name: "whisper-cpp-small"
      type: whisper
      backend: whisper_cpp
      path: "/data/models/ggml-small.en.bin"
      extra:
        n_threads: 4        # CPU 线程数
        language: "en"
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)


@register_backend(BackendType.WHISPER_CPP.value)
class WhisperCppBackend(AbstractModelBackend):
    """whisper.cpp 推理后端（低内存，支持 CPU/CUDA/Metal）。"""

    def load(self) -> None:
        try:
            from pywhispercpp.model import Model
        except ImportError:
            raise ImportError(
                "请安装 pywhispercpp: pip install pywhispercpp\n"
                "CUDA 加速: CMAKE_ARGS='-DWHISPER_CUDA=ON' pip install pywhispercpp --no-cache-dir"
            )

        model_path = str(self.config.path)
        n_threads = int(self.config.extra.get("n_threads", 4))

        logger.info(f"加载 whisper.cpp 模型: {model_path} (threads={n_threads})")
        self._model = Model(model_path, n_threads=n_threads)
        logger.info(f"whisper.cpp 模型加载完成: {self.config.name}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        beam_size: int = 5,
    ) -> Tuple[str, float]:
        """转录音频文件，返回 (文本, 推理时间ms)。"""
        self._require_loaded()
        with measure_ms() as elapsed:
            segments = self._model.transcribe(audio_path, language=language or "en")
            text = "".join(seg.text for seg in segments).strip()
        return text, elapsed[0]

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "path": str(self.config.path),
            "n_threads": self.config.extra.get("n_threads", 4),
        })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("WhisperCppBackend 未初始化，请先调用 load()")
