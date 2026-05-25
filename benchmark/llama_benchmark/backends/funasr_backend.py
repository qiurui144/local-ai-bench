"""FunASRBackend：FunASR / SenseVoice 推理后端（阿里巴巴）。

FunASR 框架同时支持：
  - Paraformer（高性能中文 ASR，AISHELL-1 SOTA）
  - SenseVoice（多语言 + 情感识别，比 Whisper large-v3 快 15×）
  - UniASR、Conformer 等多种模型

安装::

    pip install funasr modelscope huggingface_hub
    # 若使用 GPU：pip install torch torchvision torchaudio --index-url ...

配置示例::

    # SenseVoice 多语言模型（推荐，速度快、中文强）
    - name: "sensevoice-small"
      type: whisper
      backend: funasr
      extra:
        model: "iic/SenseVoiceSmall"       # ModelScope 模型 ID
        device: "cuda:0"                   # "cpu" / "cuda:0"
        disable_update: true               # 离线模式，不检查更新

    # Paraformer 中文专项
    - name: "paraformer-zh"
      type: whisper
      backend: funasr
      extra:
        model: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        device: "cpu"
        vad_model: "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"  # 可选 VAD
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)


@register_backend(BackendType.FUNASR.value)
class FunASRBackend(AbstractModelBackend):
    """FunASR / SenseVoice 推理后端。

    通过 FunASR AutoModel 接口统一加载各类 ASR 模型。
    """

    def load(self) -> None:
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "请安装 funasr: pip install funasr\n"
                "模型下载需要: pip install modelscope 或 huggingface_hub"
            )

        model_id = self.config.extra.get("model")
        if not model_id and self.config.path:
            model_id = str(self.config.path)
        if not model_id:
            raise ValueError(
                f"模型 '{self.config.name}': funasr 后端需要 extra.model 或 path"
            )

        device = self.config.extra.get("device", "cpu")
        disable_update = self.config.extra.get("disable_update", False)
        vad_model = self.config.extra.get("vad_model", None)
        punc_model = self.config.extra.get("punc_model", None)

        logger.info(f"加载 FunASR 模型: {model_id} (device={device})")
        kwargs = {
            "model": model_id,
            "device": device,
            "disable_update": disable_update,
        }
        if vad_model:
            kwargs["vad_model"] = vad_model
        if punc_model:
            kwargs["punc_model"] = punc_model

        self._model = AutoModel(**kwargs)
        self._language = self.config.extra.get("language", "auto")
        logger.info(f"FunASR 模型加载完成: {self.config.name}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 1,
    ) -> Tuple[str, float]:
        """转录音频文件，返回 (文本, 推理时间ms)。"""
        self._require_loaded()

        lang = language or self._language
        with measure_ms() as elapsed:
            result = self._model.generate(
                input=audio_path,
                language=lang,
                use_itn=True,           # 反向文本归一化（数字/标点）
                batch_size_s=300,       # 流式分批处理（秒）
            )
        # FunASR 返回 list of dict，每个含 "text" 字段
        text = ""
        if result:
            if isinstance(result[0], dict):
                text = result[0].get("text", "")
            else:
                text = str(result[0])

        return text.strip(), elapsed[0]

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "model": self.config.extra.get("model", ""),
            "device": self.config.extra.get("device", "cpu"),
            "language": self.config.extra.get("language", "auto"),
        })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("FunASRBackend 未初始化，请先调用 load()")
