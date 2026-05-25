"""PyannoteBackend：基于 pyannote.audio 的说话人分离后端（工业标准）。

pyannote.audio 是最广泛使用的说话人分离框架，支持端到端神经网络 pipeline，
无需手动配置 VAD + 嵌入提取 + 聚类等步骤。

安装::

    pip install pyannote.audio

模型使用需接受 HuggingFace 用户协议::

    # 1. 登录 https://huggingface.co/pyannote/speaker-diarization-3.1 接受协议
    # 2. 生成 HuggingFace access token
    # 3. extra.hf_token: "hf_xxxxxxxxxxxxxxxxxxxx"

离线使用（已下载模型时）::

    extra:
      model_path: "/data/models/pyannote/speaker-diarization-3.1"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)

# 类型别名
Segment = Tuple[float, float, str]


@register_backend(BackendType.PYANNOTE.value)
class PyannoteBackend(AbstractModelBackend):
    """pyannote.audio 说话人分离后端。

    extra 字段说明::

        extra:
          model: "pyannote/speaker-diarization-3.1"  # 模型 ID（HuggingFace）
          model_path: null        # 本地模型路径；null = 在线加载
          hf_token: null          # HuggingFace access token（首次下载需要）
          device: "cpu"           # "cpu" / "cuda" / "mps"
          min_speakers: null      # 约束最小说话人数
          max_speakers: null      # 约束最大说话人数
    """

    def load(self) -> None:
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError("请安装 pyannote.audio: pip install pyannote.audio")

        import torch

        model_id = self.config.extra.get("model", "pyannote/speaker-diarization-3.1")
        model_path = self.config.extra.get("model_path") or (
            str(self.config.path) if self.config.path else None
        )
        hf_token = self.config.extra.get("hf_token")
        device_str = self.config.extra.get("device", "cpu")
        self._device = torch.device(device_str)

        logger.info(f"加载 pyannote pipeline: {model_id} (device={device_str})")

        if model_path:
            self._model = Pipeline.from_pretrained(model_path)
        else:
            if not hf_token:
                logger.warning(
                    "未提供 hf_token，在线加载 pyannote 模型可能失败。"
                    "请到 https://huggingface.co/pyannote/speaker-diarization-3.1 接受协议后设置 hf_token。"
                )
            self._model = Pipeline.from_pretrained(
                model_id,
                use_auth_token=hf_token,
            )

        self._model = self._model.to(self._device)
        logger.info(f"pyannote pipeline 加载完成: {self.config.name}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info(f"pyannote pipeline 已释放: {self.config.name}")

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Tuple[List[Segment], float]:
        """说话人分离。

        Args:
            audio_path:    音频文件路径
            min_speakers:  最小说话人数（None = pipeline 自动估计）
            max_speakers:  最大说话人数（None = pipeline 自动估计）

        Returns:
            (segments, latency_ms)
        """
        self._require_loaded()

        # 优先使用 config extra 中的约束，参数可以覆盖
        _min_spk = min_speakers or self.config.extra.get("min_speakers")
        _max_spk = max_speakers or self.config.extra.get("max_speakers")

        kwargs: Dict[str, Any] = {}
        if _min_spk is not None:
            kwargs["min_speakers"] = int(_min_spk)
        if _max_spk is not None:
            kwargs["max_speakers"] = int(_max_spk)

        with measure_ms() as elapsed:
            annotation = self._model(audio_path, **kwargs)

        segments: List[Segment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))

        segments.sort(key=lambda x: x[0])
        return segments, elapsed[0]

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "model": self.config.extra.get("model", "pyannote/speaker-diarization-3.1"),
            "device": self.config.extra.get("device", "cpu"),
        })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("PyannoteBackend 未初始化，请先调用 load()")
