"""WeSpeakerBackend：基于 WeSpeaker 的说话人分离 / 说话人确认后端。

WeSpeaker 是 AISHELL 开源的说话人工具包，支持 ECAPA-TDNN、ResNet、CAM++ 等模型架构。

安装::

    pip install wespeaker

离线模型下载（以 CAM++ 为例）::

    # HuggingFace
    git clone https://huggingface.co/wenet-e2e/wespeaker-cnceleb-resnet34-LM wespeaker-cnceleb-resnet34-LM
    # 或使用 ModelScope
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download('wenet-e2e/wespeaker-cnceleb-resnet34-LM', cache_dir='/data/models/wespeaker')

使用说明：
- 说话人分离需配合 VAD（由 extra.vad_model 指定）
- extra.model 可为 HuggingFace/ModelScope 模型名 或 本地目录
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)

# 类型别名
Segment = Tuple[float, float, str]


@register_backend(BackendType.WESPEAKER.value)
class WeSpeakerBackend(AbstractModelBackend):
    """WeSpeaker 说话人分离 / 确认后端。

    extra 字段说明::

        extra:
          model: "wenet-e2e/wespeaker-cnceleb-resnet34-LM"  # 模型名或本地路径
          device: "cpu"          # "cpu" / "cuda" / "cuda:0"
          vad_model: null        # 可选：本地 VAD 模型路径；null = 内置简单 energy VAD
          clustering: "spectral" # "spectral"（默认）/ "ahc"
          threshold: 0.6         # 说话人聚类相似度阈值
    """

    def load(self) -> None:
        try:
            import wespeaker
        except ImportError:
            raise ImportError("请安装 wespeaker: pip install wespeaker")

        model_name = self.config.extra.get("model", "wespeaker-cnceleb-resnet34-LM")
        device = self.config.extra.get("device", "cpu")

        logger.info(f"加载 WeSpeaker 模型: {model_name} (device={device})")

        # 优先使用本地路径；否则从 HuggingFace 下载
        if self.config.path and self.config.path.exists():
            self._model = wespeaker.load_model_local(str(self.config.path))
        else:
            self._model = wespeaker.load_model(model_name)

        self._model.set_device(device)
        self._clustering = self.config.extra.get("clustering", "spectral")
        self._threshold = float(self.config.extra.get("threshold", 0.6))

        logger.info(f"WeSpeaker 模型加载完成: {self.config.name}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info(f"WeSpeaker 模型已释放: {self.config.name}")

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Tuple[List[Segment], float]:
        """说话人分离。

        Args:
            audio_path:    音频文件路径（WAV/FLAC/MP3）
            min_speakers:  最小说话人数量约束（None = 自动估计）
            max_speakers:  最大说话人数量约束（None = 自动估计）

        Returns:
            (segments, latency_ms)
            segments: [(start_sec, end_sec, speaker_id), ...]
        """
        self._require_loaded()
        with measure_ms() as elapsed:
            # WeSpeaker diarize API: 返回 [[start_ms, end_ms, speaker], ...]
            raw = self._model.diarize(audio_path)
        segments: List[Segment] = [
            (start / 1000.0, end / 1000.0, str(spk))
            for start, end, spk in raw
        ]
        return segments, elapsed[0]

    def get_embedding(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """提取说话人嵌入向量。

        Returns:
            (embedding, latency_ms)
            embedding: 1-D float32 numpy array（L2 归一化）
        """
        self._require_loaded()
        with measure_ms() as elapsed:
            emb = self._model.extract_embedding(audio_path)
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb, elapsed[0]

    def verify(self, audio1: str, audio2: str) -> Tuple[float, float]:
        """说话人确认：计算余弦相似度。

        Returns:
            (score, latency_ms)
            score: [-1, 1]，越高越相似
        """
        self._require_loaded()
        with measure_ms() as elapsed:
            emb1 = self._model.extract_embedding(audio1)
            emb2 = self._model.extract_embedding(audio2)
        emb1 = np.array(emb1, dtype=np.float32)
        emb2 = np.array(emb2, dtype=np.float32)
        score = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        return score, elapsed[0]

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "model": self.config.extra.get("model", ""),
            "device": self.config.extra.get("device", "cpu"),
            "clustering": self.config.extra.get("clustering", "spectral"),
        })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("WeSpeakerBackend 未初始化，请先调用 load()")
