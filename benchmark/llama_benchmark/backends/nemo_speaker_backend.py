"""NemoSpeakerBackend：基于 NVIDIA NeMo MSDD 的说话人分离后端。

NeMo MSDD（Multi-Scale Diarization Decoder）是 NVIDIA 开发的说话人分离系统，
在重叠语音和嘈杂环境下表现优越。

安装（需要 CUDA，建议 GPU >= 16GB）::

    pip install nemo_toolkit[asr]

模型权重由 NeMo 自动从 NVIDIA NGC 下载，或通过 model_path 指定本地 .nemo 文件::

    extra:
      model_path: "/data/models/nemo/diar_msdd_telephonic.nemo"

注意：NeMo ClusteringDiarizer 需要提供临时目录存放中间文件（extra.tmp_dir）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.timer import measure_ms

logger = get_logger(__name__)

# 类型别名
Segment = Tuple[float, float, str]

# 默认使用的预训练模型
_DEFAULT_MODEL = "diar_msdd_telephonic"


@register_backend(BackendType.NEMO_SPEAKER.value)
class NemoSpeakerBackend(AbstractModelBackend):
    """NeMo MSDD 说话人分离后端。

    extra 字段说明::

        extra:
          model: "diar_msdd_telephonic"   # NeMo 预训练模型名 或 .nemo 文件路径
          device: "cuda"                  # "cuda" / "cpu"
          max_num_speakers: 8             # 最大说话人数
          tmp_dir: null                   # 中间文件临时目录（null = 自动创建）
    """

    def load(self) -> None:
        try:
            from omegaconf import OmegaConf
        except ImportError:
            raise ImportError("请安装 omegaconf: pip install omegaconf")
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
        except ImportError:
            raise ImportError(
                "请安装 NeMo toolkit: pip install nemo_toolkit[asr]\n"
                "注意：NeMo 需要 CUDA 环境，详见 https://github.com/NVIDIA/NeMo"
            )

        model_name_or_path = self.config.extra.get("model", _DEFAULT_MODEL)
        device = self.config.extra.get("device", "cuda")
        max_speakers = int(self.config.extra.get("max_num_speakers", 8))

        # 使用 config.path 作为 .nemo 文件路径（如果指定）
        if self.config.path and self.config.path.exists():
            model_name_or_path = str(self.config.path)

        logger.info(f"加载 NeMo MSDD: {model_name_or_path} (device={device})")

        tmp_dir = self.config.extra.get("tmp_dir")
        if tmp_dir is None:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(prefix="nemo_diar_")
            self._tmp_dir = self._tmp_dir_obj.name
        else:
            self._tmp_dir_obj = None
            self._tmp_dir = tmp_dir
            os.makedirs(self._tmp_dir, exist_ok=True)

        # 构建 ClusteringDiarizer 配置
        cfg = OmegaConf.structured({
            "diarizer": {
                "manifest_filepath": None,
                "out_dir": self._tmp_dir,
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": False,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "onset": 0.8,
                        "offset": 0.6,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.4,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.1],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False,
                        "max_num_speakers": max_speakers,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
                    }
                },
                "msdd_model": {
                    "model_path": model_name_or_path,
                    "parameters": {
                        "use_speaker_model_from_ckpt": True,
                        "infer_batch_size": 25,
                        "sigmoid_threshold": [0.7],
                        "seq_eval_mode": False,
                        "split_infer": True,
                        "diar_eval_settings": [[0.25, True]],
                    },
                },
            }
        })

        self._model = ClusteringDiarizer(cfg=cfg)
        logger.info(f"NeMo MSDD 加载完成: {self.config.name}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        # 清理临时目录（仅自动创建的）
        if hasattr(self, "_tmp_dir_obj") and self._tmp_dir_obj is not None:
            try:
                self._tmp_dir_obj.cleanup()
            except Exception:
                pass
            self._tmp_dir_obj = None
        logger.info(f"NeMo MSDD 已释放: {self.config.name}")

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Tuple[List[Segment], float]:
        """说话人分离。

        NeMo ClusteringDiarizer 通过 manifest JSON 接受输入，
        输出 RTTM 文件，此处封装为 Segment 列表。

        Returns:
            (segments, latency_ms)
        """
        self._require_loaded()
        import json

        audio_path = str(audio_path)
        file_id = Path(audio_path).stem

        # 写入 manifest
        manifest_path = os.path.join(self._tmp_dir, f"{file_id}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None,
            }, f)
            f.write("\n")

        self._model._cfg.diarizer.manifest_filepath = manifest_path
        if max_speakers is not None:
            self._model._cfg.diarizer.clustering.parameters.max_num_speakers = max_speakers

        with measure_ms() as elapsed:
            self._model.diarize()

        # 解析输出 RTTM
        rttm_path = os.path.join(self._tmp_dir, "pred_rttms", f"{file_id}.rttm")
        segments: List[Segment] = []
        if os.path.exists(rttm_path):
            with open(rttm_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == "SPEAKER":
                        start = float(parts[3])
                        dur = float(parts[4])
                        spk = parts[7]
                        segments.append((start, start + dur, spk))

        segments.sort(key=lambda x: x[0])
        return segments, elapsed[0]

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "model": self.config.extra.get("model", _DEFAULT_MODEL),
            "device": self.config.extra.get("device", "cuda"),
            "max_num_speakers": self.config.extra.get("max_num_speakers", 8),
        })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("NemoSpeakerBackend 未初始化，请先调用 load()")
