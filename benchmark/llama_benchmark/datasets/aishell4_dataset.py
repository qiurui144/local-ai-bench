"""AISHELL-4 数据集加载器（说话人分离评测，中文会议）。

AISHELL-4 是 AISHELL 开源的中文会议录音数据集：
- 120 小时中文会议录音，8 路麦克风阵列
- 适用于中文说话人分离场景评测
- 官方：https://www.aishelltech.com/aishell_4

离线使用时期望目录结构::

    dataset_path/
      audio/    *.flac / *.wav
      textgrid/ *.TextGrid   （或 rttm/ *.rttm）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

Segment = Tuple[float, float, str]

AISHELL4_BUILTIN_SAMPLES = [
    {
        "audio_path": None,
        "duration": 100.0,
        "segments": [
            (0.0, 4.5, "S1"),
            (5.0, 9.8, "S2"),
            (10.2, 16.0, "S3"),
            (16.5, 22.0, "S1"),
            (22.5, 28.8, "S4"),
            (29.0, 35.0, "S2"),
            (35.5, 42.0, "S3"),
            (42.5, 50.0, "S1"),
        ],
        "session_id": "builtin_aishell4_001",
        "num_speakers": 4,
    },
    {
        "audio_path": None,
        "duration": 80.0,
        "segments": [
            (0.0, 6.0, "S1"),
            (6.5, 12.0, "S2"),
            (12.5, 20.0, "S3"),
            (20.5, 28.0, "S1"),
            (28.5, 36.0, "S2"),
            (36.5, 44.0, "S3"),
        ],
        "session_id": "builtin_aishell4_002",
        "num_speakers": 3,
    },
]


class AISHELL4Dataset(AbstractDataset):
    """AISHELL-4 中文会议说话人分离数据集。

    每条样本字段::

        {
            "audio_path": str | None,
            "duration": float,
            "segments": List[Segment],
            "session_id": str,
            "num_speakers": int,
        }
    """

    def __init__(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(split=split, num_samples=num_samples, **kwargs)

    def _load_hf(self) -> List[Dict[str, Any]]:
        """尝试从 HuggingFace 加载 AISHELL-4。"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("请安装 datasets: pip install datasets")

        ds = load_dataset("speechio/aishell4", split=self.split, trust_remote_code=True)
        samples = []
        for row in ds:
            audio_info = row.get("audio", {})
            segments = [
                (seg["start"], seg["end"], seg["speaker"])
                for seg in row.get("segments", [])
            ]
            samples.append({
                "audio_path": audio_info.get("path"),
                "duration": float(row.get("duration", 0.0)),
                "segments": segments,
                "session_id": row.get("session_id", ""),
                "num_speakers": len(set(s for _, _, s in segments)),
            })
        return samples

    def _load_from_path(self, path: Path) -> List[Dict[str, Any]]:
        """从本地目录加载（FLAC + RTTM 格式）。"""
        from benchmark.llama_benchmark.metrics.speaker import parse_rttm

        audio_dir = path / "audio" if (path / "audio").exists() else path
        rttm_dir = path / "rttm" if (path / "rttm").exists() else path

        samples = []
        for audio_file in sorted(audio_dir.glob("*.flac")) + sorted(audio_dir.glob("*.wav")):
            rttm_file = rttm_dir / f"{audio_file.stem}.rttm"
            if not rttm_file.exists():
                continue
            with open(rttm_file) as f:
                rttm_data = parse_rttm(f.read())
            segments = rttm_data.get(audio_file.stem) or next(iter(rttm_data.values()), [])
            duration = max((e for _, e, _ in segments), default=0.0) if segments else 0.0
            samples.append({
                "audio_path": str(audio_file),
                "duration": duration,
                "segments": segments,
                "session_id": audio_file.stem,
                "num_speakers": len(set(s for _, _, s in segments)),
            })
        return samples

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in AISHELL4_BUILTIN_SAMPLES]
