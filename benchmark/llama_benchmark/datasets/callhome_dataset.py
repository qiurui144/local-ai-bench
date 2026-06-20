"""CallHome 数据集加载器（说话人分离评测，电话场景）。

CallHome 是 LDC 发布的电话会话录音数据集：
- 双人电话通话，2 个说话人（少数 3-4 人）
- 常用于说话人分离系统的快速验证（样本较小）
- 子集：English, Chinese, German, Spanish, Arabic, Japanese
- 官方：LDC97S42（需 LDC 订阅）

供应链状态（2026-06-11 HF API 探测，datasets==4.5.0）：
``diarizers-community/callhome`` 为 **gated=auto**（需 HF token + 接受条款），
仓内纯 parquet、无加载脚本 — 即本就不需要 ``trust_remote_code``（死参数，
已移除）。revision pin 到探测时的 main commit SHA（PENDING-VERIFY：gated
导致无法匿名实测加载，仅 API 探测）。未授权时 HF 抛 gated 访问错误（信息
明确指向申请条款页），由基类响亮回退到内置合成样本并 WARN — 不会静默
冒充真实 benchmark 数据。

离线使用时期望目录结构::

    dataset_path/
      audio/    *.wav / *.sph
      rttm/     *.rttm
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

Segment = Tuple[float, float, str]

# gated 数据集无法匿名实测加载;SHA 为 2026-06-11 HF API 探测的 main commit。
CALLHOME_DEFAULT_REVISION = "17c8a153215aa7c50b805078fd6284ba81c2fc47"


def callhome_revision() -> str:
    # `or` 而非 get(k, default):env 置空串时也回落默认值
    return os.environ.get("CALLHOME_REVISION") or CALLHOME_DEFAULT_REVISION

CALLHOME_BUILTIN_SAMPLES = [
    {
        "audio_path": None,
        "duration": 60.0,
        "segments": [
            (0.0, 3.5, "SPK_0"),
            (3.8, 7.2, "SPK_1"),
            (7.5, 12.0, "SPK_0"),
            (12.3, 16.8, "SPK_1"),
            (17.0, 22.0, "SPK_0"),
            (22.4, 27.0, "SPK_1"),
            (27.3, 32.0, "SPK_0"),
            (32.3, 38.0, "SPK_1"),
            (38.3, 44.0, "SPK_0"),
            (44.3, 50.0, "SPK_1"),
        ],
        "call_id": "builtin_callhome_001",
        "num_speakers": 2,
        "language": "en",
    },
    {
        "audio_path": None,
        "duration": 50.0,
        "segments": [
            (0.0, 4.0, "SPK_0"),
            (4.3, 8.5, "SPK_1"),
            (8.8, 14.0, "SPK_0"),
            (14.3, 20.0, "SPK_1"),
            (20.3, 26.0, "SPK_0"),
            (26.3, 32.0, "SPK_1"),
            (32.3, 38.0, "SPK_0"),
            (38.3, 44.0, "SPK_1"),
        ],
        "call_id": "builtin_callhome_002",
        "num_speakers": 2,
        "language": "en",
    },
]


class CallhomeDataset(AbstractDataset):
    """CallHome 电话场景说话人分离数据集。

    每条样本字段::

        {
            "audio_path": str | None,
            "duration": float,
            "segments": List[Segment],
            "call_id": str,
            "num_speakers": int,
            "language": str,
        }
    """

    def __init__(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        language: str = "en",
        **kwargs,
    ) -> None:
        super().__init__(split=split, num_samples=num_samples, **kwargs)
        self.language = language

    def _load_hf(self) -> List[Dict[str, Any]]:
        """尝试从 HuggingFace 加载 CallHome（gated，需 HF token，见模块 docstring）。"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("请安装 datasets: pip install datasets")

        # diarizers-community/callhome 是社区维护的 RTTM 数据集（gated 纯 parquet）
        ds = load_dataset(
            "diarizers-community/callhome",
            self.language,
            split="data",
            revision=callhome_revision(),
        )
        samples = []
        for row in ds:
            audio_info = row.get("audio", {})
            segments = [
                (seg["start"], seg["end"], seg["speaker"])
                for seg in row.get("timestamps_start", [])
            ]
            # 不同版本的字段格式可能不同，做兼容处理
            if not segments and "segment_info" in row:
                segments = [
                    (seg["start"], seg["end"], seg["speaker"])
                    for seg in row["segment_info"]
                ]
            samples.append({
                "audio_path": audio_info.get("path"),
                "duration": float(row.get("duration", 0.0)),
                "segments": segments,
                "call_id": row.get("id", ""),
                "num_speakers": len(set(s for _, _, s in segments)),
                "language": self.language,
            })
        return samples

    def _load_from_path(self, path: Path) -> List[Dict[str, Any]]:
        """从本地目录加载（WAV/SPH + RTTM 格式）。"""
        from benchmark.llama_benchmark.metrics.speaker import parse_rttm

        audio_dir = path / "audio" if (path / "audio").exists() else path
        rttm_dir = path / "rttm" if (path / "rttm").exists() else path

        samples = []
        audio_files = (
            list(sorted(audio_dir.glob("*.wav")))
            + list(sorted(audio_dir.glob("*.flac")))
            + list(sorted(audio_dir.glob("*.sph")))
        )
        for audio_file in audio_files:
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
                "call_id": audio_file.stem,
                "num_speakers": len(set(s for _, _, s in segments)),
                "language": self.language,
            })
        return samples

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in CALLHOME_BUILTIN_SAMPLES]
