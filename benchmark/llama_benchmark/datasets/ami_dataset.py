"""AMI 数据集加载器（说话人分离评测）。

AMI Meeting Corpus 是最常用的说话人分离评测数据集：
- 来源：100 小时英文会议录音，带详细 RTTM 标注
- 离线：提供内置虚拟样本（用于单元测试和系统集成验证）

供应链状态（2026-06-11 实测，datasets==4.5.0）：历史 HF 源 ``Edinburgh/ami``
在 Hub 已不存在/不可访问（API 401），且其旧加载路径依赖
``trust_remote_code=True``（远程代码执行面，已全仓禁用）。HF 直载当前
不可用，``_load_hf`` 一律响亮失败。候选替代镜像：``edinburghcstr/ami``
（非 gated 纯 parquet，sha ``46f28f2503e2ec48f8867a84eef356c70476beab``）—
但它是 utterance 级 ASR 标注，与本加载器的 diarization 段落 schema 不兼容，
接入前需单独评估。推荐路径：``dataset_path`` 指向本地 AMI 数据。

离线使用时期望目录结构::

    dataset_path/
      audio/    *.wav / *.flac
      rttm/     *.rttm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

# 类型别名
Segment = Tuple[float, float, str]

# 内置虚拟样本（无需真实音频，用于单元测试）
AMI_BUILTIN_SAMPLES = [
    {
        "audio_path": None,
        "duration": 120.0,
        "segments": [
            (0.0, 5.2, "SPK_A"),
            (5.5, 10.8, "SPK_B"),
            (11.0, 18.3, "SPK_A"),
            (18.7, 25.1, "SPK_C"),
            (25.5, 32.0, "SPK_B"),
            (32.3, 40.0, "SPK_A"),
            (40.4, 48.9, "SPK_C"),
            (49.2, 55.0, "SPK_B"),
        ],
        "meeting_id": "builtin_ami_001",
        "num_speakers": 3,
    },
    {
        "audio_path": None,
        "duration": 90.0,
        "segments": [
            (0.0, 8.0, "SPK_A"),
            (8.5, 15.2, "SPK_B"),
            (15.6, 22.0, "SPK_A"),
            (22.3, 30.0, "SPK_B"),
            (30.4, 40.0, "SPK_C"),
            (40.5, 50.0, "SPK_A"),
        ],
        "meeting_id": "builtin_ami_002",
        "num_speakers": 3,
    },
]


class AMIDataset(AbstractDataset):
    """AMI Meeting Corpus 说话人分离数据集。

    每条样本字段::

        {
            "audio_path": str | None,     # 音频路径（None = 单元测试虚拟样本）
            "duration": float,            # 音频时长（秒）
            "segments": List[Segment],    # 参考标注 [(start, end, speaker_id), ...]
            "meeting_id": str,            # 会议 ID
            "num_speakers": int,          # 参考说话人数
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
        """HF 直载当前不可用 — 响亮失败，绝不静默伪造数据（见模块 docstring）。"""
        raise RuntimeError(
            "AMI 的 HuggingFace 源 'Edinburgh/ami' 已不可用(API 401, 2026-06-11 实测)，"
            "且其历史加载路径依赖 trust_remote_code（供应链已禁用）。请改用 "
            "dataset_path 指向本地 AMI 目录（audio/*.wav|*.flac + rttm/*.rttm）；"
            "候选镜像 edinburghcstr/ami 为 utterance 级 ASR schema，接入前需评估。"
        )

    def _load_from_path(self, path: Path) -> List[Dict[str, Any]]:
        """从本地目录加载（WAV + RTTM 格式）。"""
        from benchmark.llama_benchmark.metrics.speaker import parse_rttm

        audio_dir = path / "audio" if (path / "audio").exists() else path
        rttm_dir = path / "rttm" if (path / "rttm").exists() else path

        samples = []
        for audio_file in sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.flac")):
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
                "meeting_id": audio_file.stem,
                "num_speakers": len(set(s for _, _, s in segments)),
            })
        return samples

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in AMI_BUILTIN_SAMPLES]
