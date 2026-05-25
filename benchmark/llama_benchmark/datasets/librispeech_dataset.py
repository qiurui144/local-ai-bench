"""LibriSpeech 数据集加载器（Whisper WER/CER 评测）。

数据格式：每条样本包含 audio_path（音频文件路径）和 transcription（参考转录文本）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

LIBRISPEECH_BUILTIN_SAMPLES = [
    {
        "audio_path": None,
        "transcription": "The quick brown fox jumps over the lazy dog.",
        "speaker_id": "builtin_001",
    },
    {
        "audio_path": None,
        "transcription": "She sells seashells by the seashore.",
        "speaker_id": "builtin_002",
    },
    {
        "audio_path": None,
        "transcription": "How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
        "speaker_id": "builtin_003",
    },
]


class LibriSpeechDataset(AbstractDataset):
    """LibriSpeech 语音识别评测数据集。

    使用 `openslr/librispeech_asr` HuggingFace 数据集，默认 `test.clean` 分割。
    当 audio_path 为 None 时，benchmark runner 跳过实际推理（用于单元测试）。
    """

    def __init__(
        self,
        split: str = "test.clean",
        num_samples: Optional[int] = None,
        local_audio_dir: Optional[Path] = None,
        **kwargs,
    ) -> None:
        super().__init__(split=split, num_samples=num_samples, **kwargs)
        self.local_audio_dir = local_audio_dir

    def _load_hf(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split=self.split,
            trust_remote_code=True,
        )
        samples = []
        for row in ds:
            # HuggingFace audio 字段包含 {'path': ..., 'array': ..., 'sampling_rate': ...}
            audio_info = row.get("audio", {})
            samples.append(
                {
                    "audio_path": audio_info.get("path"),
                    "audio_array": audio_info.get("array"),
                    "sampling_rate": audio_info.get("sampling_rate", 16000),
                    "transcription": row["text"].upper(),
                    "speaker_id": str(row.get("speaker_id", "")),
                }
            )
        return samples

    def _load_from_path(self, path: Path) -> List[Dict[str, Any]]:
        """从本地目录加载 FLAC/WAV 文件 + 同名 .txt 转录文本。"""
        samples = []
        for audio_file in sorted(path.glob("**/*.flac")) + sorted(path.glob("**/*.wav")):
            txt_file = audio_file.with_suffix(".txt")
            if txt_file.exists():
                transcription = txt_file.read_text(encoding="utf-8").strip()
                samples.append(
                    {
                        "audio_path": str(audio_file),
                        "transcription": transcription,
                        "speaker_id": audio_file.stem,
                    }
                )
        return samples

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return list(LIBRISPEECH_BUILTIN_SAMPLES)
