"""ASR dataset loader (manifest of audio path + reference transcript).

A manifest is a JSONL file, one object per line:
``{"audio": "rel/or/abs.wav", "text": "参考文本", "duration": 3.4}``
(``duration`` optional — computed from the WAV header at run time if absent).

Audio files themselves are NOT shipped (they are large and may carry PII /
licensing constraints, same policy as ``fixtures/``). The repo ships a manifest
*template* and documents how to point it at an AISHELL / local subset. When no
manifest exists the loader returns an empty list and the runner reports the
dimension BLOCKED (needs dataset) rather than crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AudioSample:
    audio: Path
    text: str
    duration_s: float = 0.0
    uid: str = ""
    source: str = "custom"


def load_asr_manifest(
    path: Path | str,
    *,
    audio_root: Path | str | None = None,
    num_samples: Optional[int] = None,
) -> list[AudioSample]:
    """Load an ASR manifest JSONL. Missing file → empty list (BLOCKED upstream)."""
    path = Path(path)
    if not path.exists():
        return []
    root = Path(audio_root) if audio_root else path.parent
    out: list[AudioSample] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio = Path(obj["audio"])
            if not audio.is_absolute():
                audio = root / audio
            out.append(AudioSample(
                audio=audio,
                text=obj["text"],
                duration_s=float(obj.get("duration", 0.0)),
                uid=str(obj.get("uid", i)),
                source=obj.get("source", "custom"),
            ))
            if num_samples is not None and len(out) >= num_samples:
                break
    return out


def wav_duration_s(path: Path) -> float:
    """Read a PCM WAV's duration from its header (stdlib ``wave``); 0.0 on error."""
    try:
        import wave

        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / rate if rate else 0.0
    except Exception:
        return 0.0
