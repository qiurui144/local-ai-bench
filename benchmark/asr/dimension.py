"""asr 维度编排(CER/WER/RTF),自 run_benchmark 下沉。"""
from __future__ import annotations

from pathlib import Path

from benchmark.asr.runner import run_asr


def run_asr_dimension(model_cfg, asr_cfg: dict, root: Path) -> dict:
    """ASR CER/WER/RTF。缺 manifest 或 onnx 后端时 graceful BLOCKED。"""
    manifest = asr_cfg.get("manifest", "datasets/asr/manifest.jsonl")
    manifest_path = root / manifest
    return run_asr(
        model_cfg,
        manifest_path=manifest_path if manifest_path.exists() else None,
        audio_root=root / asr_cfg.get("audio_root", "datasets/asr") if asr_cfg.get("audio_root") else None,
        asr_model_dir=asr_cfg.get("model_dir"),
        num_samples=asr_cfg.get("num_samples"),
        thresholds=asr_cfg.get("thresholds"),
    )
