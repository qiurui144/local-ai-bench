"""OCR 维度编排 — 自 run_benchmark 下沉。"""
from __future__ import annotations

from pathlib import Path

from benchmark.ocr.runner import run_ocr


def run_ocr_dimension(model_cfg, ocr_cfg: dict, root: Path) -> dict:
    """OCR CER/NED/latency。缺 manifest 或后端时 graceful BLOCKED。"""
    manifest = ocr_cfg.get("manifest", "datasets/ocr/manifest.jsonl")
    manifest_path = root / manifest
    image_root_rel = ocr_cfg.get("image_root", "datasets/ocr")
    backend = getattr(model_cfg, "ocr_backend", None) or ocr_cfg.get("backend", "auto")
    model_dir = getattr(model_cfg, "ocr_model_dir", None) or ocr_cfg.get("model_dir")
    return run_ocr(
        model_cfg,
        manifest_path=manifest_path if manifest_path.exists() else None,
        image_root=root / image_root_rel if image_root_rel else None,
        backend=backend,
        ocr_model_dir=root / model_dir if model_dir else None,
        num_samples=ocr_cfg.get("num_samples"),
        thresholds=ocr_cfg.get("thresholds"),
    )
