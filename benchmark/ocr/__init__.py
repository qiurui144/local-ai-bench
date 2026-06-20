"""OCR benchmark dimension — CER / NED / latency, multi-backend.

Backends (priority order in "auto" mode):
  1. VitisAI EP  — AMD XDNA NPU (RyzenAI SDK required; Windows AMD Ryzen AI platform)
  2. rapidocr    — CPU ONNX via rapidocr-onnxruntime (install: pip install rapidocr-onnxruntime)
  3. paddleocr   — CPU PaddleOCR v4 (install: pip install paddleocr)

Force a backend via ``ocr_backend`` field in models.yaml (default "auto").
NPU path degrades gracefully to BLOCKED when VitisAI EP is unavailable.

Metrics:
  - CER  (Character Error Rate) — primary for Chinese printed text
  - NED  (Normalized Edit Distance) — partial-match scoring
  - latency p50/p95 (ms/image)

Dataset: ``datasets/ocr/manifest.jsonl`` + images in ``datasets/ocr/images/``.
"""
from __future__ import annotations

__all__ = ["datasets", "metrics", "runner"]
