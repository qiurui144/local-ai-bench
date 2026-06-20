"""OCR manifest loader — image paths + ground-truth text."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageSample:
    image: Path
    text: str
    uid: str
    source: str = "unknown"  # synthetic | curated | dataset
    description: str = ""


def load_ocr_manifest(
    manifest_path: Path | str,
    *,
    image_root: Path | str | None = None,
    num_samples: Optional[int] = None,
) -> list[ImageSample]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        logger.info("OCR manifest not found: %s", manifest_path)
        return []

    root = Path(image_root) if image_root else manifest_path.parent
    samples: list[ImageSample] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            img_path = root / entry["image"]
            if not img_path.exists():
                logger.warning("Image not found, skipping: %s", img_path)
                continue
            samples.append(ImageSample(
                image=img_path,
                text=entry["text"],
                uid=entry.get("uid", str(len(samples))),
                source=entry.get("source", "unknown"),
                description=entry.get("description", ""),
            ))
            if num_samples and len(samples) >= num_samples:
                break

    logger.info("Loaded %d OCR samples from %s", len(samples), manifest_path)
    return samples
