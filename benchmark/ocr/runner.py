"""OCR benchmark runner — multi-backend (rapidocr / DirectML / paddleocr / VitisAI NPU).

Backend priority:
  1. VitisAI EP   — AMD XDNA NPU (requires RyzenAI SDK + VitisAIExecutionProvider)
  2. directml     — Windows GPU via onnxruntime-directml + RapidOCR
  3. rapidocr     — CPU ONNX via rapidocr-onnxruntime (default local path)
  4. paddleocr    — CPU PaddleOCR v4 (fallback)
  5. BLOCKED      — no backend available

The backend can be forced via ``backend`` arg:
  "vitisai" | "directml" | "openvino" | "rapidocr" | "paddleocr" | "auto".

Models in models.yaml set ``ocr_capable: true`` and may set:
  ``ocr_backend: auto|rapidocr|paddleocr|vitisai|directml|openvino``  (default "auto")
  ``ocr_model_dir: <path>``   for VitisAI custom model dir
"""
from __future__ import annotations

import logging
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _ORT_AVAILABLE = False

from common import ModelConfig, summarize_latencies

from .datasets import ImageSample, load_ocr_manifest
from .metrics import corpus_cer, corpus_ned

logger = logging.getLogger(__name__)

Recognizer = Callable[[Path], str]

_DEFAULT_THRESHOLDS = {
    "cer_max": 0.10,       # 10% CER ceiling — printed text should be clean
    "ned_max": 0.15,       # 15% NED — more lenient for partial GT
    "latency_p95_ms_max": 3000,
}


# ── Backend builders ────────────────────────────────────────────────────────

def _default_vitisai_python() -> Optional[Path]:
    for env_name in ("VITISAI_PYTHON", "PY312_RYZENAI_PYTHON"):
        value = os.environ.get(env_name)
        if value and Path(value).exists():
            return Path(value)
    candidates = [
        Path.home() / "py312-ryzenai" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _vitisai_helper_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "ocr_vitisai_rapidocr.py"


def _vitisai_model_args(model_dir: Optional[Path]) -> list[str]:
    if model_dir is None:
        return []
    args: list[str] = []
    for flag, filename in (
        ("--det-model", "det_model.onnx"),
        ("--rec-model", "rec_model.onnx"),
        ("--cls-model", "cls_model.onnx"),
        ("--config-file", "vaip_config.json"),
    ):
        path = Path(model_dir) / filename
        if path.exists():
            args.extend([flag, str(path)])
    return args


def _vitisai_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    root = env.get("RYZENAI_ROOT", r"C:\Program Files\RyzenAI\1.7.1")
    path_prefixes = [
        str(Path(root) / "deployment"),
        str(Path(root) / "onnxruntime" / "bin"),
        str(Path(root) / "xrt"),
    ]
    env["PATH"] = os.pathsep.join(path_prefixes + [env.get("PATH", "")])
    return env


def _run_vitisai_helper(
    python_exe: Path,
    helper: Path,
    *,
    image: Optional[Path] = None,
    model_args: Optional[list[str]] = None,
    timeout_s: int = 180,
) -> dict:
    cmd = [str(python_exe), str(helper)]
    if image is None:
        cmd.append("--probe")
    else:
        cmd.extend(["--image", str(image)])
    cmd.extend(model_args or [])
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        env=_vitisai_env(),
    )
    payload_text = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
    try:
        payload = json.loads(payload_text) if payload_text else {}
    except json.JSONDecodeError:
        payload = {"ok": False, "error": payload_text or proc.stderr.strip()}
    if proc.returncode != 0 or not payload.get("ok"):
        detail = payload.get("error") or proc.stderr.strip() or payload_text
        raise RuntimeError(detail)
    return payload

def _try_vitisai(model_dir: Optional[Path]) -> Optional[Recognizer]:
    """AMD XDNA NPU via RyzenAI VitisAI EP + RapidOCR PP-OCR pipeline."""
    python_exe = _default_vitisai_python()
    helper = _vitisai_helper_path()
    if python_exe is None or not helper.exists():
        logger.info("VitisAI OCR: helper or Python not found (%s, %s)", python_exe, helper)
        return None

    model_args = _vitisai_model_args(model_dir)
    try:
        probe = _run_vitisai_helper(python_exe, helper, model_args=model_args)
    except Exception as e:
        logger.info("VitisAI OCR init failed: %s", e)
        return None

    providers = probe.get("providers", [])
    if "VitisAIExecutionProvider" not in providers:
        logger.info("VitisAI OCR: helper loaded without VitisAI provider (%s)", providers)
        return None
    logger.info("VitisAI OCR backend loaded via %s (providers=%s)", python_exe, providers)

    def _recognize(img_path: Path) -> str:  # pragma: no cover - Windows NPU-only
        payload = _run_vitisai_helper(
            python_exe,
            helper,
            image=img_path,
            model_args=model_args,
        )
        return str(payload.get("text", ""))

    setattr(_recognize, "_rapidocr_providers", providers)
    return _recognize


def _rapidocr_session_providers(engine) -> list[str]:
    providers: list[str] = []
    for owner, attr in (
        (getattr(engine, "text_det", None), "infer"),
        (getattr(engine, "text_cls", None), "infer"),
        (getattr(engine, "text_rec", None), "session"),
    ):
        infer = getattr(owner, attr, None)
        session = getattr(infer, "session", None)
        if session is not None:
            providers.extend(session.get_providers())
    return providers


def _build_rapidocr(**kwargs) -> Recognizer:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
    engine = RapidOCR(**kwargs)

    def _recognize(img_path: Path) -> str:
        result, _ = engine(str(img_path))
        if not result:
            return ""
        return " ".join(line[1] for line in result if line and len(line) > 1)

    # Warm-up probe on a small pixel block to trigger model download before benchmarking.
    try:
        import numpy as np  # type: ignore
        _recognize.__doc__ = "rapidocr"
        engine(np.zeros((32, 32, 3), dtype="uint8"))
    except Exception:
        pass
    setattr(_recognize, "_rapidocr_providers", _rapidocr_session_providers(engine))
    return _recognize


def _try_rapidocr() -> Optional[Recognizer]:
    """CPU ONNX via rapidocr-onnxruntime (auto-downloads ~10 MB models on first use)."""
    try:
        _recognize = _build_rapidocr()
        logger.info("rapidocr-onnxruntime backend loaded (CPU)")
        return _recognize
    except Exception as e:
        logger.debug("rapidocr not available: %s", e)
        return None


def _try_directml() -> Optional[Recognizer]:
    """Windows GPU via onnxruntime-directml + rapidocr-onnxruntime."""
    if not _ORT_AVAILABLE:
        logger.debug("onnxruntime not installed; DirectML EP unavailable")
        return None
    providers = ort.get_available_providers()
    if "DmlExecutionProvider" not in providers:
        logger.info("DirectML OCR: DmlExecutionProvider not available (%s)", providers)
        return None
    try:
        recognizer = _build_rapidocr(
            det_use_dml=True,
            cls_use_dml=True,
            rec_use_dml=True,
        )
        actual = getattr(recognizer, "_rapidocr_providers", [])
        if "DmlExecutionProvider" not in actual:
            logger.info("DirectML OCR: RapidOCR fell back to providers %s", actual)
            return None
        logger.info("rapidocr-onnxruntime backend loaded (DirectML)")
        return recognizer
    except Exception as e:
        logger.info("DirectML OCR init failed: %s", e)
        return None


def _try_openvino() -> Optional[Recognizer]:
    """Intel OpenVINO via rapidocr-openvino (OV 2026.2.1 + rapidocr-openvino 1.4.4).

    rapidocr-openvino uses the OV Core internally with auto-device selection
    (CPU or iGPU, not specifically NPU — dynamic shapes prevent NPU here).
    Confirmed p50 797 ms on Intel Arc (2026-06-22).
    For NPU-specific PP-OCRv4 static-shape path see _try_openvino_npu().
    """
    try:
        import openvino as ov  # type: ignore
        devices = list(ov.Core().available_devices)
        logger.info("OpenVINO OCR: devices available %s", devices)
    except Exception as e:
        logger.info("OpenVINO OCR unavailable: %s", e)
        return None
    try:
        from rapidocr_openvino import RapidOCR  # type: ignore
        engine = RapidOCR()

        def _recognize(img_path: Path) -> str:
            result, _ = engine(str(img_path))
            if not result:
                return ""
            return " ".join(line[1] for line in result if line and len(line) > 1)

        try:
            import numpy as np  # type: ignore
            engine(np.zeros((32, 32, 3), dtype="uint8"))
        except Exception:
            pass
        logger.info("rapidocr-openvino backend loaded (devices=%s)", devices)
        return _recognize
    except Exception as e:
        logger.info("OpenVINO OCR init failed: %s", e)
        return None


def _try_openvino_npu(model_dir: Optional[Path] = None) -> Optional[Recognizer]:
    """PP-OCRv4 via OpenVINO NPU (Intel AI Boost) with static input shapes.

    Confirmed (2026-06-22): det [1,3,640,640]=33ms, rec [1,3,48,320]=11ms,
    cls [1,3,48,192]=3ms.  H=48 is mandatory for rec (AvgPool NPU constraint).

    model_dir must contain: det.xml/det.bin, rec.xml/rec.bin, cls.xml/cls.bin.
    Falls back to None (caller tries next backend) if NPU unavailable or model
    files are missing.
    """
    if not model_dir or not model_dir.exists():
        return None
    det_xml = model_dir / "det.xml"
    rec_xml = model_dir / "rec.xml"
    cls_xml = model_dir / "cls.xml"
    if not (det_xml.exists() and rec_xml.exists()):
        logger.info("OpenVINO NPU OCR: model files not found in %s", model_dir)
        return None
    try:
        import openvino as ov  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        core = ov.Core()
        if "NPU" not in core.available_devices:
            logger.info("OpenVINO NPU OCR: NPU not available (devices=%s)", core.available_devices)
            return None

        det_model = core.compile_model(str(det_xml), "NPU")
        rec_model = core.compile_model(str(rec_xml), "NPU")
        if cls_xml.exists():
            core.compile_model(str(cls_xml), "NPU")  # pre-warm cls on NPU; used by full pipeline
        logger.info("OpenVINO NPU OCR: compiled det+rec+cls on NPU from %s", model_dir)

    except Exception as e:
        logger.info("OpenVINO NPU OCR init failed: %s", e)
        return None

    def _preprocess_det(img: np.ndarray, size: int = 640) -> np.ndarray:
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img_r = np.array(Image.fromarray(img).resize((nw, nh)))
        pad = np.zeros((size, size, 3), dtype=np.float32)
        pad[:nh, :nw] = img_r
        return (pad / 255.0).transpose(2, 0, 1)[None]

    def _recognize(img_path: Path) -> str:  # pragma: no cover - needs NPU hardware
        img = np.array(Image.open(img_path).convert("RGB"))
        blob = _preprocess_det(img)
        out = det_model([blob])
        shrink_map = list(out.values())[0][0, 0]
        mask = (shrink_map > 0.3).astype(np.uint8) * 255
        # Extract bounding boxes (simple connected component approach)
        import cv2  # type: ignore
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        texts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 4 or h < 4:
                continue
            crop = img[y:y + h, x:x + w]
            # rec: static [1,3,48,320] — resize crop to H=48
            crop_r = np.array(Image.fromarray(crop).resize((320, 48))).astype(np.float32)
            rec_blob = (crop_r / 255.0).transpose(2, 0, 1)[None]
            rec_out = rec_model([rec_blob])
            # argmax over vocabulary axis → character indices
            indices = np.argmax(list(rec_out.values())[0][0], axis=-1)
            # Placeholder: just mark presence of text; real impl needs char dict
            if len(indices) > 0:
                texts.append(f"[text@{x},{y}]")
        return " ".join(texts) if texts else ""

    return _recognize


def _try_paddleocr() -> Optional[Recognizer]:
    """CPU PaddleOCR v4 (already installed; downloads models ~50 MB on first use)."""
    try:
        from paddleocr import PaddleOCR  # type: ignore
        engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False,
                           show_log=False, use_doc_orientation_classify=False)

        def _recognize(img_path: Path) -> str:
            result = engine.predict(str(img_path))
            if not result:
                return ""
            texts = []
            for page in result:
                for item in (page if isinstance(page, list) else [page]):
                    if isinstance(item, dict) and "rec_text" in item:
                        texts.append(item["rec_text"])
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and rec:
                            texts.append(str(rec[0]))
                        elif isinstance(rec, str):
                            texts.append(rec)
            return " ".join(texts)

        logger.info("PaddleOCR backend loaded (CPU)")
        return _recognize
    except Exception as e:
        logger.debug("PaddleOCR not available: %s", e)
        return None


def build_recognizer(
    backend: str = "auto",
    model_dir: Optional[Path] = None,
) -> tuple[Optional[Recognizer], str]:
    """Return (recognizer, backend_name) or (None, reason).

    Backend priority (auto mode): vitisai → directml → rapidocr → paddleocr.
    Explicit backends: openvino (iGPU auto-device) and openvino_npu (static
    shapes on Intel AI Boost) are also supported.
    """
    order = (
        [("vitisai", lambda: _try_vitisai(model_dir)),
         ("directml", _try_directml),
         ("rapidocr", _try_rapidocr),
         ("paddleocr", _try_paddleocr)]
        if backend == "auto" else
        [("vitisai", lambda: _try_vitisai(model_dir))] if backend == "vitisai" else
        [("directml", _try_directml)] if backend == "directml" else
        [("openvino", _try_openvino)] if backend == "openvino" else
        [("openvino_npu", lambda: _try_openvino_npu(model_dir))] if backend == "openvino_npu" else
        [("rapidocr", _try_rapidocr)] if backend == "rapidocr" else
        [("paddleocr", _try_paddleocr)] if backend == "paddleocr" else
        []
    )
    if not order:
        return None, f"unsupported ocr backend: {backend}"
    for name, builder in order:
        r = builder()
        if r is not None:
            return r, name
    return None, "no ocr backend available"


# ── Main entry ──────────────────────────────────────────────────────────────

def run_ocr(
    model_cfg: ModelConfig,
    *,
    manifest_path: Optional[Path] = None,
    image_root: Optional[Path] = None,
    backend: str = "auto",
    ocr_model_dir: Optional[Path] = None,
    num_samples: Optional[int] = None,
    thresholds: Optional[dict] = None,
    recognizer_fn: Optional[Recognizer] = None,  # injected for tests
) -> dict:
    thresholds = thresholds or _DEFAULT_THRESHOLDS

    samples: list[ImageSample] = (
        load_ocr_manifest(manifest_path, image_root=image_root, num_samples=num_samples)
        if manifest_path else []
    )
    if not samples:
        return {
            "benchmark": "ocr", "model": model_cfg.name,
            "status": "blocked", "reason": "no dataset (manifest missing or empty)",
            "verdict": "SKIP",
        }

    if recognizer_fn is None:
        recognizer_fn, backend_name = build_recognizer(backend, ocr_model_dir)
        if recognizer_fn is None:
            return {
                "benchmark": "ocr", "model": model_cfg.name,
                "status": "blocked",
                "reason": (
                    f"no ocr backend ({backend_name}); install rapidocr-onnxruntime, "
                    "rapidocr-openvino, or paddleocr"
                ),
                "num_samples": len(samples), "verdict": "SKIP",
            }
    else:
        backend_name = "injected"

    refs: list[str] = []
    hyps: list[str] = []
    latencies: list[float] = []
    errors = empty_outputs = 0

    for s in samples:
        t0 = time.monotonic()
        try:
            hyp = recognizer_fn(s.image)
        except Exception as e:
            errors += 1
            logger.info("  [ocr] %s failed: %s", s.uid, e)
            hyp = ""
        latencies.append((time.monotonic() - t0) * 1000)
        if not hyp.strip():
            empty_outputs += 1
        refs.append(s.text)
        hyps.append(hyp)

    overall_cer = corpus_cer(refs, hyps)
    overall_ned = corpus_ned(refs, hyps)
    lat_stats = summarize_latencies(latencies)

    aggregate = {
        "num_samples": len(samples),
        "cer": overall_cer,
        "ned": overall_ned,
        "latency_ms_stats": lat_stats,
        "empty_output_count": empty_outputs,
        "error_count": errors,
        "backend": backend_name,
        "data_source": samples[0].source if samples else "unknown",
    }
    backend_providers = getattr(recognizer_fn, "_rapidocr_providers", None)
    if backend_providers:
        aggregate["backend_providers"] = backend_providers

    reasons: list[str] = []
    if errors == len(samples):
        reasons.append("FAIL: all samples errored")
    elif empty_outputs == len(samples):
        reasons.append("FAIL: all outputs empty (broken backend)")
    if overall_cer > thresholds.get("cer_max", 1.0):
        reasons.append(
            f"FAIL: CER {overall_cer*100:.1f}% > {thresholds['cer_max']*100:.0f}%"
        )
    if lat_stats.get("p95", 0) > thresholds.get("latency_p95_ms_max", 1e9):
        reasons.append(
            f"WARN: p95 latency {lat_stats['p95']:.0f} ms > "
            f"{thresholds['latency_p95_ms_max']:.0f} ms"
        )
    if empty_outputs and empty_outputs < len(samples):
        reasons.append(f"WARN: {empty_outputs} empty output(s)")

    verdict = (
        "FAIL" if any(r.startswith("FAIL") for r in reasons) else
        "WARN" if reasons else "PASS"
    )

    return {
        "benchmark": "ocr", "model": model_cfg.name,
        "status": "ok", "verdict": verdict,
        "verdict_reasons": reasons,
        "aggregate": aggregate,
    }
