"""Run RapidOCR backends in a subprocess for crash isolation."""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any


def _ensure_openvino_runtime_compat() -> None:
    """Expose the legacy openvino.runtime module expected by rapidocr-openvino."""
    if "openvino.runtime" in sys.modules:
        return
    try:
        import openvino as ov  # type: ignore
    except Exception:
        return
    runtime = types.ModuleType("openvino.runtime")
    for name in dir(ov):
        if not name.startswith("__"):
            setattr(runtime, name, getattr(ov, name))
    sys.modules["openvino.runtime"] = runtime


def _openvino_devices() -> list[str]:
    import openvino as ov  # type: ignore

    return list(ov.Core().available_devices)


def _recognize_with_openvino(image: Path | None) -> dict[str, Any]:
    devices = _openvino_devices()
    _ensure_openvino_runtime_compat()
    from rapidocr_openvino import RapidOCR  # type: ignore

    engine = RapidOCR()
    out: dict[str, Any] = {
        "ok": True,
        "backend": "openvino",
        "devices": devices,
        "python": sys.executable,
    }
    if image is None:
        out["mode"] = "probe"
        return out

    result, _ = engine(str(image))
    text = "" if not result else " ".join(line[1] for line in result if line and len(line) > 1)
    out["text"] = text
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["openvino"])
    parser.add_argument("--image")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    try:
        image = None if args.probe else Path(args.image or "")
        if image is not None and not args.image:
            raise ValueError("--image is required unless --probe is set")
        if args.backend == "openvino":
            payload = _recognize_with_openvino(image)
        else:  # pragma: no cover - argparse enforces choices
            raise ValueError(f"unsupported backend: {args.backend}")
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "backend": args.backend,
                    "error": f"{type(exc).__name__}: {exc}",
                    "python": sys.executable,
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
