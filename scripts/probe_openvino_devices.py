"""Run a tiny OpenVINO compile/infer probe on each visible device.

This is intended for Windows laptop validation after OpenVINO installation.
It records device enumeration plus whether a minimal FP32 model can compile
and run on CPU/GPU/NPU. Some NPUs only accept specific model shapes/opsets; a
compile failure is reported as a concrete blocker instead of hidden.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def _probe_device(ov, device: str) -> dict:
    try:
        opset = ov.opset8
        data = opset.parameter([1, 16], np.float32, name="data")
        relu = opset.relu(data)
        model = ov.Model([relu], [data], f"tiny_relu_{device}")
        core = ov.Core()
        started = time.perf_counter()
        compiled = core.compile_model(model, device)
        compile_ms = (time.perf_counter() - started) * 1000.0
        infer_input = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(1, 16)
        started = time.perf_counter()
        result = compiled([infer_input])[compiled.output(0)]
        infer_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": True,
            "compile_ms": round(compile_ms, 3),
            "infer_ms": round(infer_ms, 3),
            "output_sum": round(float(np.sum(result)), 6),
        }
    except Exception as exc:  # pragma: no cover - device/driver specific
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    try:
        import openvino as ov  # type: ignore
    except Exception as exc:
        out = {
            "ok": False,
            "error": f"openvino import failed: {type(exc).__name__}: {exc}",
            "python": sys.executable,
            "platform": platform.platform(),
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 2

    core = ov.Core()
    devices = list(core.available_devices)
    out = {
        "ok": True,
        "python": sys.executable,
        "platform": platform.platform(),
        "openvino_version": getattr(ov, "__version__", "unknown"),
        "available_devices": devices,
        "device_probe": {device: _probe_device(ov, device) for device in devices},
    }
    out["all_visible_devices_runnable"] = all(
        block.get("ok") for block in out["device_probe"].values()
    )
    Path("output/reports").mkdir(parents=True, exist_ok=True)
    report_path = Path("output/reports/openvino_device_probe.json")
    report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_visible_devices_runnable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
