"""Validate Windows acceleration resources for benchmark targets.

Runs locally on a Windows target after deployment. It reports Python package
availability and ONNX Runtime providers for CPU, DirectML/Windows GPU,
OpenVINO, and AMD VitisAI NPU paths.
"""
from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cmd(args: list[str]) -> dict:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout.strip()[:2000],
            "stderr": p.stderr.strip()[:2000],
        }
    except Exception as exc:  # pragma: no cover - host-specific probe
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    out: dict = {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            "onnxruntime": _has_module("onnxruntime"),
            "openvino": _has_module("openvino"),
            "rapidocr_onnxruntime": _has_module("rapidocr_onnxruntime"),
            "rapidocr_openvino": _has_module("rapidocr_openvino"),
            "sherpa_onnx": _has_module("sherpa_onnx"),
            "soundfile": _has_module("soundfile"),
            "paddleocr": _has_module("paddleocr"),
        },
        "providers": [],
        "provider_status": {},
        "commands": {
            "ollama_version": _cmd(["ollama", "--version"]),
            "ollama_list": _cmd(["ollama", "list"]),
            "ollama_ps": _cmd(["ollama", "ps"]),
        },
    }

    if platform.system().lower() == "windows":
        ryzenai_roots = [
            Path("C:/Program Files/RyzenAI"),
            Path("C:/Program Files/AMD/RyzenAI"),
            Path("C:/Program Files/AMD Ryzen AI"),
        ]
        found_roots = [str(p) for p in ryzenai_roots if p.exists()]
        out["ryzenai"] = {
            "candidate_roots": [str(p) for p in ryzenai_roots],
            "found_roots": found_roots,
            "files": {},
        }
        for root in ryzenai_roots:
            if not root.exists():
                continue
            for pattern in (
                "**/vitis-ai-runtime*.dll",
                "**/xrt-smi.exe",
                "**/xrt_core*.dll",
                "**/npu_sw_installer.exe",
                "**/onnxruntime_vitisai*.dll",
            ):
                matches = [str(p) for p in root.glob(pattern)]
                if matches:
                    out["ryzenai"]["files"][pattern] = matches[:20]
            break

    if out["packages"]["onnxruntime"]:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        out["providers"] = providers
        out["provider_status"] = {
            "cpu": "CPUExecutionProvider" in providers,
            "directml_or_windows_gpu": "DmlExecutionProvider" in providers,
            "openvino": "OpenVINOExecutionProvider" in providers,
            "vitisai_npu": "VitisAIExecutionProvider" in providers,
        }

    if out["packages"]["openvino"]:
        try:
            import openvino as ov  # type: ignore

            core = ov.Core()
            out["openvino_devices"] = list(core.available_devices)
        except Exception as exc:  # pragma: no cover - host-specific probe
            out["openvino_error"] = f"{type(exc).__name__}: {exc}"

    Path("output/reports").mkdir(parents=True, exist_ok=True)
    report_path = Path("output/reports/windows_accel_probe.json")
    report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
