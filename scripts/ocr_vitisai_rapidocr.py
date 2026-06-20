"""Run RapidOCR PP-OCR through AMD RyzenAI ONNX Runtime VitisAI EP.

This helper is intended to be executed with the RyzenAI-compatible CPython
3.12 environment. It keeps the benchmark runner independent from that Python
ABI while still reusing RapidOCR's PP-OCR pre/post processing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _prepend_ryzenai_path() -> None:
    root = os.environ.get("RYZENAI_ROOT") or r"C:\Program Files\RyzenAI\1.7.1"
    paths = [
        *(Path(p) for p in glob.glob(r"C:\WINDOWS\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*")),
        Path(root) / "deployment",
        Path(root) / "onnxruntime" / "bin",
        Path(root) / "xrt",
        Path.home() / "amd-npu-rai161" / "npu_mcdm_stack_prod",
    ]
    for path in paths:
        if path.exists() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(str(p) for p in paths if p.exists()) + os.pathsep + existing
    os.environ.setdefault("RYZEN_AI_INSTALLATION_PATH", str(Path(root)))


def _npu_provider_options(config_file: str | None) -> dict[str, Any]:
    root = Path(os.environ.get("RYZENAI_ROOT") or r"C:\Program Files\RyzenAI\1.7.1")
    opts: dict[str, Any] = {}
    if config_file and Path(config_file).exists():
        opts["config_file"] = str(Path(config_file))

    try:
        out = subprocess.check_output(
            r"pnputil /enum-devices /bus PCI /deviceids",
            shell=True,
            text=True,
            errors="replace",
        )
    except Exception:
        out = ""

    if r"PCI\VEN_1022&DEV_1502&REV_00" in out:
        xclbin = root / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin"
        if xclbin.exists():
            opts.update(
                {
                    "target": "X1",
                    "xlnx_enable_py3_round": 0,
                    "xclbin": str(xclbin),
                }
            )
    return opts


def _patch_rapidocr_for_vitisai(config_file: str | None) -> None:
    import rapidocr_onnxruntime.utils.infer_engine as infer_engine  # type: ignore

    def _get_ep_list(self) -> list[tuple[str, dict[str, Any]]]:
        providers = infer_engine.get_available_providers()
        if "VitisAIExecutionProvider" not in providers:
            raise RuntimeError(f"VitisAIExecutionProvider not available: {providers}")

        self.use_cuda = False
        self.use_directml = False
        self.use_vitisai = True

        vai_options = _npu_provider_options(config_file)
        cpu_options = {"arena_extend_strategy": "kSameAsRequested"}
        return [("VitisAIExecutionProvider", vai_options), ("CPUExecutionProvider", cpu_options)]

    def _verify_providers(self) -> None:
        session_providers = self.session.get_providers()
        if not session_providers or session_providers[0] != "VitisAIExecutionProvider":
            raise RuntimeError(
                "RapidOCR did not bind to VitisAIExecutionProvider; "
                f"actual providers={session_providers}"
            )

    infer_engine.OrtInferSession._get_ep_list = _get_ep_list  # type: ignore[method-assign]
    infer_engine.OrtInferSession._verify_providers = _verify_providers  # type: ignore[method-assign]


def _session_providers(engine: Any) -> list[str]:
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


def _build_engine(args: argparse.Namespace) -> Any:
    _prepend_ryzenai_path()
    _patch_rapidocr_for_vitisai(args.config_file)

    from rapidocr_onnxruntime import RapidOCR  # type: ignore

    kwargs: dict[str, Any] = {
        "use_cls": args.use_cls,
        "det_use_dml": False,
        "cls_use_dml": False,
        "rec_use_dml": False,
    }
    if args.det_model:
        kwargs["det_model_path"] = str(Path(args.det_model))
    if args.rec_model:
        kwargs["rec_model_path"] = str(Path(args.rec_model))
    if args.cls_model:
        kwargs["cls_model_path"] = str(Path(args.cls_model))
    return RapidOCR(**kwargs)


def _recognize(engine: Any, image: Path) -> str:
    result, _ = engine(str(image))
    if not result:
        return ""
    return " ".join(line[1] for line in result if line and len(line) > 1)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--det-model")
    parser.add_argument("--rec-model")
    parser.add_argument("--cls-model")
    parser.add_argument("--config-file")
    parser.add_argument("--use-cls", action="store_true")
    args = parser.parse_args()

    try:
        engine = _build_engine(args)
        out: dict[str, Any] = {
            "ok": True,
            "python": sys.executable,
            "providers": _session_providers(engine),
        }
        if args.probe:
            out["mode"] = "probe"
        else:
            if not args.image:
                raise ValueError("--image is required unless --probe is set")
            out["text"] = _recognize(engine, Path(args.image))
        print(json.dumps(out, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "python": sys.executable,
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
