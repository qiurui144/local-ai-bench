"""Long AMD RyzenAI NPU probe using the official quicktest ONNX model."""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=int, default=90)
    args = parser.parse_args()

    root = Path(os.environ.get("RYZENAI_ROOT", r"C:\Program Files\RyzenAI\1.7.1"))
    driverstores = [
        Path(p)
        for p in glob.glob(r"C:\WINDOWS\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*")
    ]
    paths = driverstores + [
        root / "deployment",
        root / "onnxruntime" / "bin",
        root / "xrt",
        Path.home() / "amd-npu-rai161" / "npu_mcdm_stack_prod",
    ]
    for path in paths:
        if path.exists():
            os.add_dll_directory(str(path))
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
    os.environ["RYZEN_AI_INSTALLATION_PATH"] = str(root)

    model = root / "quicktest" / "test_model.onnx"
    xclbin = root / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin"
    provider_options = {
        "target": "X1",
        "xlnx_enable_py3_round": 0,
        "xclbin": str(xclbin),
    }
    session = ort.InferenceSession(
        str(model),
        providers=[("VitisAIExecutionProvider", provider_options), "CPUExecutionProvider"],
    )

    input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)
    end = time.time() + args.duration_s
    latencies: list[float] = []
    count = 0
    print(json.dumps({"event": "started", "providers": session.get_providers(), "duration_s": args.duration_s}), flush=True)
    while time.time() < end:
        started = time.perf_counter()
        session.run(None, {"input": input_data})
        latencies.append((time.perf_counter() - started) * 1000.0)
        count += 1
    latencies_sorted = sorted(latencies)
    print(
        json.dumps(
            {
                "event": "done",
                "count": count,
                "p50_ms": latencies_sorted[len(latencies_sorted) // 2],
                "min_ms": min(latencies),
                "max_ms": max(latencies),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
