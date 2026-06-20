"""Long OpenVINO NPU probe using a tiny static-shape model."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import openvino as ov


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=int, default=90)
    args = parser.parse_args()

    core = ov.Core()
    opset = ov.opset8
    data = opset.parameter([1, 16], np.float32, name="data")
    relu = opset.relu(data)
    model = ov.Model([relu], [data], "tiny_relu_npu_long")
    started = time.perf_counter()
    compiled = core.compile_model(model, "NPU")
    compile_ms = (time.perf_counter() - started) * 1000.0
    input_data = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(1, 16)

    latencies: list[float] = []
    count = 0
    end = time.time() + args.duration_s
    print(json.dumps({"event": "started", "device": "NPU", "compile_ms": compile_ms, "duration_s": args.duration_s}), flush=True)
    while time.time() < end:
        started = time.perf_counter()
        result = compiled([input_data])[compiled.output(0)]
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
                "output_sum": float(np.sum(result)),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
