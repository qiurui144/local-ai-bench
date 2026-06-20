"""
wraps llama.cpp llama-bench binary for direct PP/TG measurement.
无网络开销，直接测模型推理速度。

Usage（在 target 机器上）：
  python -c "from benchmark.native.llama_bench import run_llama_bench; print(run_llama_bench('/path/to/model.gguf'))"
"""
from __future__ import annotations
import json
import shutil
import subprocess
from typing import Optional


def run_llama_bench(
    model_path: str,
    pp: int = 512,
    tg: int = 128,
    n_gpu_layers: int = 99,
    llama_bench_bin: Optional[str] = None,
) -> dict:
    """返回 {'pp_tps': float, 'tg_tps': float, 'n_gpu_layers': int} 或 {'error': str}"""
    binary = llama_bench_bin or shutil.which("llama-bench") or shutil.which("llama-bench.exe")
    if not binary:
        return {"error": "llama-bench binary not found in PATH"}
    cmd = [
        binary,
        "-m", str(model_path),
        "-p", str(pp),
        "-n", str(tg),
        "-ngl", str(n_gpu_layers),
        "--output", "json",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=300)
        data = json.loads(out)
        if isinstance(data, list) and data:
            row = data[0]
            return {
                "pp_tps": float(row.get("pp_tps", 0.0)),
                "tg_tps": float(row.get("tg_tps", 0.0)),
                "n_gpu_layers": int(row.get("n_gpu_layers", 0)),
            }
        return {"error": f"unexpected output: {str(data)[:200]}"}
    except subprocess.TimeoutExpired:
        return {"error": "llama-bench timed out (>300s)"}
    except Exception as exc:
        return {"error": str(exc)}
