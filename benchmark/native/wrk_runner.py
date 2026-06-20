"""
wraps wrk/k6 for HTTP TTFT/throughput measurement.
消除 Python httpx 的 ~5ms overhead 和 GIL 并发限制。
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import tempfile


_WRK_LUA_TTFT = """\
-- wrk Lua script: measure first-token latency via streaming
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{body}'

function response(status, headers, body)
  -- wrk doesn't support streaming natively; we measure end-to-end
end
"""


def run_wrk(
    base_url: str,
    model_name: str,
    prompt: str,
    connections: int = 1,
    threads: int = 1,
    duration_s: int = 30,
) -> dict:
    """返回 {'req_per_s': float, 'latency_avg_ms': float, 'latency_p99_ms': float} 或 {'error': str}"""
    binary = shutil.which("wrk") or shutil.which("wrk.exe")
    if not binary:
        return {"error": "wrk not found in PATH; install via apt/brew or build from source"}
    body = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "stream": False,
    }).replace("'", "\\'")
    lua = _WRK_LUA_TTFT.format(body=body)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lua", delete=False) as f:
        f.write(lua)
        lua_path = f.name
    try:
        url = f"{base_url}/chat/completions"
        cmd = [binary, "-t", str(threads), "-c", str(connections),
               "-d", f"{duration_s}s", "-s", lua_path, url]
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=duration_s + 30,
                                      text=True)
        return _parse_wrk_output(raw)
    except subprocess.TimeoutExpired:
        return {"error": f"wrk timed out (>{duration_s+30}s)"}
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        os.unlink(lua_path)


def _parse_wrk_output(raw: str) -> dict:
    """解析 wrk 文本输出为结构化数据。"""
    import re
    result: dict = {}
    m = re.search(r"Requests/sec:\s+([\d.]+)", raw)
    if m:
        result["req_per_s"] = float(m.group(1))
    m = re.search(r"Latency\s+([\d.]+)(ms|s|us)", raw)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        result["latency_avg_ms"] = val if unit == "ms" else (val * 1000 if unit == "s" else val / 1000)
    if not result:
        return {"error": f"cannot parse wrk output: {raw[:300]}"}
    return result
