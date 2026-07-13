#!/usr/bin/env python3
"""Run a full Windows target model matrix locally on the target machine.

This script is intentionally target-local: deploy the repo to the Windows
machine first, then run it over SSH or as a detached background process. It
clears each model's quick-test skip list so all gated dimensions are attempted.

Runtime repair policy:
- Ollama: start the server if needed and pull missing model IDs.
- AMD ORT extras / Intel OV extras: start the local FastAPI service if needed.
- Intel OpenVINO LLM: clean old same-port supervisor/serve processes before
  starting the current model service.
- Other unavailable endpoints are recorded as repair failures for follow-up
  instead of being silently skipped.

Execution policy:
- The default entrypoint accepts one model only. Batch/all-model runs require
  --allow-batch because same-machine overlap or long auto-continuation can
  contaminate performance data.
- Ollama LLM/VLM runs are checked with ollama ps before benchmarking. CPU-only
  LLM/VLM is blocked unless --allow-cpu-llm-vlm is explicitly set.
- Scenarios run L1-only by default in this target-local runner. Auto-selecting
  a same-machine L2 judge would load a second model and violate the single-model
  policy.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_benchmark as rb  # noqa: E402
from common import ModelConfig, _is_chat_capable, load_benchmarks_config, load_models  # noqa: E402


REPORTS = ROOT / "output" / "reports"
RUNS = REPORTS / "windows-full-matrix"
DETACHED_PROCESS = 0x00000008 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0
START_NEW_SESSION = os.name != "nt"


TARGET_ENV = {
    "amd-win-x86": {
        "OLLAMA_AMD_BASE_URL": "http://localhost:11434/v1",
        "ORT_AMD_EXTRAS_BASE_URL": "http://localhost:8091/v1",
        "LEMONADE_AMD_BASE_URL": "http://localhost:8000/v1",
    },
    "intel-win-x86": {
        "OLLAMA_INTEL_WIN_BASE_URL": "http://localhost:11434/v1",
        "OV_INTEL_BASE_URL": "http://localhost:8080/v1",
        "OV_EXTRAS_INTEL_BASE_URL": "http://localhost:8081/v1",
        "OV_INTEL_QWEN3_06B_BASE_URL": "http://localhost:8082/v1",
        "OV_INTEL_QWEN3_17B_BASE_URL": "http://localhost:8083/v1",
        "OV_INTEL_QWEN3_4B_BASE_URL": "http://localhost:8084/v1",
        "OV_INTEL_QWEN25_7B_BASE_URL": "http://localhost:8085/v1",
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _json_get(url: str, timeout: float = 5.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _json_post(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any] | None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _ollama_exe() -> str:
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(local) / "Programs" / "Ollama" / "ollama.exe",
        shutil.which("ollama"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return "ollama"


def _start_process(
    cmd: list[str],
    log_prefix: Path,
    env: dict[str, str] | None = None,
    settle_s: float = 0.0,
) -> int:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    out = open(str(log_prefix) + ".out.log", "a", encoding="utf-8", errors="replace")
    err = open(str(log_prefix) + ".err.log", "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env or os.environ.copy(),
        stdout=out,
        stderr=err,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        start_new_session=START_NEW_SESSION,
    )
    if settle_s > 0:
        time.sleep(settle_s)
        if proc.poll() is not None:
            raise RuntimeError(f"process exited during startup: rc={proc.returncode}")
    return int(proc.pid)


def ensure_ollama(target: str, manifest: list[dict[str, Any]]) -> bool:
    if _http_ok("http://localhost:11434/api/version"):
        return True

    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", "0.0.0.0:11434")
    if target == "amd-win-x86":
        env.setdefault("OLLAMA_IGPU_ENABLE", "1")
    cmd = [_ollama_exe(), "serve"]
    try:
        pid = _start_process(cmd, RUNS / f"{target}-ollama-serve", env)
        manifest.append({"event": "repair", "runtime": "ollama", "action": "start", "pid": pid})
    except Exception as exc:
        manifest.append({"event": "repair_failed", "runtime": "ollama", "action": "start", "error": repr(exc)})
        return False

    for _ in range(40):
        time.sleep(3)
        if _http_ok("http://localhost:11434/api/version"):
            return True
    manifest.append({"event": "repair_failed", "runtime": "ollama", "action": "wait_ready"})
    return False


def _ollama_models() -> set[str]:
    data = _json_get("http://localhost:11434/api/tags", timeout=10)
    out: set[str] = set()
    for item in (data or {}).get("models", []):
        name = item.get("name")
        if name:
            out.add(name)
    return out


def ensure_ollama_model(model_id: str, manifest: list[dict[str, Any]]) -> bool:
    if model_id in _ollama_models():
        return True
    log(f"pulling missing Ollama model: {model_id}")
    manifest.append({"event": "repair", "runtime": "ollama", "action": "pull", "model_id": model_id})
    proc = subprocess.run([_ollama_exe(), "pull", model_id], text=True)
    if proc.returncode != 0:
        manifest.append({
            "event": "repair_failed",
            "runtime": "ollama",
            "action": "pull",
            "model_id": model_id,
            "returncode": proc.returncode,
        })
        return False
    return model_id in _ollama_models()


def _ollama_ps_text() -> str:
    try:
        proc = subprocess.run(
            [_ollama_exe(), "ps"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return f"ollama ps failed: {exc!r}"
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _ollama_ps_acceleration_state(ps_text: str, model_id: str) -> str:
    """Return gpu/cpu/unknown/missing for a loaded Ollama model."""
    needle = model_id.lower()
    for line in ps_text.splitlines():
        lower = line.lower()
        if not lower.strip() or lower.lstrip().startswith("name"):
            continue
        if needle not in lower:
            continue
        if "gpu" in lower:
            return "gpu"
        if "cpu" in lower:
            return "cpu"
        return "unknown"
    return "missing"


def _load_ollama_model_for_accel_check(model_id: str, manifest: list[dict[str, Any]]) -> bool:
    payload = {
        "model": model_id,
        "prompt": "ping",
        "stream": False,
        "options": {"num_predict": 1},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            ok = 200 <= getattr(resp, "status", 200) < 300
    except Exception as exc:
        manifest.append({
            "event": "acceleration_check_failed",
            "runtime": "ollama",
            "model_id": model_id,
            "error": repr(exc),
        })
        return False
    return ok


def _check_llm_vlm_acceleration(
    model: ModelConfig,
    target: str,
    manifest: list[dict[str, Any]],
    *,
    allow_cpu_llm_vlm: bool,
) -> bool:
    if model.provider != "ollama" or not _is_chat_capable(model):
        return True
    if allow_cpu_llm_vlm:
        manifest.append({
            "event": "cpu_llm_vlm_allowed",
            "model": model.name,
            "target": target,
            "model_id": model.effective_model_id,
        })
        return True

    model_id = model.effective_model_id
    if not _load_ollama_model_for_accel_check(model_id, manifest):
        return False
    ps_text = _ollama_ps_text()
    state = _ollama_ps_acceleration_state(ps_text, model_id)
    manifest.append({
        "event": "llm_vlm_acceleration_check",
        "model": model.name,
        "target": target,
        "model_id": model_id,
        "state": state,
        "ollama_ps": ps_text,
    })
    return state == "gpu"


def ensure_amd_extras(manifest: list[dict[str, Any]]) -> bool:
    if _http_ok("http://localhost:8091/v1/models"):
        return True
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "serve_ort_extras_amd.py"),
        "--host", "0.0.0.0",
        "--port", "8091",
    ]
    try:
        pid = _start_process(cmd, RUNS / "amd-ort-extras")
        manifest.append({"event": "repair", "runtime": "amd_ort_extras", "action": "start", "pid": pid})
    except Exception as exc:
        manifest.append({"event": "repair_failed", "runtime": "amd_ort_extras", "action": "start", "error": repr(exc)})
        return False
    for _ in range(40):
        time.sleep(3)
        if _http_ok("http://localhost:8091/v1/models"):
            return True
    manifest.append({"event": "repair_failed", "runtime": "amd_ort_extras", "action": "wait_ready"})
    return False


def ensure_intel_extras(manifest: list[dict[str, Any]]) -> bool:
    if _http_ok("http://localhost:8081/v1/models"):
        return True
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "serve_ov_extras.py"),
        "--host", "0.0.0.0",
        "--port", "8081",
        "--emb-device", "GPU",
        "--rank-device", "GPU",
        "--asr-device", "CPU",
    ]
    try:
        pid = _start_process(cmd, RUNS / "intel-ov-extras")
        manifest.append({"event": "repair", "runtime": "intel_ov_extras", "action": "start", "pid": pid})
    except Exception as exc:
        manifest.append({"event": "repair_failed", "runtime": "intel_ov_extras", "action": "start", "error": repr(exc)})
        return False
    for _ in range(40):
        time.sleep(3)
        if _http_ok("http://localhost:8081/v1/models"):
            return True
    manifest.append({"event": "repair_failed", "runtime": "intel_ov_extras", "action": "wait_ready"})
    return False


def _needs_intel_ov_llm(model: ModelConfig, target: str) -> bool:
    if target != "intel-win-x86":
        return False
    if model.provider != "openai" or not _is_chat_capable(model):
        return False
    env_name = model.base_url_env or ""
    return env_name.startswith("OV_INTEL") and env_name != "OV_EXTRAS_INTEL_BASE_URL"


def _intel_ov_llm_python() -> str:
    configured = os.environ.get("OV_INTEL_LLM_PYTHON", "").strip()
    if configured:
        return configured
    user_profile = os.environ.get("USERPROFILE", r"C:\Users\happy")
    default = Path(user_profile) / "ov-llm-venv" / "Scripts" / "python.exe"
    if default.exists():
        return str(default)
    return sys.executable


def _intel_ov_model_dir(model: ModelConfig) -> str:
    env_name = (model.base_url_env or "").strip()
    if env_name:
        override = os.environ.get(f"{env_name}_MODEL_DIR", "").strip()
        if override:
            return override
    root = os.environ.get("OV_INTEL_MODEL_ROOT", r"C:\ov_models").strip() or r"C:\ov_models"
    return str(Path(root) / model.effective_model_id)


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _intel_ov_llm_max_concurrent() -> int:
    return _env_int("OV_INTEL_LLM_MAX_CONCURRENT", 1, minimum=1)


def _intel_ov_llm_reload_every() -> int:
    return _env_int("OV_INTEL_LLM_RELOAD_EVERY", 0, minimum=0)


def _intel_ov_llm_exit_every() -> int:
    return _env_int("OV_INTEL_LLM_EXIT_EVERY", 0, minimum=0)


def _endpoint_port(model: ModelConfig) -> int:
    if model.port:
        return int(model.port)
    parsed = urllib.parse.urlparse(model.base_url)
    if parsed.port:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else 80


def _stop_windows_port(port: int, manifest: list[dict[str, Any]]) -> None:
    if os.name != "nt":
        return
    ps = (
        f"Get-NetTCPConnection -LocalPort {int(port)} -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty OwningProcess -Unique | "
        "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    manifest.append({
        "event": "repair",
        "runtime": "intel_ov_llm",
        "action": "stop_port",
        "port": port,
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "").strip()[-500:],
    })


def _stop_windows_ov_llm_processes(port: int, supervisor_name: str, manifest: list[dict[str, Any]]) -> None:
    """Stop stale Intel OV LLM supervisors and children before a clean model run."""
    if os.name != "nt":
        return
    patterns = {
        "supervisor_name": supervisor_name,
        "port": str(int(port)),
    }
    ps = f"""
$supervisorName = {json.dumps(patterns["supervisor_name"])};
$port = {json.dumps(patterns["port"])};
$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue;
$procs | Where-Object {{
    $_.CommandLine -and
    $_.CommandLine -like '*supervise_process.py*' -and
    $_.CommandLine -like ('*' + $supervisorName + '*')
}} | ForEach-Object {{
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}};
$procs | Where-Object {{
    $_.CommandLine -and
    $_.CommandLine -like '*serve_ov_intel.py*' -and
    $_.CommandLine -like '*--port*' -and
    $_.CommandLine -like ('*' + $port + '*')
}} | ForEach-Object {{
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}};
"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    manifest.append({
        "event": "repair",
        "runtime": "intel_ov_llm",
        "action": "stop_named_processes",
        "port": port,
        "supervisor_name": supervisor_name,
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "").strip()[-500:],
    })


def _probe_openai_chat(model: ModelConfig, timeout: float = 360.0) -> tuple[bool, dict[str, Any] | str]:
    payload = {
        "model": model.effective_model_id,
        "messages": [{"role": "user", "content": "Return exactly OK."}],
        "max_tokens": 1,
        "temperature": 0,
    }
    t0 = time.monotonic()
    data = _json_post(model.base_url.rstrip("/") + "/chat/completions", payload, timeout=timeout)
    if not data:
        return False, "empty_or_failed_response"
    content = (
        ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if isinstance(data, dict)
        else ""
    )
    return True, {
        "latency_s": round(time.monotonic() - t0, 3),
        "content": content,
        "usage": data.get("usage") if isinstance(data, dict) else None,
    }


def ensure_intel_ov_llm(model: ModelConfig, manifest: list[dict[str, Any]]) -> bool:
    ready_url = model.base_url.rstrip("/") + "/models"
    port = _endpoint_port(model)
    supervisor_name = f"intel-ov-llm-{port}-{model.name}"
    _stop_windows_ov_llm_processes(port, supervisor_name, manifest)
    _stop_windows_port(port, manifest)

    py = _intel_ov_llm_python()
    model_dir = _intel_ov_model_dir(model)
    device = os.environ.get("OV_INTEL_LLM_DEVICE", "GPU").strip() or "GPU"
    max_concurrent = _intel_ov_llm_max_concurrent()
    reload_every = _intel_ov_llm_reload_every()
    exit_every = _intel_ov_llm_exit_every()
    service_cmd = [
        py,
        "-u",
        str(ROOT / "scripts" / "serve_ov_intel.py"),
        "--llm",
        model_dir,
        "--llm-device",
        device,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--llm-max-concurrent",
        str(max_concurrent),
        "--llm-reload-every",
        str(reload_every),
        "--llm-exit-every",
        str(exit_every),
    ]
    cmd = [
        py,
        "-u",
        str(ROOT / "scripts" / "supervise_process.py"),
        "--name",
        supervisor_name,
        "--restart-delay",
        "3",
        "--",
        *service_cmd,
    ]
    try:
        pid = _start_process(cmd, RUNS / f"intel-ov-llm-{port}-{model.name}", settle_s=3.0)
        manifest.append({
            "event": "repair",
            "runtime": "intel_ov_llm",
            "action": "start",
            "model": model.name,
            "pid": pid,
            "python": py,
            "model_dir": model_dir,
            "device": device,
            "port": port,
            "supervisor_name": supervisor_name,
            "max_concurrent": max_concurrent,
            "reload_every": reload_every,
            "exit_every": exit_every,
            "supervisor": True,
        })
    except Exception as exc:
        manifest.append({
            "event": "repair_failed",
            "runtime": "intel_ov_llm",
            "action": "start",
            "model": model.name,
            "error": repr(exc),
        })
        return False

    for _ in range(60):
        time.sleep(2)
        if _http_ok(ready_url, timeout=3):
            ok, detail = _probe_openai_chat(model)
            manifest.append({
                "event": "runtime_probe",
                "runtime": "intel_ov_llm",
                "model": model.name,
                "ready_url": ready_url,
                "ok": ok,
                "detail": detail,
            })
            return ok
    manifest.append({
        "event": "repair_failed",
        "runtime": "intel_ov_llm",
        "action": "wait_ready",
        "model": model.name,
        "url": ready_url,
    })
    return False


def _needs_extras(model: ModelConfig, target: str) -> bool:
    if not model.base_url_env:
        return False
    if target == "amd-win-x86" and model.base_url_env == "ORT_AMD_EXTRAS_BASE_URL":
        return True
    if target == "intel-win-x86" and model.base_url_env == "OV_EXTRAS_INTEL_BASE_URL":
        return True
    return False


def repair_model_runtime(model: ModelConfig, target: str, manifest: list[dict[str, Any]]) -> bool:
    if model.provider == "ollama":
        if not ensure_ollama(target, manifest):
            return False
        return ensure_ollama_model(model.effective_model_id, manifest)
    if _needs_extras(model, target):
        return ensure_amd_extras(manifest) if target == "amd-win-x86" else ensure_intel_extras(manifest)
    if _needs_intel_ov_llm(model, target):
        return ensure_intel_ov_llm(model, manifest)
    if model.port and model.base_url:
        ready_url = model.base_url.rstrip("/") + "/models"
        if _http_ok(ready_url, timeout=5):
            return True
        manifest.append({
            "event": "repair_failed",
            "model": model.name,
            "runtime": model.provider,
            "action": "endpoint_not_ready",
            "url": ready_url,
        })
        return False
    return True


def _selected_models(target: str, names: str) -> list[ModelConfig]:
    models = [
        m for m in load_models(ROOT / "models.yaml")
        if (getattr(m, "target", None) or "local") == target
    ]
    if names != "all":
        requested = [x.strip() for x in names.split(",") if x.strip()]
        by_name = {m.name: m for m in models}
        missing = set(requested) - set(by_name)
        if missing:
            raise SystemExit(f"unknown model(s) for {target}: {sorted(missing)}")
        models = [by_name[name] for name in requested]
    return models


def _is_batch_selection(models_arg: str) -> bool:
    names = [x.strip() for x in models_arg.split(",") if x.strip()]
    return models_arg.strip() == "all" or len(names) > 1


def _validate_selection_policy(args: argparse.Namespace) -> None:
    if args.child_model:
        return
    if not args.models.strip():
        raise SystemExit("--models must name exactly one model; use --allow-batch for comma lists or all")
    if _is_batch_selection(args.models) and not args.allow_batch:
        raise SystemExit(
            "batch model selection is disabled by default; run one model per target or pass --allow-batch"
        )


def _apply_single_model_benchmark_policy(
    cfg: ModelConfig,
    manifest: list[dict[str, Any]],
    *,
    allow_local_scenarios_judge: bool,
) -> None:
    if allow_local_scenarios_judge or not _is_chat_capable(cfg):
        return
    scenarios_cfg = cfg.benchmarks.setdefault("scenarios", {})
    if scenarios_cfg.get("judge_model") != cfg.name:
        scenarios_cfg["judge_model"] = cfg.name
        manifest.append({
            "event": "single_model_scenarios_l1_only",
            "model": cfg.name,
            "reason": "same-machine L2 judge would load a second model",
        })


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=rb._default), encoding="utf-8")


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-max_chars:]


def _detached_child_cmd(args: argparse.Namespace, tag: str) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--target",
        args.target,
        "--models",
        args.models,
        "--seeds",
        str(args.seeds),
        "--tag",
        tag,
    ]
    if args.start_index:
        cmd += ["--start-index", str(args.start_index)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.allow_batch:
        cmd.append("--allow-batch")
    if args.allow_cpu_llm_vlm:
        cmd.append("--allow-cpu-llm-vlm")
    if args.allow_local_scenarios_judge:
        cmd.append("--allow-local-scenarios-judge")
    if getattr(args, "preserve_model_skip", False):
        cmd.append("--preserve-model-skip")
    return cmd


def _windows_detach_cmd_lines(args: argparse.Namespace, tag: str) -> tuple[Path, list[str]]:
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd_path = RUNS / f"{tag}.cmd"
    out_path = RUNS / f"{tag}.cmd.out.log"
    err_path = RUNS / f"{tag}.cmd.err.log"
    run_line = (
        subprocess.list2cmdline(_detached_child_cmd(args, tag))
        + " > "
        + subprocess.list2cmdline([str(out_path)])
        + " 2> "
        + subprocess.list2cmdline([str(err_path)])
    )
    lines = [
        "@echo off",
        "cd /d " + subprocess.list2cmdline([str(ROOT)]),
        "set PYTHONUTF8=1",
        "set PYTHONUNBUFFERED=1",
        run_line,
    ]
    return cmd_path, lines


def _spawn_detached_windows_task(args: argparse.Namespace, tag: str) -> int:
    cmd_path, lines = _windows_detach_cmd_lines(args, tag)
    cmd_path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    task_name = f"vlm-{tag}"
    subprocess.run(
        ["schtasks", "/Delete", "/TN", task_name, "/F"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    start_time = (dt.datetime.now() + dt.timedelta(minutes=10)).strftime("%H:%M")
    task_run = f'cmd.exe /c "{cmd_path}"'
    subprocess.run(
        [
            "schtasks", "/Create",
            "/TN", task_name,
            "/TR", task_run,
            "/SC", "ONCE",
            "/ST", start_time,
            "/F",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    subprocess.run(
        ["schtasks", "/Run", "/TN", task_name],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _write_json(RUNS / f"{tag}_launcher.json", {
        "event": "scheduled_task_start",
        "task_name": task_name,
        "tag": tag,
        "target": args.target,
        "models": args.models,
        "seeds": args.seeds,
        "start_index": args.start_index,
        "limit": args.limit,
        "allow_batch": args.allow_batch,
        "allow_cpu_llm_vlm": args.allow_cpu_llm_vlm,
        "allow_local_scenarios_judge": args.allow_local_scenarios_judge,
        "preserve_model_skip": getattr(args, "preserve_model_skip", False),
        "cmd_file": str(cmd_path),
        "timestamp": dt.datetime.now().isoformat(),
    })
    return 0


def _spawn_detached(args: argparse.Namespace, tag: str) -> int:
    if os.name == "nt":
        return _spawn_detached_windows_task(args, tag)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = _detached_child_cmd(args, tag)
    pid = _start_process(cmd, RUNS / tag, env, settle_s=5.0)
    _write_json(RUNS / f"{tag}_launcher.json", {
        "event": "detached_start",
        "pid": pid,
        "tag": tag,
        "target": args.target,
        "models": args.models,
        "seeds": args.seeds,
        "start_index": args.start_index,
        "limit": args.limit,
        "allow_batch": args.allow_batch,
        "allow_cpu_llm_vlm": args.allow_cpu_llm_vlm,
        "allow_local_scenarios_judge": args.allow_local_scenarios_judge,
        "preserve_model_skip": getattr(args, "preserve_model_skip", False),
        "cmd": cmd,
        "timestamp": dt.datetime.now().isoformat(),
    })
    return pid


def _run_child_model(args: argparse.Namespace, tag: str) -> int:
    if not args.child_output:
        raise SystemExit("--child-output is required with --child-model")
    selected = _selected_models(args.target, args.child_model)
    if len(selected) != 1:
        raise SystemExit(f"--child-model must select exactly one model, got {len(selected)}")

    golden = json.loads((ROOT / "golden" / "expectations.json").read_text(encoding="utf-8"))
    bench_cfg = load_benchmarks_config(ROOT / "models.yaml")
    manifest: list[dict[str, Any]] = []
    row = _run_one_model(
        selected[0],
        bench_cfg,
        golden,
        args.seeds,
        tag,
        manifest,
        allow_local_scenarios_judge=args.allow_local_scenarios_judge,
        preserve_model_skip=getattr(args, "preserve_model_skip", False),
    )
    if manifest:
        row["child_manifest"] = manifest
    _write_json(Path(args.child_output), row)
    return 0


def _run_one_model_isolated(
    model: ModelConfig,
    target: str,
    seeds: int,
    tag: str,
    manifest: list[dict[str, Any]],
    *,
    allow_local_scenarios_judge: bool = False,
    preserve_model_skip: bool = False,
) -> dict[str, Any]:
    stem = f"{tag}_{model.name}"
    row_path = RUNS / f"{stem}_row.json"
    out_path = RUNS / f"{stem}.out.log"
    err_path = RUNS / f"{stem}.err.log"
    row_path.unlink(missing_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--target",
        target,
        "--models",
        model.name,
        "--seeds",
        str(seeds),
        "--tag",
        tag,
        "--child-model",
        model.name,
        "--child-output",
        str(row_path),
    ]
    if allow_local_scenarios_judge:
        cmd.append("--allow-local-scenarios-judge")
    if preserve_model_skip:
        cmd.append("--preserve-model-skip")
    log(f"CHILD {model.name} start")
    with open(out_path, "a", encoding="utf-8", errors="replace") as out, open(
        err_path, "a", encoding="utf-8", errors="replace"
    ) as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy(), stdout=out, stderr=err)
    if proc.returncode == 0 and row_path.exists():
        row = json.loads(row_path.read_text(encoding="utf-8"))
        row["child_logs"] = {"stdout": str(out_path), "stderr": str(err_path)}
        log(f"CHILD {model.name} done")
        return row

    row = {
        "model": model.name,
        "error": "model_process_failed",
        "returncode": proc.returncode,
        "provider": model.provider,
        "benchmarks": {},
        "child_logs": {"stdout": str(out_path), "stderr": str(err_path)},
        "stderr_tail": _tail_text(err_path),
    }
    manifest.append({"event": "model_process_failed", **row})
    log(f"CHILD {model.name} failed rc={proc.returncode}")
    return row


def _run_one_model(
    model: ModelConfig,
    bench_cfg: dict,
    golden: dict,
    seeds: int,
    tag: str,
    manifest: list[dict[str, Any]],
    *,
    allow_local_scenarios_judge: bool = False,
    preserve_model_skip: bool = False,
) -> dict:
    seed_runs: list[dict] = []
    durations: list[float] = []
    for idx in range(seeds):
        cfg = copy.deepcopy(model)
        cfg.benchmarks = copy.deepcopy(cfg.benchmarks or {})
        if not preserve_model_skip:
            cfg.benchmarks["skip"] = []
        _apply_single_model_benchmark_policy(
            cfg,
            manifest,
            allow_local_scenarios_judge=allow_local_scenarios_judge,
        )
        log(f"START {cfg.name} seed {idx + 1}/{seeds}")
        t0 = time.monotonic()
        try:
            seed_runs.append(rb.run_all_for_model(cfg, golden, set(), bench_cfg))
        except Exception as exc:
            seed_runs.append({
                "model": cfg.name,
                "timestamp": dt.datetime.now().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "benchmarks": {},
            })
        durations.append(time.monotonic() - t0)
        log(f"DONE  {cfg.name} seed {idx + 1}/{seeds} duration={durations[-1]:.1f}s")

    result = seed_runs[0]
    result["full_matrix"] = {
        "tag": tag,
        "seeds": seeds,
        "quick_skip_cleared": not preserve_model_skip,
        "model_skip_preserved": preserve_model_skip,
        "duration_s": round(sum(durations), 3),
    }
    if seeds > 1:
        for idx, seed_result in enumerate(seed_runs):
            _write_json(REPORTS / f"{model.name}_{tag}_seed{idx}.json", seed_result)
        result["multi_seed"] = rb.aggregate_multi_seed(seed_runs, durations)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{model.name}_{tag}_{stamp}"
    _write_json(REPORTS / f"{stem}.json", result)
    (REPORTS / f"{stem}.md").write_text(rb.render_markdown(result), encoding="utf-8")
    try:
        from benchmark.report.html_report import generate_html

        (REPORTS / f"{stem}.html").write_text(generate_html(result), encoding="utf-8")
    except Exception as exc:
        manifest.append({"event": "html_report_failed", "model": model.name, "error": repr(exc)})
    return {
        "model": model.name,
        "report": str(REPORTS / f"{stem}.json"),
        "duration_s": round(sum(durations), 3),
        "error": result.get("error"),
        "benchmarks": {
            name: (block.get("verdict", "MEASURED") if isinstance(block, dict) else "MEASURED")
            for name, block in (result.get("benchmarks") or {}).items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(TARGET_ENV))
    parser.add_argument("--models", default="", help="single model name; all/comma lists require --allow-batch")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--detach", action="store_true", help="start the run in a detached background process")
    parser.add_argument("--allow-batch", action="store_true", help="allow all/comma model selections")
    parser.add_argument(
        "--allow-cpu-llm-vlm",
        action="store_true",
        help="explicitly allow CPU-only Ollama LLM/VLM baseline runs",
    )
    parser.add_argument(
        "--allow-local-scenarios-judge",
        action="store_true",
        help="allow scenarios to auto-load a second same-machine L2 judge model",
    )
    parser.add_argument(
        "--preserve-model-skip",
        action="store_true",
        help="preserve each model's configured skip list; use for contract baseline runs that must not add extra dimensions",
    )
    parser.add_argument("--child-model", default="", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.seeds < 1:
        raise SystemExit("--seeds must be >= 1")

    for key, value in TARGET_ENV[args.target].items():
        os.environ[key] = value
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    tag = args.tag or f"{args.target}-full-{dt.datetime.now().strftime('%Y%m%d')}"
    if args.child_model:
        return _run_child_model(args, tag)
    _validate_selection_policy(args)

    if args.detach:
        pid = _spawn_detached(args, tag)
        log(f"DETACHED pid={pid} tag={tag} log={RUNS / (tag + '.out.log')}")
        return 0

    manifest: list[dict[str, Any]] = [{
        "event": "start",
        "target": args.target,
        "tag": tag,
        "seeds": args.seeds,
        "timestamp": dt.datetime.now().isoformat(),
    }]

    models = _selected_models(args.target, args.models)
    if args.start_index:
        models = models[args.start_index:]
    if args.limit:
        models = models[:args.limit]
    manifest.append({"event": "selected_models", "models": [m.name for m in models]})

    results: list[dict[str, Any]] = []

    for model in models:
        log(f"PREP {model.name} provider={model.provider} endpoint={model.base_url or 'local'}")
        if not repair_model_runtime(model, args.target, manifest):
            row = {
                "model": model.name,
                "error": "runtime_repair_failed",
                "provider": model.provider,
                "endpoint": model.base_url,
            }
            results.append(row)
            manifest.append({"event": "model_runtime_unavailable", **row})
            _write_json(RUNS / f"{tag}_summary.json", {"manifest": manifest, "results": results})
            continue
        if not _check_llm_vlm_acceleration(
            model,
            args.target,
            manifest,
            allow_cpu_llm_vlm=args.allow_cpu_llm_vlm,
        ):
            row = {
                "model": model.name,
                "error": "cpu_only_llm_vlm_blocked",
                "provider": model.provider,
                "endpoint": model.base_url,
                "policy": "LLM/VLM CPU-only tests require --allow-cpu-llm-vlm",
            }
            results.append(row)
            manifest.append({"event": "model_policy_blocked", **row})
            _write_json(RUNS / f"{tag}_summary.json", {"manifest": manifest, "results": results})
            continue
        row = _run_one_model_isolated(
            model,
            args.target,
            args.seeds,
            tag,
            manifest,
            allow_local_scenarios_judge=args.allow_local_scenarios_judge,
            preserve_model_skip=getattr(args, "preserve_model_skip", False),
        )
        results.append(row)
        _write_json(RUNS / f"{tag}_summary.json", {"manifest": manifest, "results": results})

    manifest.append({"event": "end", "timestamp": dt.datetime.now().isoformat()})
    _write_json(RUNS / f"{tag}_summary.json", {"manifest": manifest, "results": results})
    log(f"SUMMARY {RUNS / f'{tag}_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
