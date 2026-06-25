"""Launch qwen3-4b-igpu-intel-win GA re-test (3-seed) detached on Intel Windows machine."""
import subprocess, sys, os, time

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

env = dict(os.environ)
env["OV_INTEL_QWEN3_4B_BASE_URL"] = "http://localhost:8084/v1"
env["PYTHONUNBUFFERED"] = "1"

logfile = r"C:\Users\happy\bench_4b_igpu_ga.log"
cwd = r"C:\Users\happy\local-ai-bench"

cmd = [
    sys.executable, "run_benchmark.py",
    "--model", "qwen3-4b-igpu-intel-win",
    "--seeds", "3",
    "--skip", "stability,concurrency,conditioned,scenarios,conversation_drift,translation,asr,ocr,embedding,rerank,prefill_decode,ttft,throughput",
]

with open(logfile, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching qwen3-4b-igpu-intel-win GA 3-seed (harness fix 2026-06-23)\n")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=f, stderr=f,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )

print(f"PID: {proc.pid}  log: {logfile}")
