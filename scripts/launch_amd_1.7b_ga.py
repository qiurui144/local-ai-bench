"""Launch qwen3-1.7b-amd GA re-test (3-seed) detached on AMD Windows machine."""
import subprocess, sys, os, time

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

env = dict(os.environ)
env["OLLAMA_AMD_BASE_URL"] = "http://localhost:11434/v1"
env["PYTHONUNBUFFERED"] = "1"

logfile = r"C:\Users\happy\bench_1.7b_ga_fixed.log"
cwd = r"C:\Users\happy\local-ai-bench"

cmd = [
    sys.executable, "run_benchmark.py",
    "--model", "qwen3-1.7b-amd",
    "--seeds", "3",
    "--skip", "stability,concurrency,conditioned,scenarios,conversation_drift,translation,asr,ocr,embedding,rerank,prefill_decode,ttft,throughput",
]

with open(logfile, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching qwen3-1.7b-amd GA 3-seed (harness fix 2026-06-23)\n")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=f, stderr=f,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )

print(f"PID: {proc.pid}  log: {logfile}")
