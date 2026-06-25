"""Targeted re-run: qwen3nt-4b-amd only, 3 seeds.

Harness fix applied 2026-06-25 (commit 1c5c656):
- Ollama empty content retry when reasoning field present
- Unclosed <think> tag stripping

Expected improvement over prior run:
- translation empty_rate should drop from 1.000 to near-0
- GA scores should reflect actual model capability (not empty-content failure)
"""
import subprocess, os, time

PYTHON = r"C:\Users\happy\AppData\Local\Programs\Python\Python311\python.exe"
CWD    = r"C:\Users\happy\vlm-llm-benchmark"

env = dict(os.environ)
env["OLLAMA_AMD_BASE_URL"]    = "http://localhost:11434/v1"
env["ORT_AMD_EXTRAS_BASE_URL"] = "http://localhost:8091/v1"
env["PYTHONUNBUFFERED"]        = "1"

SKIP = ("ttft,stability,concurrency,prefill_decode,"
        "embedding,rerank,asr,ocr,conditioned,scenarios,conversation_drift")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("AMD qwen3nt-4b targeted re-run (harness fix 1c5c656)")
cmd = [PYTHON, "run_benchmark.py",
       "--model", "qwen3nt-4b-amd",
       "--seeds", "3",
       "--skip", SKIP]
log(f"cmd: {' '.join(cmd)}")
proc = subprocess.Popen(cmd, cwd=CWD, env=env)
proc.wait()
log(f"DONE  qwen3nt-4b-amd  rc={proc.returncode}")
log("Check output/reports/ for results.")
