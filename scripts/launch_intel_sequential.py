"""Intel sequential 3-seed benchmark: qwen2.5-7b-int4-ov then qwen3-4b-int4-ov.

Both OpenVINO LLM servers already running:
  Port 8085: qwen2.5-7b-int4-ov  (OV_INTEL_QWEN25_7B_BASE_URL)
  Port 8084: qwen3-4b-int4-ov    (OV_INTEL_QWEN3_4B_BASE_URL)

Skip: ttft (100% error on OpenVINO streaming), stability, concurrency,
      prefill_decode, embedding, rerank, asr, ocr, conditioned, scenarios
Runs: general_ability + translation + throughput — 3 seeds each.
"""
import subprocess
import os
import time

PYTHON = r"C:\Users\happy\AppData\Local\Programs\Python\Python312\python.exe"
CWD    = r"C:\Users\happy\vlm-llm-benchmark"

env = dict(os.environ)
env["OV_INTEL_QWEN25_7B_BASE_URL"] = "http://localhost:8085/v1"
env["OV_INTEL_QWEN3_4B_BASE_URL"]  = "http://localhost:8084/v1"
env["OV_INTEL_EXTRAS_BASE_URL"]    = "http://localhost:8081/v1"
env["PYTHONUNBUFFERED"]            = "1"

SKIP = ("ttft,stability,concurrency,prefill_decode,"
        "embedding,rerank,asr,ocr,conditioned,scenarios,conversation_drift")

MODELS = [
    ("qwen2.5-7b-igpu-intel-win", SKIP),
    ("qwen3-4b-igpu-intel-win",   SKIP),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_model(model_name, skip_dims):
    cmd = [PYTHON, "run_benchmark.py",
           "--model", model_name,
           "--seeds", "3",
           "--skip", skip_dims]
    log(f"START {model_name}")
    log(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=CWD, env=env)
    proc.wait()
    log(f"DONE  {model_name}  rc={proc.returncode}")
    return proc.returncode


log(f"Intel sequential 3-seed benchmark — {len(MODELS)} models")
for model_name, skip_dims in MODELS:
    rc = run_model(model_name, skip_dims)
    if rc not in (0, 1):
        log(f"WARNING: {model_name} exited rc={rc}, continuing")

log("All done. Check output/reports/ for results.")
