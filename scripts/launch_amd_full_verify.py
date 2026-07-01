"""AMD full 3-seed verification — sequential runner (run on AMD Windows machine).

Run via run_amd_verify.bat for reliable background execution.

Models run in sequence (no concurrency — GPU must be exclusive for perf tests):
  1. qwen3-embedding-0.6b-amd  (embedding, 3-seed)
  2. bge-m3-amd                (embedding, 3-seed)
  3. bge-reranker-base-amd-win (rerank CPU, 3-seed)
  4. bge-reranker-v2-m3-amd-win(rerank CPU, 3-seed)
  5. sensevoice-small-amd-win  (asr, 3-seed)
  6. rapidocr-amd-directml     (ocr DML, 3-seed)
  7. rapidocr-amd-npu          (ocr NPU, 3-seed)
  8. rapidocr-cpu              (ocr CPU, 3-seed)
  9. qwen3nt-4b-amd            (ttft + throughput + general_ability + translation, 3-seed)

Estimated total: ~90 min (non-LLM) + ~3 hours (qwen3nt GA+translation)
"""
import subprocess
import os
import time

PYTHON = r"C:\Users\happy\AppData\Local\Programs\Python\Python311\python.exe"
CWD    = r"C:\Users\happy\vlm-llm-benchmark"

env = dict(os.environ)
env["OLLAMA_AMD_BASE_URL"]     = "http://localhost:11434/v1"
env["ORT_AMD_EXTRAS_BASE_URL"] = "http://localhost:8091/v1"
env["PYTHONUNBUFFERED"]        = "1"

# (model_name, skip_dims)  — skip everything not relevant to each model type
MODELS = [
    # ── Non-LLM small models (fast, ~5-15 min each including 3 seeds) ──
    ("qwen3-embedding-0.6b-amd",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,asr,ocr,rerank,prefill_decode,ttft,throughput"),
    ("bge-m3-amd",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,asr,ocr,rerank,prefill_decode,ttft,throughput"),
    ("bge-reranker-base-amd-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,asr,ocr,embedding,prefill_decode,ttft,throughput"),
    ("bge-reranker-v2-m3-amd-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,asr,ocr,embedding,prefill_decode,ttft,throughput"),
    ("sensevoice-small-amd-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,embedding,rerank,ocr,prefill_decode,ttft,throughput"),
    ("rapidocr-amd-directml",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,embedding,rerank,asr,prefill_decode,ttft,throughput"),
    ("rapidocr-amd-npu",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,embedding,rerank,asr,prefill_decode,ttft,throughput"),
    ("rapidocr-cpu",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,embedding,rerank,asr,prefill_decode,ttft,throughput"),
    # ── LLM: qwen3nt-4b — TTFT + throughput + GA + translation, 3-seed ──
    ("qwen3nt-4b-amd",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "asr,ocr,embedding,rerank,prefill_decode"),
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


log(f"AMD full 3-seed verification start — {len(MODELS)} models")
for model_name, skip_dims in MODELS:
    rc = run_model(model_name, skip_dims)
    if rc not in (0, 1):
        log(f"WARNING: {model_name} exited rc={rc}, continuing anyway")

log("All models done. Check output/reports/ for results.")
