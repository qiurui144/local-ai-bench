"""Intel non-LLM 3-seed verification — sequential runner (run on Intel Windows machine).

Run via run_intel_nonllm_verify.bat for reliable background execution.

Models run in sequence (no concurrency — iGPU must be exclusive for OV tests):
  1. qwen3-embedding-0.6b-intel-win  (Ollama CPU embedding, 3-seed)
  2. bge-reranker-base-intel-win     (CPU ONNX reranker, 3-seed)
  3. bge-reranker-v2-m3-intel-win    (CPU ONNX reranker stronger, 3-seed)
  4. rapidocr-intel-openvino         (iGPU OpenVINO OCR, 3-seed)
  5. sensevoice-small-intel-win      (DirectML ASR, 3-seed)

None of these need the OV extras server (port 8081).
The embedding model needs Ollama running on localhost:11434 — script starts it if missing.

Estimated total: ~60-90 min.
"""
import subprocess
import os
import time

PYTHON = r"C:\Users\happy\AppData\Local\Programs\Python\Python312\python.exe"
CWD    = r"C:\Users\happy\vlm-llm-benchmark"
OLLAMA = r"C:\Users\happy\AppData\Local\Programs\Ollama\ollama.exe"

env = dict(os.environ)
env["OLLAMA_INTEL_WIN_BASE_URL"] = "http://localhost:11434/v1"
env["PYTHONUNBUFFERED"] = "1"

MODELS = [
    ("qwen3-embedding-0.6b-intel-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,asr,ocr,rerank,prefill_decode,ttft,throughput"),
    ("bge-reranker-base-intel-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,asr,ocr,embedding,prefill_decode,ttft,throughput"),
    ("bge-reranker-v2-m3-intel-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,asr,ocr,embedding,prefill_decode,ttft,throughput"),
    ("rapidocr-intel-openvino",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,embedding,rerank,asr,prefill_decode,ttft,throughput"),
    ("sensevoice-small-intel-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,embedding,rerank,ocr,prefill_decode,ttft,throughput"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_ollama():
    """Start Ollama if not already running."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=3)
        log("Ollama already running on localhost:11434")
        return
    except Exception:
        pass
    log(f"Starting Ollama: {OLLAMA} serve")
    subprocess.Popen(
        [OLLAMA, "serve"],
        creationflags=0x00000008,  # DETACHED_PROCESS
        close_fds=True,
    )
    for attempt in range(12):
        time.sleep(3)
        try:
            urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
            log("Ollama started successfully")
            return
        except Exception:
            pass
    log("WARNING: Ollama may not have started; continuing anyway")


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


log(f"Intel non-LLM 3-seed verification — {len(MODELS)} models")
ensure_ollama()

for model_name, skip_dims in MODELS:
    rc = run_model(model_name, skip_dims)
    if rc not in (0, 1):
        log(f"WARNING: {model_name} exited rc={rc}, continuing anyway")

log("All models done. Check output/reports/ for results.")
