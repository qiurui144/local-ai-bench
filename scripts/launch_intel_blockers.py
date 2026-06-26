"""Intel blocker resolution — sequential runner (run on Intel Windows machine).

Resolves:
  1. rapidocr-intel-openvino  3-seed OCR (rapidocr-openvino 1.3.25 installed, OV openvino backend)
  2. qwen3-0.6b-intel-win     3-seed GA+Translation (port 8082 running; confirm PASS/FAIL)

Run via: wmic process call create "cmd /c C:\\Users\\happy\\run_intel_blockers.bat"
Log:     C:\\Users\\happy\\intel_blockers.log
"""
import subprocess, os, time

PYTHON = r"C:\Users\happy\AppData\Local\Programs\Python\Python312\python.exe"
CWD    = r"C:\Users\happy\vlm-llm-benchmark"
OLLAMA = r"C:\Users\happy\AppData\Local\Programs\Ollama\ollama.exe"

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"

# rapidocr: run OCR only; skip all other dims
# qwen3-0.6b: GA + translation only; skip non-LLM dims + heavy dims
MODELS = [
    ("rapidocr-intel-openvino",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "general_ability,translation,embedding,rerank,asr,prefill_decode,ttft,throughput"),
    ("qwen3-0.6b-intel-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,"
     "embedding,rerank,asr,ocr,prefill_decode,ttft,throughput"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_ollama():
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
    for _ in range(12):
        time.sleep(3)
        try:
            urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
            log("Ollama started")
            return
        except Exception:
            pass
    log("WARNING: Ollama may not have started; continuing anyway")


def run_model(model_name, skip_dims):
    cmd = [PYTHON, "run_benchmark.py", "--model", model_name, "--seeds", "3", "--skip", skip_dims]
    log(f"START {model_name}")
    proc = subprocess.Popen(cmd, cwd=CWD, env=env)
    proc.wait()
    log(f"DONE  {model_name}  rc={proc.returncode}")
    return proc.returncode


log(f"Intel blocker resolution — {len(MODELS)} models")
ensure_ollama()

for model_name, skip_dims in MODELS:
    rc = run_model(model_name, skip_dims)
    if rc not in (0, 1):
        log(f"WARNING: {model_name} exited rc={rc}")

log("All done. Check output/reports/ for results.")
