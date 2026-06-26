"""AMD qwen2.5-7b-amd-win formal 3-seed benchmark.

Runs: accuracy, ttft, throughput, prefill_decode, general_ability, translation
(skip list in models.yaml handles the rest)

Log: C:\\Users\\happy\\amd_7b_3seed.log
"""
import subprocess, os, time, urllib.request

PYTHON = r"C:\Users\happy\AppData\Local\Programs\Python\Python312\python.exe"
CWD = r"C:\Users\happy\vlm-llm-benchmark"
OLLAMA = r"C:\Users\happy\AppData\Local\Programs\Ollama\ollama.exe"

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_ollama():
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=3)
        log("Ollama already running")
        return
    except Exception:
        pass
    log("Starting Ollama...")
    subprocess.Popen([OLLAMA, "serve"], creationflags=0x00000008, close_fds=True)
    for _ in range(15):
        time.sleep(3)
        try:
            urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
            log("Ollama started")
            return
        except Exception:
            pass
    log("WARNING: Ollama may not have started; continuing")


log("AMD qwen2.5-7b-amd-win formal 3-seed")
ensure_ollama()

cmd = [PYTHON, "run_benchmark.py", "--model", "qwen2.5-7b-amd-win", "--seeds", "3"]
log(f"START: {' '.join(cmd)}")
proc = subprocess.Popen(cmd, cwd=CWD, env=env)
proc.wait()
log(f"DONE  rc={proc.returncode}")
