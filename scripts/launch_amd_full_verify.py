"""AMD full 3-seed verification — run on AMD Windows machine (not from this host).

Models run in sequence (no concurrency per AMD benchmarking rules):
  1. qwen3-embedding-0.6b-amd  (embedding, 3-seed)
  2. bge-m3-amd                (embedding, 3-seed)
  3. bge-reranker-base-amd-win (rerank, 3-seed)
  4. bge-reranker-v2-m3-amd-win(rerank, 3-seed)
  5. sensevoice-small-amd-win  (asr, 3-seed)
  6. rapidocr-amd-directml     (ocr, 3-seed)
  7. rapidocr-amd-npu          (ocr, 3-seed)
  8. rapidocr-cpu              (ocr, 3-seed)
  9. qwen3nt-4b-amd            (general_ability + translation, 3-seed)

Usage: python scripts/launch_amd_full_verify.py
Estimated total: ~90 min (non-LLM) + ~3 hours (qwen3nt GA+translation)
"""
import subprocess, sys, os, time

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

cwd = r"C:\Users\happy\vlm-llm-benchmark"
logfile = r"C:\Users\happy\amd_full_verify.log"

env = dict(os.environ)
env["OLLAMA_AMD_BASE_URL"] = "http://localhost:11434/v1"
env["ORT_AMD_EXTRAS_BASE_URL"] = "http://localhost:8091/v1"
env["PYTHONUNBUFFERED"] = "1"

# Models to run in sequence: (model_name, skip_dims)
MODELS = [
    # Non-LLM small models — fast, 3-seed each
    ("qwen3-embedding-0.6b-amd",
     "stability,concurrency,conditioned,scenarios,conversation_drift,general_ability,translation,asr,ocr,rerank,prefill_decode,ttft,throughput"),
    ("bge-m3-amd",
     "stability,concurrency,conditioned,scenarios,conversation_drift,general_ability,translation,asr,ocr,rerank,prefill_decode,ttft,throughput"),
    ("bge-reranker-base-amd-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,general_ability,translation,asr,ocr,embedding,prefill_decode,ttft,throughput"),
    ("bge-reranker-v2-m3-amd-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,general_ability,translation,asr,ocr,embedding,prefill_decode,ttft,throughput"),
    ("sensevoice-small-amd-win",
     "stability,concurrency,conditioned,scenarios,conversation_drift,general_ability,translation,embedding,rerank,ocr,prefill_decode,ttft,throughput"),
    ("rapidocr-amd-directml",
     "stability,concurrency,conditioned,scenarios,conversation_drift,general_ability,translation,embedding,rerank,asr,prefill_decode,ttft,throughput"),
    ("rapidocr-amd-npu",
     "stability,concurrency,conditioned,scenarios,conversation_drift,general_ability,translation,embedding,rerank,asr,prefill_decode,ttft,throughput"),
    ("rapidocr-cpu",
     "stability,concurrency,conditioned,scenarios,conversation_drift,general_ability,translation,embedding,rerank,asr,prefill_decode,ttft,throughput"),
    # LLM: qwen3nt-4b — GA + translation, 3-seed
    ("qwen3nt-4b-amd",
     "stability,concurrency,conditioned,scenarios,conversation_drift,asr,ocr,embedding,rerank,prefill_decode"),
]

def run_model(model_name, skip_dims):
    cmd = [
        sys.executable, "run_benchmark.py",
        "--model", model_name,
        "--seeds", "3",
        "--skip", skip_dims,
    ]
    ts = time.strftime("%H:%M:%S")
    with open(logfile, "a") as f:
        f.write(f"\n[{ts}] START {model_name}\n")
        f.write(f"  cmd: {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=f, stderr=f,
        )
        proc.wait()
        f.write(f"[{time.strftime('%H:%M:%S')}] DONE  {model_name}  rc={proc.returncode}\n")
    return proc.returncode

with open(logfile, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] AMD full 3-seed verification start\n")
    f.write(f"  models: {[m[0] for m in MODELS]}\n")

print(f"[{time.strftime('%H:%M:%S')}] Launching AMD full verification ({len(MODELS)} models, 3-seed each)")
print(f"  log: {logfile}")

for model_name, skip_dims in MODELS:
    print(f"[{time.strftime('%H:%M:%S')}] Running {model_name}...")
    rc = run_model(model_name, skip_dims)
    print(f"[{time.strftime('%H:%M:%S')}] {model_name} done (rc={rc})")

print(f"[{time.strftime('%H:%M:%S')}] All done. Check {logfile} and output/reports/")
