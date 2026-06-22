"""
Intel NPU validation model download script (hf-mirror)

Downloads OpenVINO official INT8 models for embedding/reranker/asr/ocr to C:\\ov_models\\
Usage: python npu_download_intel.py --task all
       python npu_download_intel.py --task embedding,asr
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Fix GBK console encoding on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MIRROR = "https://hf-mirror.com"
CURL = r"C:\Windows\System32\curl.exe"
BASE_DIR = Path(r"C:\ov_models")

# Confirmed available on OpenVINO HF hub (verified 2026-06-22)
MODELS = {
    "embedding": {
        "repo": "OpenVINO/bge-base-en-v1.5-int8-ov",
        "dest": BASE_DIR / "embedding" / "bge-base-en-v1.5-int8-ov",
        "desc": "BGE-base-en-v1.5 INT8 for NPU embedding (English)",
    },
    "reranker": {
        "repo": "OpenVINO/bge-reranker-base-int8-ov",
        "dest": BASE_DIR / "reranker" / "bge-reranker-base-int8-ov",
        "desc": "BGE-reranker-base INT8 for NPU reranking",
    },
    "asr": {
        "repo": "OpenVINO/whisper-base-int8-ov",
        "dest": BASE_DIR / "asr" / "whisper-base-int8-ov",
        "desc": "Whisper-base INT8 for NPU ASR",
    },
    "asr_large": {
        "repo": "OpenVINO/whisper-large-v3-int8-ov",
        "dest": BASE_DIR / "asr" / "whisper-large-v3-int8-ov",
        "desc": "Whisper-large-v3 INT8 for NPU ASR (high accuracy, ~3GB)",
    },
}


def curl_get(url: str, out: str | None = None) -> tuple[int, str, str]:
    cmd = [CURL, "-L", "--retry", "3", "--retry-delay", "5", "-f"]
    if out:
        cmd += ["-o", out]
    else:
        cmd += ["-s"]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def download_model(task_key: str, model_info: dict) -> dict:
    repo = model_info["repo"]
    dest = model_info["dest"]
    print(f"\n{'─'*50}", flush=True)
    print(f"Downloading: {repo} → {dest}", flush=True)

    # Get file list from HF API
    api_url = f"{MIRROR}/api/models/{repo}?blobs=true"
    rc, out, err = curl_get(api_url)
    if rc != 0:
        # Try without blobs param
        rc, out, err = curl_get(f"{MIRROR}/api/models/{repo}")
        if rc != 0:
            return {"status": "FAIL", "repo": repo, "error": f"API error: {err[:200]}"}

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {"status": "FAIL", "repo": repo, "error": "Invalid JSON from API"}

    siblings = data.get("siblings", [])
    if not siblings:
        return {"status": "FAIL", "repo": repo, "error": "No files found in repo"}

    print(f"Found {len(siblings)} files", flush=True)
    dest.mkdir(parents=True, exist_ok=True)

    failed = []
    for s in siblings:
        fname = s["rfilename"]
        out_path = dest / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  SKIP (exists): {fname}", flush=True)
            continue

        url = f"{MIRROR}/{repo}/resolve/main/{fname}"
        print(f"  Downloading: {fname}", flush=True)
        rc, _, err2 = curl_get(url, str(out_path))
        if rc != 0:
            print(f"  FAIL: {fname} rc={rc}", flush=True)
            failed.append(fname)
        else:
            sz = out_path.stat().st_size if out_path.exists() else 0
            print(f"  OK: {fname} ({sz/1024/1024:.1f}MB)", flush=True)
        time.sleep(0.3)

    if failed:
        return {"status": "PARTIAL", "repo": repo, "failed": failed, "dest": str(dest)}
    return {"status": "OK", "repo": repo, "files": len(siblings), "dest": str(dest)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
        help="all | embedding | embedding_zh | reranker | asr | asr_large")
    args = parser.parse_args()

    if args.task == "all":
        keys = list(MODELS.keys())
    else:
        keys = [k.strip() for k in args.task.split(",")]

    results = {}
    for k in keys:
        if k not in MODELS:
            print(f"Unknown task: {k} (available: {list(MODELS.keys())})", flush=True)
            continue
        results[k] = download_model(k, MODELS[k])

    print(f"\n{'='*50}", flush=True)
    print("DOWNLOAD SUMMARY:", flush=True)
    for k, r in results.items():
        icon = "OK" if r["status"] == "OK" else ("PARTIAL" if r["status"] == "PARTIAL" else "FAIL")
        print(f"  [{icon}] {k}: {r['status']}", flush=True)
        if r["status"] in ("FAIL", "PARTIAL"):
            print(f"    → {r.get('error', r.get('failed', ''))}", flush=True)

    out_path = Path(r"C:\npu_download_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}", flush=True)


if __name__ == "__main__":
    main()
