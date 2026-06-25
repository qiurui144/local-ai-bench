"""
Download ONNX models for AMD iGPU DirectML server (serve_ort_extras_amd.py).

Models downloaded to C:\\ort_models\\:
  embedding/bge-base-en-v1.5/   — Xenova/bge-base-en-v1.5 (ONNX quantized INT8)
  reranker/bge-reranker-base/   — Xenova/bge-reranker-base (ONNX FP32)
  asr/whisper-base/             — Xenova/whisper-base (ONNX, encoder+decoder)

Usage (on AMD Windows machine):
  python dl_ort_models_amd.py            # download all
  python dl_ort_models_amd.py --task embedding
  python dl_ort_models_amd.py --hf-mirror https://hf-mirror.com

Dependencies:
  pip install huggingface_hub
"""

import argparse
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(r"C:\ort_models")

# Models to download: (repo_id, subdir, local_dir, files_to_download)
MODELS = {
    "embedding": (
        "Xenova/bge-base-en-v1.5",
        None,
        BASE_DIR / "embedding" / "bge-base-en-v1.5",
        [
            "onnx/model.onnx",
            "onnx/model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ],
    ),
    "reranker": (
        "Xenova/bge-reranker-base",
        None,
        BASE_DIR / "reranker" / "bge-reranker-base",
        [
            "onnx/model.onnx",
            "onnx/model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ],
    ),
    "asr": (
        "Xenova/whisper-base",
        None,
        BASE_DIR / "asr" / "whisper-base",
        [
            "onnx/encoder_model.onnx",
            "onnx/decoder_model.onnx",
            "onnx/decoder_with_past_model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
            "vocab.json",
            "merges.txt",
            "normalizer.json",
            "preprocessor_config.json",
        ],
    ),
}


def download_model(task: str, hf_mirror: str | None = None):
    from huggingface_hub import hf_hub_download

    repo_id, subdir, local_dir, files = MODELS[task]
    local_dir.mkdir(parents=True, exist_ok=True)

    if hf_mirror:
        import os
        os.environ.setdefault("HF_ENDPOINT", hf_mirror)

    print(f"\n[{task}] Downloading {repo_id} → {local_dir}", flush=True)
    ok = True
    for fname in files:
        dst = local_dir / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"  [skip] {fname} already exists", flush=True)
            continue
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=str(local_dir),
            )
            print(f"  [OK] {fname} → {path}", flush=True)
        except Exception as e:
            print(f"  [WARN] {fname} download failed: {e}", flush=True)
            ok = False

    if ok:
        print(f"[{task}] Done — {local_dir}", flush=True)
    else:
        print(f"[{task}] Partial — some files missing (see above)", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        help="all | embedding | reranker | asr")
    parser.add_argument("--hf-mirror", default=None,
                        help="HF mirror URL (e.g. https://hf-mirror.com)")
    args = parser.parse_args()

    tasks = list(MODELS.keys()) if args.task == "all" else args.task.split(",")
    for task in tasks:
        if task in MODELS:
            download_model(task, hf_mirror=args.hf_mirror)
        else:
            print(f"[WARN] Unknown task: {task}", flush=True)

    print("\nAll done. Run serve_ort_extras_amd.py to start the server.", flush=True)
    print(f"  python serve_ort_extras_amd.py --port 8091", flush=True)


if __name__ == "__main__":
    main()
