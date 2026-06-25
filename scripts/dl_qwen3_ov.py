"""
Qwen3 OpenVINO INT4 模型下载脚本 (Intel Windows)

下载目标:
  OpenVINO/Qwen3-0.6B-int4-ov  → C:\ov_models\llm\qwen3-0.6b-int4-ov\
  OpenVINO/Qwen3-1.7B-int4-ov  → C:\ov_models\llm\qwen3-1.7b-int4-ov\
  OpenVINO/Qwen3-4B-int4-ov    → C:\ov_models\llm\qwen3-4b-int4-ov\   (可选)

用法:
  python dl_qwen3_ov.py               # 下载 0.6B + 1.7B
  python dl_qwen3_ov.py --all         # 同时下载 4B
  python dl_qwen3_ov.py --model 0.6B  # 只下载指定模型

依赖: pip install huggingface-hub
镜像: HF_ENDPOINT=https://hf-mirror.com (若直连 HF 慢)
"""

import argparse
import logging
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── 模型配置 ──────────────────────────────────────────────────────────────────

MODELS = {
    "0.6B": {
        "repo_id": "OpenVINO/Qwen3-0.6B-int4-ov",
        "local_dir": Path(r"C:\ov_models\llm\qwen3-0.6b-int4-ov"),
    },
    "1.7B": {
        "repo_id": "OpenVINO/Qwen3-1.7B-int4-ov",
        "local_dir": Path(r"C:\ov_models\llm\qwen3-1.7b-int4-ov"),
    },
    "4B": {
        "repo_id": "OpenVINO/Qwen3-4B-int4-ov",
        "local_dir": Path(r"C:\ov_models\llm\qwen3-4b-int4-ov"),
    },
}

# ── HF mirror support ─────────────────────────────────────────────────────────

def _try_hf_mirror():
    """If HF direct is slow, try hf-mirror.com."""
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if not endpoint:
        log.info("Tip: set HF_ENDPOINT=https://hf-mirror.com if HuggingFace is slow")


def download_model(name: str, repo_id: str, local_dir: Path) -> bool:
    if (local_dir / "config.json").exists():
        log.info("[%s] Already downloaded at %s, skipping.", name, local_dir)
        return True

    local_dir.mkdir(parents=True, exist_ok=True)
    log.info("[%s] Downloading %s → %s ...", name, repo_id, local_dir)

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        log.error("huggingface-hub not installed. Run: pip install huggingface-hub")
        return False

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        log.info("[%s] Download complete: %s", name, local_dir)
        return True
    except Exception as e:
        log.error("[%s] Download failed: %s", name, e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Download a specific model size only")
    parser.add_argument("--all", action="store_true",
                        help="Download all models including 4B")
    args = parser.parse_args()

    _try_hf_mirror()

    if args.model:
        targets = {args.model: MODELS[args.model]}
    elif args.all:
        targets = MODELS
    else:
        # Default: 0.6B + 1.7B (skip 4B as speed is borderline)
        targets = {k: v for k, v in MODELS.items() if k != "4B"}

    results = {}
    for name, cfg in targets.items():
        ok = download_model(name, cfg["repo_id"], cfg["local_dir"])
        results[name] = "OK" if ok else "FAIL"

    log.info("=== Download summary ===")
    for name, status in results.items():
        log.info("  Qwen3-%s: %s", name, status)

    if any(s == "FAIL" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
