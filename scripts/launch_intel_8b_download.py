"""Download Qwen3-8B INT4 OV model to Intel machine (for iGPU 7B GA test)."""
import subprocess
import sys
import os
import time

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

env = dict(os.environ)
env["HF_ENDPOINT"] = "https://hf-mirror.com"
env["PYTHONUNBUFFERED"] = "1"

logfile = r"C:\Users\happy\dl_8b.log"

# Inline download script - avoids modifying dl_qwen3_ov.py
code = r"""
import os, sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

repo_id = 'OpenVINO/Qwen3-8B-int4-ov'
local_dir = Path(r'C:\ov_models\llm\qwen3-8b-int4-ov')

if (local_dir / 'config.json').exists():
    log.info('Already downloaded at %s', local_dir)
    sys.exit(0)

local_dir.mkdir(parents=True, exist_ok=True)
log.info('Downloading %s -> %s ...', repo_id, local_dir)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    log.error('pip install huggingface-hub')
    sys.exit(1)

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'],
    )
    log.info('Download complete: %s', local_dir)
except Exception as e:
    log.error('Download failed: %s', e)
    sys.exit(1)
"""

script_path = r"C:\Users\happy\dl_8b_inline.py"
with open(script_path, "w") as f:
    f.write(code)

with open(logfile, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Downloading Qwen3-8B-int4-ov\n")
    proc = subprocess.Popen(
        [sys.executable, script_path],
        env=env,
        stdout=f, stderr=f,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )

print(f"PID: {proc.pid}  log: {logfile}")
