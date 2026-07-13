"""
Intel Arc iGPU + NPU multi-service OV server (port 8081).

Endpoints:
  POST /v1/embeddings          — BGE-base-en-v1.5 INT8 on Arc iGPU (GPU)
  POST /v1/rerank              — BGE-reranker-base INT8 on Arc iGPU (GPU)
  POST /asr/transcribe         — Whisper-base INT8 on Intel AI Boost (NPU encoder / CPU decoder)
  GET  /v1/models              — liveness / readiness probe

Model dirs (Windows paths):
  EMB   = C:\\ov_models\\embedding\\bge-base-en-v1.5-int8-ov
  RANK  = C:\\ov_models\\reranker\\bge-reranker-base-int8-ov
  ASR   = C:\\ov_models\\asr\\whisper-base-int8-ov

Run:
  python serve_ov_extras.py
  python serve_ov_extras.py --no-asr        (skip ASR if NPU unavailable)
  python serve_ov_extras.py --emb-device CPU (CPU fallback for embedding)

Harness env: OV_EXTRAS_INTEL_BASE_URL=http://192.168.100.116:8081/v1
"""

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

# Fix GBK console on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── model dirs ────────────────────────────────────────────────────────────────

EMB_DIR  = Path(os.environ.get("OV_EXTRAS_EMB_DIR",  r"C:\ov_models\embedding\bge-base-en-v1.5-int8-ov"))
RANK_DIR = Path(os.environ.get("OV_EXTRAS_RANK_DIR", r"C:\ov_models\reranker\bge-reranker-base-int8-ov"))
ASR_DIR  = Path(os.environ.get("OV_EXTRAS_ASR_DIR",  r"C:\ov_models\asr\whisper-base-int8-ov"))

# ── lazy model singletons ─────────────────────────────────────────────────────

_emb_tok = _emb_model = None
_rank_tok = _rank_model = None
_asr_proc = _asr_model = None


def _load_embedding(device: str = "GPU"):
    global _emb_tok, _emb_model
    if _emb_model is not None:
        return _emb_tok, _emb_model
    from optimum.intel import OVModelForFeatureExtraction  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    log.info("Loading BGE embedding from %s on %s ...", EMB_DIR, device)
    t0 = time.monotonic()
    _emb_tok = AutoTokenizer.from_pretrained(str(EMB_DIR))
    _emb_model = OVModelForFeatureExtraction.from_pretrained(str(EMB_DIR), device=device)
    log.info("BGE embedding loaded in %.0fms", (time.monotonic() - t0) * 1000)
    return _emb_tok, _emb_model


def _load_reranker(device: str = "GPU"):
    global _rank_tok, _rank_model
    if _rank_model is not None:
        return _rank_tok, _rank_model
    from optimum.intel import OVModelForSequenceClassification  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    log.info("Loading BGE reranker from %s on %s ...", RANK_DIR, device)
    t0 = time.monotonic()
    _rank_tok = AutoTokenizer.from_pretrained(str(RANK_DIR))
    _rank_model = OVModelForSequenceClassification.from_pretrained(str(RANK_DIR), device=device)
    log.info("BGE reranker loaded in %.0fms", (time.monotonic() - t0) * 1000)
    return _rank_tok, _rank_model


def _load_asr():
    global _asr_proc, _asr_model
    if _asr_model is not None:
        return _asr_proc, _asr_model
    import openvino as ov  # type: ignore
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # type: ignore
    from transformers import AutoProcessor  # type: ignore

    available = ov.Core().available_devices
    if _ASR_DEVICE_OVERRIDE:
        # User specified device (e.g. --asr-device GPU): skip NPU to avoid LLVM C++ crash
        enc_device = _ASR_DEVICE_OVERRIDE
        candidates = [enc_device]
        if enc_device != "CPU":
            candidates.append("CPU")
        log.info("Loading Whisper ASR from %s, encoder=%s (override) ...", ASR_DIR, enc_device)
    else:
        # Auto-detect: prefer NPU encoder, but NPU may crash (LLVM dynamic-shape error)
        enc_device = "NPU" if "NPU" in available else ("GPU" if "GPU" in available else "CPU")
        device_map = {
            "encoder_model": enc_device,
            "decoder_model": "CPU",
            "decoder_with_past_model": "CPU",
        }
        # Try per-sub-model device dict (optimum-intel ≥ 1.18) first;
        # fall back to a single string device for older versions.
        # NPU often fails with dynamic-shape LLVM errors; fall back gracefully.
        candidates = [device_map, enc_device]
        if enc_device not in ("GPU", "CPU"):
            candidates.extend(["GPU", "CPU"])
        log.info("Loading Whisper ASR from %s, encoder=%s ...", ASR_DIR, enc_device)
    _asr_proc = AutoProcessor.from_pretrained(str(ASR_DIR))
    last_err = None
    for candidate in candidates:
        try:
            t1 = time.monotonic()
            _asr_model = OVModelForSpeechSeq2Seq.from_pretrained(str(ASR_DIR), device=candidate)
            log.info("Whisper ASR loaded (device=%s) in %.0fms", candidate, (time.monotonic() - t1) * 1000)
            last_err = None
            break
        except Exception as e:
            log.warning("Whisper ASR load failed with device=%s: %s", candidate, str(e)[:120])
            last_err = e
            _asr_model = None
    if _asr_model is None:
        raise RuntimeError(f"Whisper ASR: all device candidates failed. Last: {last_err}")
    return _asr_proc, _asr_model


# ── FastAPI app ───────────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, Request as _FastAPIRequest, HTTPException  # type: ignore
    import uvicorn  # type: ignore
except ImportError:
    sys.exit("fastapi and uvicorn required: pip install fastapi uvicorn[standard]")

app = FastAPI(title="OV extras server", version="1.0")

_EMB_DEVICE  = "GPU"
_RANK_DEVICE = "GPU"
_ASR_ENABLED = True
_ASR_DEVICE_OVERRIDE = None  # set via --asr-device; bypasses NPU to avoid LLVM crash


@app.get("/v1/models")
async def list_models():
    models = []
    if EMB_DIR.exists():
        models.append({"id": "bge-base-en-v1.5-int8-ov", "object": "model"})
    if RANK_DIR.exists():
        models.append({"id": "bge-reranker-base-int8-ov", "object": "model"})
    if ASR_DIR.exists() and _ASR_ENABLED:
        models.append({"id": "whisper-base-int8-ov", "object": "model"})
    return {"object": "list", "data": models}


@app.post("/v1/embeddings")
async def embeddings(request: _FastAPIRequest):
    import torch  # type: ignore
    body = await request.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    inputs = list(inputs)
    if not inputs:
        return {"data": [], "model": "bge-base-en-v1.5-int8-ov", "usage": {}}

    try:
        tok, model = _load_embedding(_EMB_DEVICE)
    except Exception as e:
        raise HTTPException(500, f"embedding model load failed: {e}")

    t0 = time.monotonic()
    try:
        encoded = tok(inputs, return_tensors="pt", padding=True,
                      truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**encoded)
        # CLS-token pooling (standard for BGE)
        vecs = out.last_hidden_state[:, 0, :].tolist()
    except Exception as e:
        raise HTTPException(500, f"embedding inference failed: {e}")

    elapsed_ms = (time.monotonic() - t0) * 1000
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vecs)
        ],
        "model": body.get("model", "bge-base-en-v1.5-int8-ov"),
        "usage": {
            "prompt_tokens": sum(len(s.split()) for s in inputs),
            "total_tokens":  sum(len(s.split()) for s in inputs),
        },
        "_perf": {"latency_ms": round(elapsed_ms, 1)},
    }


@app.post("/v1/rerank")
async def rerank(request: _FastAPIRequest):
    import torch  # type: ignore
    body = await request.json()
    query = body.get("query", "")
    docs  = body.get("documents", [])
    if not docs:
        return {"results": [], "model": "bge-reranker-base-int8-ov"}

    try:
        tok, model = _load_reranker(_RANK_DEVICE)
    except Exception as e:
        raise HTTPException(500, f"reranker model load failed: {e}")

    t0 = time.monotonic()
    try:
        pairs_q = [query] * len(docs)
        encoded = tok(pairs_q, docs, return_tensors="pt", padding=True,
                      truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**encoded)
        scores = torch.sigmoid(out.logits[:, 0]).tolist()
    except Exception as e:
        raise HTTPException(500, f"reranker inference failed: {e}")

    elapsed_ms = (time.monotonic() - t0) * 1000
    results = [
        {"index": i, "relevance_score": float(s), "document": {"text": d}}
        for i, (s, d) in enumerate(zip(scores, docs))
    ]
    # Sort by score descending (caller re-aligns by index)
    results.sort(key=lambda x: -x["relevance_score"])
    return {
        "results": results,
        "model": body.get("model", "bge-reranker-base-int8-ov"),
        "_perf": {"latency_ms": round(elapsed_ms, 1)},
    }


@app.post("/asr/transcribe")
async def asr_transcribe(request: _FastAPIRequest):
    """Transcribe raw WAV bytes. Content-Type: audio/wav."""
    if not _ASR_ENABLED:
        raise HTTPException(503, "ASR disabled (--no-asr)")
    if not ASR_DIR.exists():
        raise HTTPException(503, f"ASR model not found: {ASR_DIR}")

    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(400, "empty audio body")

    helper = Path(__file__).with_name("whisper_ov_transcribe.py")
    if not helper.exists():
        raise HTTPException(500, f"ASR helper not found: {helper}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        wav_path = Path(tmp.name)
    try:
        cmd = [
            sys.executable,
            str(helper),
            "--model-dir", str(ASR_DIR),
            "--wav", str(wav_path),
            "--device", _ASR_DEVICE_OVERRIDE or "CPU",
            "--language", "zh",
            "--task", "transcribe",
        ]
        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=240,
        )
        if proc.returncode != 0:
            raise HTTPException(
                500,
                "ASR subprocess failed: "
                f"rc={proc.returncode}; stderr={proc.stderr[-1000:]}",
            )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            raise HTTPException(500, f"ASR subprocess produced no output; stderr={proc.stderr[-1000:]}")
        try:
            return json.loads(lines[-1])
        except Exception as e:
            raise HTTPException(
                500,
                f"ASR subprocess returned invalid JSON: {e}; stdout={proc.stdout[-1000:]}; stderr={proc.stderr[-1000:]}",
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "ASR subprocess timed out")
    finally:
        try:
            wav_path.unlink()
        except Exception:
            pass


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    global _EMB_DEVICE, _RANK_DEVICE, _ASR_ENABLED, _ASR_DEVICE_OVERRIDE

    parser = argparse.ArgumentParser()
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8081)
    parser.add_argument("--emb-device", default="GPU",
                        help="OpenVINO device for embedding (GPU/CPU/NPU)")
    parser.add_argument("--rank-device", default="GPU",
                        help="OpenVINO device for reranker (GPU/CPU/NPU)")
    parser.add_argument("--asr-device", default=None,
                        help="Force ASR to this device (GPU/CPU) — skips NPU to avoid LLVM crash")
    parser.add_argument("--no-asr",     action="store_true",
                        help="Skip loading the ASR model")
    parser.add_argument("--preload",    action="store_true",
                        help="Load all models at startup instead of lazily")
    args = parser.parse_args()

    _EMB_DEVICE  = args.emb_device
    _RANK_DEVICE = args.rank_device
    _ASR_ENABLED = not args.no_asr
    if args.asr_device:
        _ASR_DEVICE_OVERRIDE = args.asr_device
    else:
        _ASR_DEVICE_OVERRIDE = None

    log.info("OV extras server starting on %s:%d", args.host, args.port)
    log.info("  embedding  → %s  device=%s", EMB_DIR,  _EMB_DEVICE)
    log.info("  reranker   → %s  device=%s", RANK_DIR, _RANK_DEVICE)
    log.info("  asr        → %s  %s", ASR_DIR, "ENABLED" if _ASR_ENABLED else "DISABLED")

    if args.preload:
        if EMB_DIR.exists():
            _load_embedding(_EMB_DEVICE)
        if RANK_DIR.exists():
            _load_reranker(_RANK_DEVICE)
        if _ASR_ENABLED and ASR_DIR.exists():
            _load_asr()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
