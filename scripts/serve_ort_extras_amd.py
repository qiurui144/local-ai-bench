"""
AMD iGPU (Radeon 780M) ONNX Runtime + DirectML multi-service server (port 8091).

Endpoints:
  POST /v1/embeddings          — BGE-base-en-v1.5 ONNX on Radeon 780M (DirectML)
  POST /v1/rerank              — BGE-reranker-base ONNX on Radeon 780M (DirectML)
  POST /asr/transcribe         — Whisper-base ONNX on iGPU/CPU (DirectML or CPU)
  GET  /v1/models              — liveness / readiness probe

Model dirs (Windows paths):
  EMB   = C:\\ort_models\\embedding\\bge-base-en-v1.5
  RANK  = C:\\ort_models\\reranker\\bge-reranker-base
  ASR   = C:\\ort_models\\asr\\whisper-base

Run:
  python serve_ort_extras_amd.py
  python serve_ort_extras_amd.py --no-asr
  python serve_ort_extras_amd.py --emb-device CPU

Setup (AMD machine, Python 3.11 with onnxruntime-directml):
  pip install onnxruntime-directml transformers optimum soundfile fastapi uvicorn[standard]
  python scripts/dl_ort_models_amd.py   # download ONNX models to C:\\ort_models\\

Harness env: ORT_AMD_EXTRAS_BASE_URL=http://192.168.100.201:8091/v1
"""

import argparse
import io
import logging
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMB_DIR  = Path(r"C:\ort_models\embedding\bge-base-en-v1.5")
RANK_DIR = Path(r"C:\ort_models\reranker\bge-reranker-base")
ASR_DIR  = Path(r"C:\ort_models\asr\whisper-base")

_emb_tok = _emb_sess = None
_rank_tok = _rank_sess = None
_asr_proc = _asr_sess = None

_EMB_DEVICE  = "DML"
_RANK_DEVICE = "DML"
_ASR_DEVICE  = "DML"
_ASR_ENABLED = True


def _pick_providers(device: str):
    """Return ORT provider list for requested device."""
    if device.upper() in ("DML", "DIRECTML", "GPU"):
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_embedding(device: str = "DML"):
    global _emb_tok, _emb_sess
    if _emb_sess is not None:
        return _emb_tok, _emb_sess
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = EMB_DIR / "onnx" / "model_quantized.onnx"
    if not model_path.exists():
        model_path = EMB_DIR / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Embedding ONNX not found under {EMB_DIR}")

    log.info("Loading BGE embedding from %s on %s ...", model_path, device)
    t0 = time.monotonic()
    providers = _pick_providers(device)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _emb_sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    _emb_tok = AutoTokenizer.from_pretrained(str(EMB_DIR))
    log.info("BGE embedding loaded (providers=%s) in %.0fms",
             _emb_sess.get_providers(), (time.monotonic() - t0) * 1000)
    return _emb_tok, _emb_sess


def _load_reranker(device: str = "DML"):
    global _rank_tok, _rank_sess
    if _rank_sess is not None:
        return _rank_tok, _rank_sess
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = RANK_DIR / "onnx" / "model_quantized.onnx"
    if not model_path.exists():
        model_path = RANK_DIR / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Reranker ONNX not found under {RANK_DIR}")

    log.info("Loading BGE reranker from %s on %s ...", model_path, device)
    t0 = time.monotonic()
    providers = _pick_providers(device)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _rank_sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    _rank_tok = AutoTokenizer.from_pretrained(str(RANK_DIR))
    log.info("BGE reranker loaded (providers=%s) in %.0fms",
             _rank_sess.get_providers(), (time.monotonic() - t0) * 1000)
    return _rank_tok, _rank_sess


def _load_asr(device: str = "DML"):
    global _asr_proc, _asr_sess
    if _asr_sess is not None:
        return _asr_proc, _asr_sess
    import onnxruntime as ort
    from transformers import AutoProcessor

    encoder_path = ASR_DIR / "onnx" / "encoder_model.onnx"
    if not encoder_path.exists():
        encoder_path = ASR_DIR / "encoder_model.onnx"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Whisper encoder ONNX not found under {ASR_DIR}")

    log.info("Loading Whisper ASR from %s on %s ...", ASR_DIR, device)
    t0 = time.monotonic()
    providers = _pick_providers(device)
    opts = ort.SessionOptions()
    _asr_sess = ort.InferenceSession(str(encoder_path), sess_options=opts, providers=providers)
    _asr_proc = AutoProcessor.from_pretrained(str(ASR_DIR))
    log.info("Whisper ASR loaded (providers=%s) in %.0fms",
             _asr_sess.get_providers(), (time.monotonic() - t0) * 1000)
    return _asr_proc, _asr_sess


try:
    from fastapi import FastAPI, Request as _FastAPIRequest, HTTPException
    import uvicorn
except ImportError:
    sys.exit("fastapi and uvicorn required: pip install fastapi uvicorn[standard]")

app = FastAPI(title="ORT AMD extras server", version="1.0")


@app.get("/v1/models")
async def list_models():
    models = []
    if (EMB_DIR / "onnx" / "model_quantized.onnx").exists() or (EMB_DIR / "model.onnx").exists():
        models.append({"id": "bge-base-en-v1.5-ort-dml", "object": "model"})
    if (RANK_DIR / "onnx" / "model_quantized.onnx").exists() or (RANK_DIR / "model.onnx").exists():
        models.append({"id": "bge-reranker-base-ort-dml", "object": "model"})
    if _ASR_ENABLED and (ASR_DIR / "onnx" / "encoder_model.onnx").exists():
        models.append({"id": "whisper-base-ort-dml", "object": "model"})
    return {"object": "list", "data": models}


@app.post("/v1/embeddings")
async def embeddings(request: _FastAPIRequest):

    body = await request.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    inputs = list(inputs)
    if not inputs:
        return {"data": [], "model": "bge-base-en-v1.5-ort-dml", "usage": {}}

    try:
        tok, sess = _load_embedding(_EMB_DEVICE)
    except Exception as e:
        raise HTTPException(500, f"embedding model load failed: {e}")

    t0 = time.monotonic()
    try:
        encoded = tok(inputs, return_tensors="np", padding=True,
                      truncation=True, max_length=512)
        inp_names = {i.name for i in sess.get_inputs()}
        feed = {k: v for k, v in encoded.items() if k in inp_names}
        out = sess.run(None, feed)
        # out[0] shape: [batch, seq_len, hidden] or [batch, hidden]
        hidden = out[0]
        if hidden.ndim == 3:
            vecs = hidden[:, 0, :].tolist()   # CLS pooling
        else:
            vecs = hidden.tolist()
    except Exception as e:
        raise HTTPException(500, f"embedding inference failed: {e}")

    elapsed_ms = (time.monotonic() - t0) * 1000
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v}
                 for i, v in enumerate(vecs)],
        "model": body.get("model", "bge-base-en-v1.5-ort-dml"),
        "usage": {
            "prompt_tokens": sum(len(s.split()) for s in inputs),
            "total_tokens":  sum(len(s.split()) for s in inputs),
        },
        "_perf": {"latency_ms": round(elapsed_ms, 1)},
    }


@app.post("/v1/rerank")
async def rerank(request: _FastAPIRequest):
    import numpy as np

    body = await request.json()
    query = body.get("query", "")
    docs  = body.get("documents", [])
    if not docs:
        return {"results": [], "model": "bge-reranker-base-ort-dml"}

    try:
        tok, sess = _load_reranker(_RANK_DEVICE)
    except Exception as e:
        raise HTTPException(500, f"reranker model load failed: {e}")

    t0 = time.monotonic()
    try:
        pairs_q = [query] * len(docs)
        encoded = tok(pairs_q, docs, return_tensors="np", padding=True,
                      truncation=True, max_length=512)
        inp_names = {i.name for i in sess.get_inputs()}
        feed = {k: v for k, v in encoded.items() if k in inp_names}
        out = sess.run(None, feed)
        logits = out[0]  # shape [batch, 1] or [batch]
        if logits.ndim > 1:
            logits = logits[:, 0]
        scores = (1.0 / (1.0 + np.exp(-logits))).tolist()  # sigmoid
    except Exception as e:
        raise HTTPException(500, f"reranker inference failed: {e}")

    elapsed_ms = (time.monotonic() - t0) * 1000
    results = [
        {"index": i, "relevance_score": float(s), "document": {"text": d}}
        for i, (s, d) in enumerate(zip(scores, docs))
    ]
    results.sort(key=lambda x: -x["relevance_score"])
    return {
        "results": results,
        "model": body.get("model", "bge-reranker-base-ort-dml"),
        "_perf": {"latency_ms": round(elapsed_ms, 1)},
    }


@app.post("/asr/transcribe")
async def asr_transcribe(request: _FastAPIRequest):
    """Transcribe raw WAV bytes. Content-Type: audio/wav."""
    if not _ASR_ENABLED:
        raise HTTPException(503, "ASR disabled (--no-asr)")

    try:
        proc, enc_sess = _load_asr(_ASR_DEVICE)
    except Exception as e:
        raise HTTPException(500, f"ASR model load failed: {e}")

    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(400, "empty audio body")

    t0 = time.monotonic()
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        # Whisper encoder: get mel features
        inputs = proc(audio, sampling_rate=16000, return_tensors="np")
        mel = inputs.input_features  # [1, 80, 3000]

        inp_names = {i.name for i in enc_sess.get_inputs()}
        feed = {}
        if "input_features" in inp_names:
            feed["input_features"] = mel
        else:
            feed[enc_sess.get_inputs()[0].name] = mel

        enc_out = enc_sess.run(None, feed)

        # For full transcription we need decoder too; for now use optimum pipeline
        # Try optimum WhisperPipeline if available
        try:
            from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
            from transformers import pipeline as hf_pipeline

            model = ORTModelForSpeechSeq2Seq.from_pretrained(
                str(ASR_DIR),
                provider="DmlExecutionProvider" if _ASR_DEVICE.upper() != "CPU" else "CPUExecutionProvider",
            )
            pipe = hf_pipeline("automatic-speech-recognition", model=model, tokenizer=proc,
                                feature_extractor=proc.feature_extractor)
            result = pipe(audio)
            text = result["text"].strip()
        except Exception:
            # Fallback: encoder-only, return placeholder
            text = f"[encoder-only: shape={enc_out[0].shape}]"

    except Exception as e:
        raise HTTPException(500, f"ASR inference failed: {e}")

    elapsed_ms = (time.monotonic() - t0) * 1000
    return {"result": text, "_perf": {"latency_ms": round(elapsed_ms, 1)}}


def main():
    global _EMB_DEVICE, _RANK_DEVICE, _ASR_DEVICE, _ASR_ENABLED

    parser = argparse.ArgumentParser()
    parser.add_argument("--host",        default="0.0.0.0")
    parser.add_argument("--port",        type=int, default=8091)
    parser.add_argument("--emb-device",  default="DML",
                        help="ORT device for embedding: DML (iGPU DirectML) or CPU")
    parser.add_argument("--rank-device", default="DML",
                        help="ORT device for reranker: DML or CPU")
    parser.add_argument("--asr-device",  default="DML",
                        help="ORT device for ASR: DML or CPU")
    parser.add_argument("--no-asr",      action="store_true")
    parser.add_argument("--preload",     action="store_true")
    args = parser.parse_args()

    _EMB_DEVICE  = args.emb_device
    _RANK_DEVICE = args.rank_device
    _ASR_DEVICE  = args.asr_device
    _ASR_ENABLED = not args.no_asr

    log.info("ORT AMD extras server starting on %s:%d", args.host, args.port)
    log.info("  embedding  → %s  device=%s", EMB_DIR,  _EMB_DEVICE)
    log.info("  reranker   → %s  device=%s", RANK_DIR, _RANK_DEVICE)
    log.info("  asr        → %s  %s", ASR_DIR, "ENABLED" if _ASR_ENABLED else "DISABLED")

    if args.preload:
        try:
            _load_embedding(_EMB_DEVICE)
        except Exception as e:
            log.warning("Embedding preload failed: %s", e)
        try:
            _load_reranker(_RANK_DEVICE)
        except Exception as e:
            log.warning("Reranker preload failed: %s", e)
        if _ASR_ENABLED:
            try:
                _load_asr(_ASR_DEVICE)
            except Exception as e:
                log.warning("ASR preload failed: %s", e)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
