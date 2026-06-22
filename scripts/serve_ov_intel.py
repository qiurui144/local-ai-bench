"""Thin OpenAI-compatible HTTP server for Intel iGPU via optimum-intel.

Exposes two endpoints so the benchmark harness can use iGPU instead of CPU:
  POST /v1/chat/completions   — OVModelForCausalLM on Arc GPU
  POST /v1/embeddings         — OVModelForFeatureExtraction on Arc GPU
  GET  /v1/models             — list loaded models

Usage (run on the Intel Windows machine):
    pip install fastapi uvicorn optimum[openvino]
    python serve_ov_intel.py \\
        --llm  C:\\ov_models\\qwen2.5-7b-int4-ov  --llm-device GPU \\
        --emb  C:\\ov_models\\embedding\\bge-base-en-v1.5-int8-ov  --emb-device GPU \\
        --port 8080

Benchmark models.yaml entry:
    base_url_env: OV_INTEL_BASE_URL   # default http://192.168.100.116:8080
    provider: openai

Notes:
- OVModelForCausalLM at device=GPU is ~3x slower than openvino_genai LLMPipeline
  (no KV-cache fusion). Use until LLMPipeline DLL issue is resolved.
- Cold-start compile: 7B ~ 115s, 1.5B ~ 54s. Keep server running between runs.
- Embedding warm latency: ~25 ms (bge-base-en INT8 on Arc).
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy global singletons — loaded once on first request to avoid startup delay
# ──────────────────────────────────────────────────────────────────────────────

_llm_model = None
_llm_tokenizer = None
_llm_name: str = ""

_emb_model = None
_emb_tokenizer = None
_emb_name: str = ""


def _load_llm(model_dir: str, device: str) -> None:
    global _llm_model, _llm_tokenizer, _llm_name
    if _llm_model is not None:
        return
    logger.info("Loading LLM from %s on %s (cold compile may take 60-120s)…", model_dir, device)
    t0 = time.monotonic()
    from optimum.intel import OVModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    _llm_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _llm_model = OVModelForCausalLM.from_pretrained(model_dir, device=device)
    _llm_name = model_dir.rstrip("/\\").split("\\")[-1].split("/")[-1]
    logger.info("LLM loaded in %.1fs", time.monotonic() - t0)


def _load_emb(model_dir: str, device: str) -> None:
    global _emb_model, _emb_tokenizer, _emb_name
    if _emb_model is not None:
        return
    logger.info("Loading embedding model from %s on %s…", model_dir, device)
    t0 = time.monotonic()
    from optimum.intel import OVModelForFeatureExtraction  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    _emb_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _emb_model = OVModelForFeatureExtraction.from_pretrained(model_dir, device=device)
    _emb_name = model_dir.rstrip("/\\").split("\\")[-1].split("/")[-1]
    logger.info("Embedding model loaded in %.1fs", time.monotonic() - t0)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

def build_app(
    llm_dir: Optional[str],
    llm_device: str,
    emb_dir: Optional[str],
    emb_device: str,
):
    try:
        from fastapi import FastAPI, HTTPException  # type: ignore
        from pydantic import BaseModel  # type: ignore
    except ImportError as e:
        raise SystemExit(f"Install fastapi + uvicorn: pip install fastapi uvicorn\n{e}")

    import torch  # type: ignore

    app = FastAPI(title="OV Intel iGPU Server")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = ""
        messages: list[ChatMessage]
        max_tokens: int = 512
        temperature: float = 0.7

    class EmbRequest(BaseModel):
        model: str = ""
        input: list[str] | str

    @app.get("/v1/models")
    def list_models():
        models = []
        if llm_dir:
            models.append({"id": _llm_name or "llm", "object": "model"})
        if emb_dir:
            models.append({"id": _emb_name or "embedding", "object": "model"})
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if not llm_dir:
            raise HTTPException(status_code=501, detail="LLM not configured (--llm flag missing)")
        _load_llm(llm_dir, llm_device)
        prompt = _llm_tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in req.messages],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = _llm_tokenizer(prompt, return_tensors="pt")
        t0 = time.monotonic()
        with torch.no_grad():
            out = _llm_model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature if req.temperature > 0 else 1.0,
                do_sample=req.temperature > 0,
            )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        text = _llm_tokenizer.decode(gen_ids, skip_special_tokens=True)
        elapsed = time.monotonic() - t0
        n_tok = len(gen_ids)
        return {
            "id": "chatcmpl-ov",
            "object": "chat.completion",
            "model": _llm_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": inputs.input_ids.shape[1],
                      "completion_tokens": n_tok, "total_tokens": inputs.input_ids.shape[1] + n_tok},
            "_perf": {"latency_s": elapsed, "tps": n_tok / elapsed if elapsed > 0 else 0},
        }

    @app.post("/v1/embeddings")
    def embeddings(req: EmbRequest):
        if not emb_dir:
            raise HTTPException(status_code=501, detail="Embedding not configured (--emb flag missing)")
        _load_emb(emb_dir, emb_device)
        texts = [req.input] if isinstance(req.input, str) else req.input
        inputs = _emb_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            out = _emb_model(**inputs)
        # CLS pooling
        vecs = out.last_hidden_state[:, 0, :].tolist()
        data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]
        return {"object": "list", "data": data, "model": _emb_name}

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="OV Intel iGPU OpenAI-compat server")
    p.add_argument("--llm", default="", help="Path to OV LLM model dir (e.g. C:\\ov_models\\qwen2.5-7b-int4-ov)")
    p.add_argument("--llm-device", default="GPU", help="Device for LLM (GPU or CPU)")
    p.add_argument("--emb", default="", help="Path to OV embedding model dir")
    p.add_argument("--emb-device", default="GPU", help="Device for embedding (GPU or CPU)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    if not args.llm and not args.emb:
        p.error("Specify at least one of --llm or --emb")

    app = build_app(
        llm_dir=args.llm or None,
        llm_device=args.llm_device,
        emb_dir=args.emb or None,
        emb_device=args.emb_device,
    )

    try:
        import uvicorn  # type: ignore
    except ImportError:
        raise SystemExit("Install uvicorn: pip install uvicorn")

    logger.info("Starting OV Intel server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
