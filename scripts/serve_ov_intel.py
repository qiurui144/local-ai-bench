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

import argparse
import gc
import logging
import os
import threading
import time
from typing import List, Optional, Union

# Request must be at module scope so typing.get_type_hints() can resolve it
# (from __future__ import annotations would make all annotations lazy strings,
# causing FastAPI to fail resolution when _FastAPIRequest is only a local var)
try:
    from fastapi import Request as _FastAPIRequest  # type: ignore
except ImportError:
    _FastAPIRequest = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pydantic models at module scope so FastAPI resolves type hints correctly
try:
    from pydantic import BaseModel as _BaseModel  # type: ignore

    class _ChatMessage(_BaseModel):
        role: str
        content: str

    class _ChatRequest(_BaseModel):
        model: str = ""
        messages: List[_ChatMessage]
        max_tokens: int = 512
        temperature: float = 0.7
        stream: bool = False
        logprobs: bool = False
        top_logprobs: int = 0

    class _EmbRequest(_BaseModel):
        model: str = ""
        input: Union[List[str], str]

except ImportError:
    _ChatMessage = _ChatRequest = _EmbRequest = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Lazy global singletons — loaded once on first request to avoid startup delay
# ──────────────────────────────────────────────────────────────────────────────

_llm_model = None
_llm_tokenizer = None
_llm_name: str = ""
_llm_completed_requests = 0
_llm_unhealthy_error = ""
_llm_lock = threading.BoundedSemaphore(1)
_hard_exit_scheduled = False

_emb_model = None
_emb_tokenizer = None
_emb_name: str = ""


def _llm_ov_config(device: str, enable_large_allocations: bool) -> dict[str, str]:
    if device.upper().startswith("GPU"):
        cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
        if enable_large_allocations:
            cfg["GPU_ENABLE_LARGE_ALLOCATIONS"] = "YES"
        return cfg
    return {}


def _is_gpu_resource_error(exc: BaseException) -> bool:
    text = str(exc)
    return (
        "CL_OUT_OF_RESOURCES" in text
        or "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" in text
        or "subsequent OpenCL API call may cause the application to hang" in text
    )


def _is_sampling_probability_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "probability tensor contains" in text
        and ("inf" in text or "nan" in text or "element < 0" in text)
    )


def _schedule_hard_exit(reason: str, *, code: int = 120, delay_s: float = 0.5) -> None:
    """Exit the process after the HTTP response has a chance to flush.

    OpenVINO GPU plugin state can be unsafe after OpenCL resource errors. Python
    object unload may call back into the same native runtime and abort. A hard
    process exit lets the external supervisor restart from a clean GPU context.
    """
    global _hard_exit_scheduled
    if _hard_exit_scheduled:
        return
    _hard_exit_scheduled = True

    def _exit_later() -> None:
        time.sleep(max(0.0, delay_s))
        logger.error("Hard exiting process rc=%d: %s", code, reason)
        os._exit(code)

    threading.Thread(target=_exit_later, daemon=True).start()


def _reset_llm_request_state() -> None:
    request = getattr(_llm_model, "request", None)
    reset = getattr(request, "reset_state", None)
    if callable(reset):
        reset()


def _generate_llm_with_sampling_fallback(inputs, max_new_tokens: int, temperature: float):
    kwargs = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        kwargs.update({"temperature": temperature, "do_sample": True})
    else:
        kwargs["do_sample"] = False
    try:
        return _llm_model.generate(**inputs, **kwargs)
    except Exception as exc:
        if _is_gpu_resource_error(exc) or not _is_sampling_probability_error(exc):
            raise
        logger.warning(
            "LLM sampling failed with invalid probability tensor; retrying deterministic decode: %s",
            exc,
        )
        try:
            _reset_llm_request_state()
        except Exception as reset_exc:
            logger.debug("LLM request state reset before deterministic retry failed: %r", reset_exc)
        return _llm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)


def _unload_llm(reason: str) -> None:
    global _llm_model, _llm_tokenizer, _llm_name, _llm_completed_requests
    if _llm_model is not None or _llm_tokenizer is not None:
        logger.info("Unloading LLM (%s)", reason)
    _llm_model = None
    _llm_tokenizer = None
    _llm_name = ""
    _llm_completed_requests = 0
    gc.collect()


def _load_llm(model_dir: str, device: str, ov_config: dict[str, str]) -> None:
    global _llm_model, _llm_tokenizer, _llm_name
    if _llm_model is not None:
        return
    logger.info(
        "Loading LLM from %s on %s (cold compile may take 60-120s), ov_config=%s",
        model_dir,
        device,
        ov_config,
    )
    t0 = time.monotonic()
    from optimum.intel import OVModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    _llm_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _llm_model = OVModelForCausalLM.from_pretrained(model_dir, device=device, ov_config=ov_config or None)
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
    llm_large_allocations: bool,
    llm_max_concurrent: int,
    llm_reload_every: int,
    llm_exit_every: int,
    emb_dir: Optional[str],
    emb_device: str,
):
    global _llm_lock
    try:
        from fastapi import FastAPI, HTTPException  # type: ignore
    except ImportError as e:
        raise SystemExit(f"Install fastapi + uvicorn: pip install fastapi uvicorn\n{e}")

    if _FastAPIRequest is None:
        raise SystemExit("Install fastapi: pip install fastapi uvicorn")

    import torch  # type: ignore

    app = FastAPI(title="OV Intel iGPU Server")
    _llm_lock = threading.BoundedSemaphore(max(1, int(llm_max_concurrent)))

    @app.get("/v1/models")
    def list_models():
        models = []
        if llm_dir:
            models.append({"id": _llm_name or "llm", "object": "model"})
        if emb_dir:
            models.append({"id": _emb_name or "embedding", "object": "model"})
        return {"object": "list", "data": models}

    @app.get("/health")
    def health():
        if _llm_unhealthy_error:
            raise HTTPException(status_code=503, detail=f"LLM unhealthy: {_llm_unhealthy_error}")
        return {"status": "ok", "llm_requests_since_reload": _llm_completed_requests}

    @app.post("/v1/chat/completions")
    async def chat(request: _FastAPIRequest):
        global _llm_completed_requests, _llm_unhealthy_error
        body = await request.json()
        req = _ChatRequest(**body)
        if not llm_dir:
            raise HTTPException(status_code=501, detail="LLM not configured (--llm flag missing)")
        if _llm_unhealthy_error:
            raise HTTPException(status_code=503, detail=f"LLM unhealthy: {_llm_unhealthy_error}")
        with _llm_lock:
            if _llm_unhealthy_error:
                raise HTTPException(status_code=503, detail=f"LLM unhealthy: {_llm_unhealthy_error}")
            if llm_reload_every > 0 and _llm_completed_requests >= llm_reload_every:
                _unload_llm(f"periodic reload after {llm_reload_every} requests")
            try:
                _load_llm(llm_dir, llm_device, _llm_ov_config(llm_device, llm_large_allocations))
                prompt = _llm_tokenizer.apply_chat_template(
                    [{"role": m.role, "content": m.content} for m in req.messages],
                    tokenize=False, add_generation_prompt=True,
                )
                inputs = _llm_tokenizer(prompt, return_tensors="pt")
                t0 = time.monotonic()
                with torch.no_grad():
                    out = _generate_llm_with_sampling_fallback(
                        inputs,
                        max_new_tokens=req.max_tokens,
                        temperature=req.temperature,
                    )
                gen_ids = out[0][inputs.input_ids.shape[1]:]
                text = _llm_tokenizer.decode(gen_ids, skip_special_tokens=True)
                elapsed = time.monotonic() - t0
                n_tok = len(gen_ids)
                response = {
                    "id": "chatcmpl-ov",
                    "object": "chat.completion",
                    "model": _llm_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": inputs.input_ids.shape[1],
                              "completion_tokens": n_tok, "total_tokens": inputs.input_ids.shape[1] + n_tok},
                    "_perf": {"latency_s": elapsed, "tps": n_tok / elapsed if elapsed > 0 else 0},
                }
                _llm_completed_requests += 1
                completed = _llm_completed_requests
                try:
                    _reset_llm_request_state()
                except Exception as reset_exc:
                    logger.debug("LLM request state reset failed: %r", reset_exc)
                del out, inputs
                gc.collect()
                if llm_exit_every > 0 and completed >= llm_exit_every:
                    _llm_unhealthy_error = f"process recycle after {llm_exit_every} successful requests"
                    _schedule_hard_exit(_llm_unhealthy_error, code=0)
                return response
            except Exception as exc:
                logger.exception("LLM generation failed")
                if _is_gpu_resource_error(exc):
                    _llm_unhealthy_error = f"{type(exc).__name__}: {exc}"
                    _schedule_hard_exit(f"GPU resource error: {_llm_unhealthy_error}")
                    raise HTTPException(status_code=503, detail=f"LLM GPU resource error: {exc}")
                raise HTTPException(status_code=500, detail=f"LLM generation failed: {exc}")

    @app.post("/v1/embeddings")
    async def embeddings(request: _FastAPIRequest):
        body = await request.json()
        req = _EmbRequest(**body)
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
    p.add_argument(
        "--disable-large-allocations",
        action="store_true",
        help="Disable GPU_ENABLE_LARGE_ALLOCATIONS for LLM GPU loads",
    )
    p.add_argument(
        "--llm-max-concurrent",
        type=int,
        default=int(os.environ.get("OV_INTEL_LLM_MAX_CONCURRENT", "1")),
        help="Maximum concurrent LLM generate calls inside this process",
    )
    p.add_argument(
        "--llm-reload-every",
        type=int,
        default=int(os.environ.get("OV_INTEL_LLM_RELOAD_EVERY", "0")),
        help="Unload/reload the LLM after N successful chat requests; 0 disables periodic reload",
    )
    p.add_argument(
        "--llm-exit-every",
        type=int,
        default=int(os.environ.get("OV_INTEL_LLM_EXIT_EVERY", "0")),
        help="Hard-exit after N successful chat requests so a supervisor can restart the process; 0 disables",
    )
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
        llm_large_allocations=not args.disable_large_allocations,
        llm_max_concurrent=args.llm_max_concurrent,
        llm_reload_every=max(0, args.llm_reload_every),
        llm_exit_every=max(0, args.llm_exit_every),
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
