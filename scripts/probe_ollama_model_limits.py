#!/usr/bin/env python3
"""Probe Ollama local-model concurrency and long-context limits.

The script intentionally talks to Ollama's native /api/chat endpoint so it can
request num_ctx per step and capture Ollama timing counters from the final
streaming event.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ordered[int(k)])
    return float(ordered[f] * (c - k) + ordered[c] * (k - f))


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 95), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
    }


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _ollama_base(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url[:-3]
    return url


def chat_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    num_ctx: int,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
            "temperature": 0,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_ollama_base(base_url)}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_token_s: float | None = None
    chunks: list[str] = []
    final: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line.decode("utf-8", errors="replace"))
                msg = event.get("message") or {}
                content = msg.get("content") or ""
                if content and first_token_s is None:
                    first_token_s = time.perf_counter()
                if content:
                    chunks.append(content)
                if event.get("done"):
                    final = event
                    break
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        return {"ok": False, "error": f"HTTP {exc.code}: {body}", "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

    t1 = time.perf_counter()
    eval_count = int(final.get("eval_count") or 0)
    prompt_eval_count = int(final.get("prompt_eval_count") or 0)
    eval_duration_ns = int(final.get("eval_duration") or 0)
    prompt_eval_duration_ns = int(final.get("prompt_eval_duration") or 0)
    return {
        "ok": True,
        "latency_ms": round((t1 - t0) * 1000, 3),
        "ttft_ms": round(((first_token_s or t1) - t0) * 1000, 3),
        "output": "".join(chunks),
        "output_chars": sum(len(c) for c in chunks),
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "eval_duration_ns": eval_duration_ns,
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "decode_tps": (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0,
        "prefill_tps": (prompt_eval_count / (prompt_eval_duration_ns / 1e9)) if prompt_eval_duration_ns > 0 else 0.0,
        "done_reason": final.get("done_reason"),
    }


def run_concurrency(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt = (
        "Answer in one short paragraph. Compare local CPU, GPU, and NPU inference "
        "for edge AI deployment, focusing on latency and throughput."
    )
    steps: list[dict[str, Any]] = []
    for concurrency in args.concurrency_levels:
        deadline = time.perf_counter() + args.duration_s
        results: list[dict[str, Any]] = []

        def worker() -> None:
            while time.perf_counter() < deadline:
                results.append(chat_once(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    num_ctx=args.short_num_ctx,
                    max_tokens=args.max_tokens,
                    timeout_s=args.request_timeout_s,
                ))

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(worker) for _ in range(concurrency)]
            concurrent.futures.wait(futs)
        elapsed_s = max(time.perf_counter() - started, 1e-9)

        ok = [r for r in results if r.get("ok")]
        errors = [r for r in results if not r.get("ok")]
        output_tokens = sum(int(r.get("eval_count") or 0) for r in ok)
        steps.append({
            "concurrency": concurrency,
            "target_duration_s": args.duration_s,
            "elapsed_s": round(elapsed_s, 3),
            "requests": len(results),
            "success": len(ok),
            "errors": len(errors),
            "success_rate": round(len(ok) / len(results), 4) if results else 0.0,
            "aggregate_decode_tps": round(output_tokens / elapsed_s, 3),
            "latency_ms": _stats([float(r["latency_ms"]) for r in ok]),
            "ttft_ms": _stats([float(r["ttft_ms"]) for r in ok]),
            "decode_tps_per_request": _stats([float(r.get("decode_tps") or 0.0) for r in ok]),
            "first_error": errors[0].get("error") if errors else None,
        })
        print(f"concurrency={concurrency} success={len(ok)}/{len(results)} tps={steps[-1]['aggregate_decode_tps']}", flush=True)
    return steps


def _context_prompt(target_tokens: int) -> tuple[str, str]:
    needle = f"LIMIT_NEEDLE_{target_tokens}_ZXQ"
    filler_unit = (
        "local inference capacity measurement requires stable latency throughput "
        "memory bandwidth scheduling cache behavior and deterministic validation "
    )
    filler_words = filler_unit.split()
    target_words = max(128, target_tokens - 96)
    half = target_words // 2
    before = " ".join(filler_words[i % len(filler_words)] for i in range(half))
    after = " ".join(filler_words[(i + 3) % len(filler_words)] for i in range(target_words - half))
    prompt = (
        f"You must find and repeat one exact marker from a long context.\n"
        f"Context begins:\n{before}\nMARKER: {needle}\n{after}\nContext ends.\n"
        f"Question: output only the exact marker string."
    )
    return prompt, needle


def run_context(args: argparse.Namespace) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for target in args.context_levels:
        prompt, needle = _context_prompt(target)
        requested_ctx = max(args.short_num_ctx, target + args.max_tokens + args.context_margin)
        result = chat_once(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            num_ctx=requested_ctx,
            max_tokens=args.max_tokens,
            timeout_s=args.context_timeout_s,
        )
        row: dict[str, Any] = {
            "target_prompt_tokens_approx": target,
            "requested_num_ctx": requested_ctx,
            "ok": bool(result.get("ok")),
            "latency_ms": result.get("latency_ms"),
            "ttft_ms": result.get("ttft_ms"),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
            "prefill_tps": round(float(result.get("prefill_tps") or 0.0), 3),
            "decode_tps": round(float(result.get("decode_tps") or 0.0), 3),
            "needle_recalled": needle in (result.get("output") or ""),
            "output_preview": (result.get("output") or "")[:160],
            "error": result.get("error"),
        }
        steps.append(row)
        print(
            f"context~{target} ok={row['ok']} recalled={row['needle_recalled']} "
            f"latency_ms={row['latency_ms']}",
            flush=True,
        )
    return steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--concurrency-levels", type=_parse_int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--context-levels", type=_parse_int_list, default=[1024, 4096, 8192, 16384, 32768])
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--short-num-ctx", type=int, default=2048)
    parser.add_argument("--context-margin", type=int, default=512)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--context-timeout-s", type=float, default=420.0)
    args = parser.parse_args()

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "benchmark": "ollama_model_limits",
        "started_at": started,
        "host": os.environ.get("COMPUTERNAME") or os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
        "base_url": args.base_url,
        "model": args.model,
        "settings": {
            "concurrency_levels": args.concurrency_levels,
            "context_levels": args.context_levels,
            "duration_s": args.duration_s,
            "max_tokens": args.max_tokens,
            "short_num_ctx": args.short_num_ctx,
            "context_margin": args.context_margin,
        },
        "concurrency": run_concurrency(args),
        "context": run_context(args),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    out = Path(args.out) if args.out else Path("output/reports") / f"ollama_model_limits_{args.model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
