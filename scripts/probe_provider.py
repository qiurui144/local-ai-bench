"""Provider capability probe — smoke-test any configured model endpoint.

Usage:
    python3 scripts/probe_provider.py --model <model-name>
    python3 scripts/probe_provider.py --model <model-name> --models-yaml models.yaml

For each check, prints ✓ / ✗ / ? / N/A and a one-line summary.
Exit codes:
  0  all checks passed (READY)
  1  at least one check failed (DEGRADED)
  2  model not found in models.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running from the repo root without pip-installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402  (after sys.path tweak)
from common import (  # noqa: E402
    ModelConfig,
    _CLOUD_PROVIDERS,
    _apply_ollama_think_controls,
    load_models,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny 1×1 PNG for VLM image tests (base64-encoded, no external file needed)
# ─────────────────────────────────────────────────────────────────────────────

_1X1_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_1X1_PNG_DATA_URL = f"data:image/png;base64,{_1X1_PNG_B64}"

_PROBE_TIMEOUT = 15.0  # seconds for each individual request


def _prepare_probe_payload(cfg: ModelConfig, payload: dict) -> dict:
    _apply_ollama_think_controls(payload, cfg, int(payload.get("max_tokens") or 0))
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Individual probe functions — each returns (symbol, detail_line)
# ─────────────────────────────────────────────────────────────────────────────


def probe_endpoint_reachable(cfg: ModelConfig) -> tuple[str, str]:
    """GET /models — only attempted for local providers."""
    if cfg.provider in _CLOUD_PROVIDERS:
        return "?", "Endpoint reachable (skipped — cloud provider, auth via API key)"
    try:
        t0 = time.monotonic()
        r = httpx.get(
            f"{cfg.base_url}/models",
            timeout=5.0,
            headers={"Authorization": cfg.auth_header},
        )
        elapsed = time.monotonic() - t0
        if r.status_code == 200:
            return "✓", f"Endpoint reachable ({elapsed:.2f}s)"
        return "✗", f"Endpoint returned HTTP {r.status_code} ({elapsed:.2f}s)"
    except Exception as e:
        return "✗", f"Endpoint unreachable: {type(e).__name__}: {e}"


def probe_chat_completion(cfg: ModelConfig) -> tuple[str, str, float, int]:
    """POST /chat/completions with a trivial prompt."""
    api_key_warn = ""
    if cfg.api_key_env and not os.environ.get(cfg.api_key_env):
        api_key_warn = f"  [WARN: {cfg.api_key_env} not set — auth may fail]"

    payload = {
        "model": cfg.effective_model_id,
        "messages": [{"role": "user", "content": "Reply with 'OK' only."}],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    _prepare_probe_payload(cfg, payload)
    url = f"{cfg.base_url}/chat/completions"
    try:
        t0 = time.monotonic()
        r = httpx.post(
            url,
            json=payload,
            timeout=_PROBE_TIMEOUT,
            headers={"Authorization": cfg.auth_header},
        )
        elapsed = time.monotonic() - t0
        if r.status_code != 200:
            return "✗", f"Chat completion HTTP {r.status_code}: {r.text[:120]}{api_key_warn}", elapsed, 0
        data = r.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        suffix = f"[requires {cfg.api_key_env}]" if cfg.api_key_env else ""
        note = f" {suffix}".rstrip()
        return "✓", f"Chat completion works ({elapsed:.2f}s, {tokens} tokens){note}{api_key_warn}", elapsed, tokens
    except Exception as e:
        return "✗", f"Chat completion error: {type(e).__name__}: {e}", 0.0, 0


def probe_json_mode(cfg: ModelConfig) -> tuple[str, str]:
    """POST /chat/completions with response_format json_object."""
    payload = {
        "model": cfg.effective_model_id,
        "messages": [
            {"role": "user", "content": 'Reply ONLY with valid JSON: {"ok": true}'},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    _prepare_probe_payload(cfg, payload)
    url = f"{cfg.base_url}/chat/completions"
    try:
        r = httpx.post(
            url,
            json=payload,
            timeout=_PROBE_TIMEOUT,
            headers={"Authorization": cfg.auth_header},
        )
        if r.status_code != 200:
            return "✗", f"JSON mode HTTP {r.status_code}: {r.text[:80]}"
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            json.loads(content)
            return "✓", "JSON mode: valid JSON returned"
        except json.JSONDecodeError:
            return "✗", f"JSON mode: response not valid JSON — got: {content[:80]!r}"
    except Exception as e:
        return "✗", f"JSON mode error: {type(e).__name__}: {e}"


def probe_seed_consistency(cfg: ModelConfig) -> tuple[str, str]:
    """Two calls with seed=42; outputs should match for deterministic backends."""
    if cfg.provider in _CLOUD_PROVIDERS:
        return "N/A", "Seed consistency (cloud provider, seed behavior not guaranteed)"

    def _call() -> Optional[str]:
        payload = {
            "model": cfg.effective_model_id,
            "messages": [{"role": "user", "content": "Say exactly: hello world"}],
            "max_tokens": 20,
            "temperature": 0.0,
            "seed": 42,
        }
        _prepare_probe_payload(cfg, payload)
        try:
            r = httpx.post(
                f"{cfg.base_url}/chat/completions",
                json=payload,
                timeout=_PROBE_TIMEOUT,
                headers={"Authorization": cfg.auth_header},
            )
            if r.status_code != 200:
                return None
            return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            return None

    a, b = _call(), _call()
    if a is None or b is None:
        return "?", "Seed consistency: call failed (endpoint unreachable?)"
    if a == b:
        return "✓", "Seed consistency: outputs match (seed=42)"
    return "?", f"Seed consistency: outputs differ — may be non-deterministic\n    a={a[:60]!r}\n    b={b[:60]!r}"


def probe_vlm(cfg: ModelConfig) -> tuple[str, str]:
    """POST /chat/completions with a 1×1 PNG — only for VLM-capable models."""
    if not cfg.is_vlm:
        return "?", "VLM: not tested (task_type=text_only)"

    payload = {
        "model": cfg.effective_model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _1X1_PNG_DATA_URL}},
                    {"type": "text", "text": "What colour is this pixel? One word."},
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    _prepare_probe_payload(cfg, payload)
    url = f"{cfg.base_url}/chat/completions"
    try:
        r = httpx.post(
            url,
            json=payload,
            timeout=_PROBE_TIMEOUT,
            headers={"Authorization": cfg.auth_header},
        )
        if r.status_code != 200:
            return "✗", f"VLM: HTTP {r.status_code}: {r.text[:80]}"
        return "✓", "VLM: image accepted (1×1 PNG test)"
    except Exception as e:
        return "✗", f"VLM error: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Top-level probe_model — runs all checks and prints a report
# ─────────────────────────────────────────────────────────────────────────────


def probe_model(cfg: ModelConfig) -> int:
    """Run all probe checks for *cfg*.  Returns 0 (pass) or 1 (degraded)."""
    # Warn early if API key missing
    if cfg.api_key_env and not os.environ.get(cfg.api_key_env):
        print(f"  WARN: {cfg.api_key_env} not set — auth may fail")

    failures: list[str] = []

    # 1. Endpoint reachability
    sym, msg = probe_endpoint_reachable(cfg)
    print(f"  {sym} {msg}")
    if sym == "✗":
        failures.append(msg)

    # 2. Chat completion
    sym, msg, latency_s, tokens = probe_chat_completion(cfg)
    print(f"  {sym} {msg}")
    if sym == "✗":
        failures.append(msg)

    # 3. JSON mode
    sym, msg = probe_json_mode(cfg)
    print(f"  {sym} {msg}")
    if sym == "✗":
        failures.append(msg)

    # 4. Seed consistency
    sym, msg = probe_seed_consistency(cfg)
    print(f"  {sym} {msg}")
    if sym == "✗":
        failures.append(msg)

    # 5. VLM capability
    sym, msg = probe_vlm(cfg)
    print(f"  {sym} {msg}")
    if sym == "✗":
        failures.append(msg)

    # Summary
    if not failures:
        print("  READY — all checks passed")
        return 0
    print(f"  DEGRADED — {len(failures)} check(s) failed")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a configured model provider for capability and health."
    )
    parser.add_argument("--model", required=True, help="Model name (as in models.yaml)")
    parser.add_argument(
        "--models-yaml",
        default=str(_REPO_ROOT / "models.yaml"),
        help="Path to models.yaml (default: <repo-root>/models.yaml)",
    )
    args = parser.parse_args()

    models = load_models(args.models_yaml)
    by_name = {m.name: m for m in models}

    if args.model not in by_name:
        print(
            f"ERROR: model '{args.model}' not found in {args.models_yaml}.\n"
            f"Available: {', '.join(sorted(by_name)) or '(none)'}",
            file=sys.stderr,
        )
        sys.exit(2)

    cfg = by_name[args.model]
    print(
        f"Probing model: {cfg.name} "
        f"(provider={cfg.provider} @ {cfg.base_url})"
    )
    rc = probe_model(cfg)
    sys.exit(rc)


if __name__ == "__main__":
    main()
