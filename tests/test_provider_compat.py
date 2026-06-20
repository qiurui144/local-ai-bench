"""Provider compatibility tests — llama_cpp routing, 429 retry, wait_for_server.

All tests are fully offline (mock httpx / time.sleep).  No real endpoints are
contacted, no GPU is required.
"""

from __future__ import annotations

import json

from common import ModelConfig, infer_sync, wait_for_server


# ─────────────────────────────────────────────────────────────────────────────
# llama_cpp base_url routing
# ─────────────────────────────────────────────────────────────────────────────


def test_llama_cpp_base_url_from_port():
    """llama_cpp provider uses the port field."""
    cfg = ModelConfig(name="test", provider="llama_cpp", port=8080)
    assert cfg.base_url == "http://localhost:8080/v1"


def test_llama_cpp_base_url_default_port():
    """llama_cpp with port=0 falls back to the default port 8080."""
    cfg = ModelConfig(name="test", provider="llama_cpp", port=0)
    assert cfg.base_url == "http://localhost:8080/v1"


def test_llama_cpp_custom_port():
    """llama_cpp with an explicit non-zero port uses that port."""
    cfg = ModelConfig(name="test", provider="llama_cpp", port=9999)
    assert cfg.base_url == "http://localhost:9999/v1"


def test_llama_cpp_base_url_override_wins():
    """base_url_override takes priority over provider routing."""
    cfg = ModelConfig(
        name="test",
        provider="llama_cpp",
        base_url_override="http://192.168.1.10:8888/v1",
    )
    assert cfg.base_url == "http://192.168.1.10:8888/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Cloud provider base_url sanity (not duplicated from test_provider_auth.py)
# ─────────────────────────────────────────────────────────────────────────────


def test_openai_base_url():
    cfg = ModelConfig(name="test", provider="openai")
    assert cfg.base_url == "https://api.openai.com/v1"


def test_deepseek_base_url():
    cfg = ModelConfig(name="test", provider="deepseek")
    assert "deepseek.com" in cfg.base_url


# ─────────────────────────────────────────────────────────────────────────────
# 429 retry logic
# ─────────────────────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status_code = status
        self.text = body

    def json(self) -> dict:
        return json.loads(self.text)


_OK_BODY = json.dumps({
    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
})

_429_BODY = json.dumps({"error": "rate limited"})


def test_429_retry_cloud(monkeypatch):
    """Cloud provider 429 triggers exponential back-off retry; succeeds on 3rd try."""
    call_count = [0]

    def mock_post(url, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            return _FakeResponse(429, _429_BODY)
        return _FakeResponse(200, _OK_BODY)

    monkeypatch.setattr("httpx.post", mock_post)
    monkeypatch.setattr("time.sleep", lambda s: None)  # do not actually sleep

    cfg = ModelConfig(name="test-openai", provider="openai", api_key_env="TEST_KEY")
    monkeypatch.setenv("TEST_KEY", "sk-test")

    result = infer_sync(cfg, prompt="test")

    assert result.ok, f"Expected ok=True, got error={result.error!r}"
    assert call_count[0] == 3, f"Expected 3 attempts, got {call_count[0]}"


def test_429_retry_deepseek(monkeypatch):
    """DeepSeek (also a cloud provider) retries on 429."""
    call_count = [0]

    def mock_post(url, **kwargs):
        call_count[0] += 1
        if call_count[0] < 2:
            return _FakeResponse(429, _429_BODY)
        return _FakeResponse(200, _OK_BODY)

    monkeypatch.setattr("httpx.post", mock_post)
    monkeypatch.setattr("time.sleep", lambda s: None)

    cfg = ModelConfig(name="test-ds", provider="deepseek", api_key_env="DS_KEY")
    monkeypatch.setenv("DS_KEY", "sk-ds-test")

    result = infer_sync(cfg, prompt="test")
    assert result.ok
    assert call_count[0] == 2


def test_429_no_retry_local(monkeypatch):
    """Local provider (local_vllm) does NOT retry on 429 — fails immediately."""
    call_count = [0]

    def mock_post(url, **kwargs):
        call_count[0] += 1
        return _FakeResponse(429, _429_BODY)

    monkeypatch.setattr("httpx.post", mock_post)

    cfg = ModelConfig(name="test-local", provider="local_vllm", port=8000)
    result = infer_sync(cfg, prompt="test")

    assert not result.ok
    assert call_count[0] == 1, f"Expected 1 attempt, got {call_count[0]}"


def test_429_no_retry_llama_cpp(monkeypatch):
    """llama_cpp (local) does NOT retry on 429."""
    call_count = [0]

    def mock_post(url, **kwargs):
        call_count[0] += 1
        return _FakeResponse(429, _429_BODY)

    monkeypatch.setattr("httpx.post", mock_post)

    cfg = ModelConfig(name="test-llama", provider="llama_cpp", port=8080)
    result = infer_sync(cfg, prompt="test")

    assert not result.ok
    assert call_count[0] == 1


def test_429_exhaust_cloud_retries(monkeypatch):
    """After 3 attempts all returning 429, infer_sync returns ok=False."""
    call_count = [0]

    def mock_post(url, **kwargs):
        call_count[0] += 1
        return _FakeResponse(429, _429_BODY)

    monkeypatch.setattr("httpx.post", mock_post)
    monkeypatch.setattr("time.sleep", lambda s: None)

    cfg = ModelConfig(name="test-cloud", provider="openai", api_key_env="OAI_KEY")
    monkeypatch.setenv("OAI_KEY", "sk-x")

    result = infer_sync(cfg, prompt="test")
    assert not result.ok
    assert call_count[0] == 3


# ─────────────────────────────────────────────────────────────────────────────
# wait_for_server alias
# ─────────────────────────────────────────────────────────────────────────────


def test_wait_for_server_skips_cloud():
    """Cloud provider wait_for_server returns True immediately (no HTTP call)."""
    cfg = ModelConfig(name="test", provider="openai")
    # If this actually sent HTTP it would error (no monkeypatch); silence = skipped.
    result = wait_for_server(cfg, timeout_s=1.0)
    assert result is True


def test_wait_for_server_skips_deepseek():
    cfg = ModelConfig(name="test", provider="deepseek")
    result = wait_for_server(cfg, timeout_s=1.0)
    assert result is True


def test_wait_for_server_skips_dashscope():
    cfg = ModelConfig(name="test", provider="dashscope")
    result = wait_for_server(cfg, timeout_s=1.0)
    assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# probe_provider.py importability
# ─────────────────────────────────────────────────────────────────────────────


def test_probe_script_importable():
    """probe_provider.py can be imported and exposes probe_model."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "probe_provider",
        Path(__file__).resolve().parent.parent / "scripts" / "probe_provider.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "probe_model"), "probe_model not found in probe_provider"
    assert callable(mod.probe_model)
