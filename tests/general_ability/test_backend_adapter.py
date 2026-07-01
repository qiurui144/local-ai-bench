"""Adapter contract: harness ModelConfig → llama_benchmark backend protocol.

llama_benchmark 的 llm runners 需要 backend.generate(prompt, max_tokens,
temperature) -> str 与 backend.generate_with_logprobs(prompt, candidates,
max_tokens) -> dict[str, float]。协议漂移由本测试拦截(spec §11 风险 2)。
"""
import httpx
import pytest

from benchmark.general_ability.backend_adapter import make_backend


class _Cfg:
    name = "m1"
    hf_repo = "org/m1"
    port = 8123

    @property
    def base_url(self):
        return f"http://localhost:{self.port}/v1"


def _chat_response(content="#### 42", logprobs=None):
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    if logprobs is not None:
        body["choices"][0]["logprobs"] = {"content": [{"token": "B", "logprob": -0.1,
            "top_logprobs": [{"token": k, "logprob": v} for k, v in logprobs.items()]}]}
    return httpx.Response(200, json=body, request=httpx.Request("POST", "http://x"))


def test_generate_returns_text(monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw["json"]
        captured["timeout"] = kw["timeout"]
        return _chat_response("hello")

    monkeypatch.setattr(httpx, "post", fake_post)
    be = make_backend(_Cfg())
    assert be.generate("Q", max_tokens=8, temperature=0.0) == "hello"
    assert captured["url"] == "http://localhost:8123/v1/chat/completions"
    assert captured["json"]["model"] == "org/m1"
    assert captured["timeout"] == 600.0


def test_generate_with_logprobs_maps_candidates(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **kw: _chat_response(
        "B", logprobs={"A": -3.0, "B": -0.1, "C": -5.0}))
    be = make_backend(_Cfg())
    lp = be.generate_with_logprobs("Q", ["A", "B", "C", "D"], max_tokens=1)
    assert lp["B"] == -0.1 and lp["D"] == float("-inf")


def test_generate_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(
        500, request=httpx.Request("POST", "http://x")))
    be = make_backend(_Cfg())
    with pytest.raises(Exception):
        be.generate("Q")


def test_backend_timeout_can_be_overridden(monkeypatch):
    captured = {}

    class _TimeoutCfg(_Cfg):
        benchmarks = {"general_ability": {"timeout_s": 900}}

    def fake_post(url, **kw):
        captured["timeout"] = kw["timeout"]
        return _chat_response("hello")

    monkeypatch.setattr(httpx, "post", fake_post)
    be = make_backend(_TimeoutCfg())

    assert be.generate("Q") == "hello"
    assert captured["timeout"] == 900.0


def test_generate_retries_transient_503(monkeypatch):
    calls = []
    sleeps = []

    class _RetryCfg(_Cfg):
        benchmarks = {"general_ability": {"retry_attempts": 2, "retry_initial_s": 0.1}}

    def fake_post(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return httpx.Response(503, request=httpx.Request("POST", url))
        return _chat_response("ok")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("benchmark.llama_benchmark.backends.openai_compatible_backend.time.sleep", sleeps.append)
    be = make_backend(_RetryCfg())

    assert be.generate("Q") == "ok"
    assert len(calls) == 2
    assert sleeps == [0.1]


def test_generate_retries_transport_error(monkeypatch):
    calls = []

    class _RetryCfg(_Cfg):
        benchmarks = {"general_ability": {"retry_attempts": 2, "retry_initial_s": 0}}

    def fake_post(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            raise httpx.ConnectError("service restarting", request=httpx.Request("POST", url))
        return _chat_response("ok")

    monkeypatch.setattr(httpx, "post", fake_post)
    be = make_backend(_RetryCfg())

    assert be.generate("Q") == "ok"
    assert len(calls) == 2
