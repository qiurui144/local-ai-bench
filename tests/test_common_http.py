"""HTTP client robustness: a 200 response with a non-JSON body must degrade
to ``ok=False`` instead of raising and killing the whole benchmark run.

Real-world trigger: a reverse proxy / gateway in front of the model server
returns an HTML error page (or truncated body) with status 200.
"""
import json

import pytest

import common


class _Model:
    name = "stub"
    hf_repo = "org/stub"
    is_vlm = False
    auth_header = "Bearer EMPTY"
    provider = "local_vllm"  # needed by _post_with_retry (added in v0.5)

    @property
    def base_url(self):
        return "http://localhost:9999/v1"

    @property
    def effective_model_id(self):
        return self.hf_repo


class _BadJSONResponse:
    """200 response whose body is not JSON (e.g. proxy HTML error page)."""
    status_code = 200
    text = "<html>gateway error</html>"

    def json(self):
        raise json.JSONDecodeError("Expecting value", self.text, 0)


def _patch_post(monkeypatch):
    monkeypatch.setattr(common.httpx, "post", lambda url, **kw: _BadJSONResponse())


def test_infer_sync_bad_json_returns_err(monkeypatch):
    _patch_post(monkeypatch)
    res = common.infer_sync(_Model(), prompt="hi")
    assert not res.ok
    assert "json" in res.error.lower()


def test_infer_embedding_bad_json_returns_err(monkeypatch):
    _patch_post(monkeypatch)
    res = common.infer_embedding(_Model(), "hello")
    assert not res.ok
    assert "json" in res.error.lower()


def test_infer_rerank_bad_json_returns_err(monkeypatch):
    _patch_post(monkeypatch)
    res = common.infer_rerank(_Model(), "q", ["d0", "d1"])
    assert not res.ok
    assert "json" in res.error.lower()


@pytest.mark.asyncio
async def test_infer_async_bad_json_returns_err():
    class _Client:
        async def post(self, url, **kw):
            return _BadJSONResponse()

    res = await common.infer_async(_Client(), _Model(), prompt="hi")
    assert not res.ok
    assert "json" in res.error.lower()


class _OkResp:
    status_code = 200

    def __init__(self, content: str):
        self._content = content

    def json(self):
        return {
            "choices": [{"message": {"content": self._content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }


def test_strip_think_tags_basic():
    assert common._strip_think_tags("<think>reasoning</think>answer") == "answer"


def test_strip_think_tags_multiline():
    text = "<think>\nstep 1\nstep 2\n</think>\n\n你好，世界"
    assert common._strip_think_tags(text) == "你好，世界"


def test_strip_think_tags_empty_after_strip():
    text = "<think>only thinking, no answer</think>"
    assert common._strip_think_tags(text) == ""


def test_strip_think_tags_passthrough():
    assert common._strip_think_tags("no think tags here") == "no think tags here"


def test_infer_sync_strips_think_from_content(monkeypatch):
    """infer_sync must strip <think>...</think> before returning content."""
    raw = "<think>I should translate this</think>\n\n你好，世界"

    def fake_post(url, json=None, **kw):
        return _OkResp(raw)

    monkeypatch.setattr(common.httpx, "post", fake_post)
    res = common.infer_sync(_Model(), prompt="translate: Hello world")
    assert res.ok
    assert res.content == "你好，世界"
    assert "<think>" not in res.content


def test_infer_sync_passes_seed_into_payload(monkeypatch):
    """seed 显式给定时进 payload(OpenAI 兼容字段);缺省不带 seed 键。"""
    captured = {}

    class _OkResp:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def fake_post(url, json=None, **kw):
        captured.update(json or {})
        return _OkResp()

    monkeypatch.setattr(common.httpx, "post", fake_post)

    common.infer_sync(_Model(), prompt="hi", seed=42)
    assert captured.get("seed") == 42

    captured.clear()
    common.infer_sync(_Model(), prompt="hi")
    assert "seed" not in captured
