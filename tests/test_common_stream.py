"""Coverage backfill for common.InferResult.looks_truncated and common.infer_stream.

- looks_truncated is the truncation predicate feeding the token-budget dimension:
  finish_reason=="length" short-circuits; otherwise output_tokens >= 0.95*max_tokens
  with a max_tokens_requested>0 guard.
- infer_stream is exercised with a fake httpx.stream context manager yielding
  controlled SSE lines, and a fake clock injected via monkeypatch on common.time
  so TTFT / latency are deterministic.
"""
import pytest

import common


class _Model:
    name = "stub"
    hf_repo = "org/stub"
    is_vlm = False
    auth_header = "Bearer EMPTY"

    @property
    def base_url(self):
        return "http://localhost:9999/v1"

    @property
    def effective_model_id(self):
        return self.hf_repo


# ────────────────────────────────────────────────────────────
# looks_truncated boundaries
# ────────────────────────────────────────────────────────────


def _res(finish_reason="", output_tokens=0):
    return common.InferResult(model="stub", finish_reason=finish_reason,
                              output_tokens=output_tokens)


def test_truncated_finish_reason_length_regardless_of_counts():
    assert _res("length", output_tokens=0).looks_truncated(800)
    assert _res("length", output_tokens=1).looks_truncated(0)  # even with the guard off


@pytest.mark.parametrize("max_tokens,at,below", [
    (800, 760, 759),   # 0.95*800 = 760 exactly
    (10, 10, 9),       # 0.95*10 = 9.5 → first int >= is ceil = 10
])
def test_truncated_095_boundary(max_tokens, at, below):
    assert _res("stop", output_tokens=at).looks_truncated(max_tokens)
    assert not _res("stop", output_tokens=below).looks_truncated(max_tokens)


def test_truncated_max_tokens_zero_guard():
    # guard: max_tokens_requested == 0 must not flag (would be 0 >= 0 otherwise)
    assert not _res("stop", output_tokens=500).looks_truncated(0)


def test_truncated_normal_stop_false():
    assert not _res("stop", output_tokens=100).looks_truncated(800)


# ────────────────────────────────────────────────────────────
# infer_stream — fake SSE stream + fake clock
# ────────────────────────────────────────────────────────────


class _FakeClock:
    """Deterministic time.monotonic: each call advances 10 ms."""

    def __init__(self, step_s=0.010):
        self._now = 0.0
        self._step = step_s

    def monotonic(self):
        t = self._now
        self._now += self._step
        return t


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)


class _FakeStreamCM:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, *exc):
        return False


def _patch_stream(monkeypatch, lines, status_code=200):
    resp = _FakeStreamResponse(lines, status_code)
    monkeypatch.setattr(common.httpx, "stream",
                        lambda method, url, **kw: _FakeStreamCM(resp))
    monkeypatch.setattr(common, "time", _FakeClock())


def test_infer_stream_happy_path(monkeypatch):
    _patch_stream(monkeypatch, [
        "",                                                       # keep-alive blank
        ": comment",                                              # non-data SSE line
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',  # no content → no TTFT
        'data: {"choices": [{"delta": {"content": "{\\"summ"}}]}',
        'data: {"choices": [{"delta": {"content": "ary\\": \\"x\\"}"}}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}}',
        "data: [DONE]",
        'data: {"choices": [{"delta": {"content": "AFTER-DONE"}}]}',  # must be ignored
    ])
    res = common.infer_stream(_Model(), prompt="hi")
    assert res.ok
    assert res.content == '{"summary": "x"}'
    assert res.parsed_json == {"summary": "x"}
    # clock calls: t0=0ms → first content token=10ms → elapsed=20ms.
    # Role-only chunk before it must NOT capture TTFT; second content
    # chunk must not move it (single intermediate monotonic call).
    assert res.ttft_ms == pytest.approx(10.0)
    assert res.latency_ms == pytest.approx(20.0)
    assert res.input_tokens == 12
    assert res.output_tokens == 7
    assert res.finish_reason == "stop"
    assert res.tokens_per_sec == pytest.approx(7 / 0.020)


def test_infer_stream_malformed_chunk_skipped(monkeypatch):
    _patch_stream(monkeypatch, [
        "data: {not json at all",                            # skipped, no clock call
        'data: {"choices": [{"delta": {"content": "ok"}}]}',
        "data: [DONE]",
    ])
    res = common.infer_stream(_Model(), prompt="hi")
    assert res.ok
    assert res.content == "ok"
    assert res.ttft_ms == pytest.approx(10.0)  # malformed line didn't consume the clock


def test_infer_stream_regex_json_fallback(monkeypatch):
    _patch_stream(monkeypatch, [
        'data: {"choices": [{"delta": {"content": "noise {\\"k\\": 2} tail"}}]}',
        "data: [DONE]",
    ])
    res = common.infer_stream(_Model(), prompt="hi")
    assert res.ok
    assert res.parsed_json == {"k": 2}


def test_infer_stream_no_content_tokens(monkeypatch):
    _patch_stream(monkeypatch, ["data: [DONE]"])
    res = common.infer_stream(_Model(), prompt="hi")
    assert res.ok
    assert res.content == ""
    assert res.ttft_ms == 0.0
    assert res.parsed_json is None
    assert res.tokens_per_sec == 0.0


def test_infer_stream_non_200(monkeypatch):
    _patch_stream(monkeypatch, [], status_code=429)
    res = common.infer_stream(_Model(), prompt="hi")
    assert not res.ok
    assert "HTTP 429" in res.error


def test_infer_stream_seed_in_payload(monkeypatch):
    """seed= 进请求 payload(缓存 A/B 一致性校验需确定性采样);默认不带 key。"""
    captured = {}

    def fake_stream(method, url, **kw):
        captured.clear()
        captured.update(kw)
        return _FakeStreamCM(_FakeStreamResponse(["data: [DONE]"]))

    monkeypatch.setattr(common.httpx, "stream", fake_stream)
    monkeypatch.setattr(common, "time", _FakeClock())
    common.infer_stream(_Model(), prompt="hi", seed=0)
    assert captured["json"]["seed"] == 0
    common.infer_stream(_Model(), prompt="hi")
    assert "seed" not in captured["json"]


def test_infer_stream_ollama_think_false_payload_and_token_floor(monkeypatch):
    captured = {}

    def fake_stream(method, url, **kw):
        captured.clear()
        captured.update(kw)
        return _FakeStreamCM(_FakeStreamResponse(["data: [DONE]"]))

    monkeypatch.setattr(common.httpx, "stream", fake_stream)
    monkeypatch.setattr(common, "time", _FakeClock())

    model = _Model()
    model.ollama_think = False
    common.infer_stream(model, prompt="hi", max_tokens=64)

    payload = captured["json"]
    assert payload["think"] is False
    assert payload["options"]["think"] is False
    assert payload["max_tokens"] == 2048


def test_infer_stream_transport_exception(monkeypatch):
    def boom(method, url, **kw):
        raise common.httpx.ConnectError("connection refused")

    monkeypatch.setattr(common.httpx, "stream", boom)
    monkeypatch.setattr(common, "time", _FakeClock())
    res = common.infer_stream(_Model(), prompt="hi")
    assert not res.ok
    assert "ConnectError" in res.error
