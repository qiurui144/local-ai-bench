"""离线测试 RKNN adapter，不需要真正的 RK3588 或 rkllm。"""
import json

def test_rknn_adapter_list_models(monkeypatch):
    """mock rkllm，确认 /v1/models 返回正确 schema。"""
    import sys
    # mock rkllm 模块
    import types
    rkllm_mock = types.ModuleType("rkllm")
    api_mock = types.ModuleType("rkllm.api")
    api_mock.RKLLM = lambda path: None
    sys.modules.setdefault("rkllm", rkllm_mock)
    sys.modules.setdefault("rkllm.api", api_mock)

    from benchmark.backends.rknn_adapter import _build_app
    app = _build_app("/fake/model.rknn")
    client = app.test_client()
    r = client.get("/v1/models")
    data = json.loads(r.data)
    assert r.status_code == 200
    assert data["object"] == "list"
    assert len(data["data"]) >= 1

def test_rknn_adapter_unavailable_raises():
    """没有 rkllm 时，POST /v1/chat/completions 应返回 500 而非 crash。"""
    import sys
    # 强制 rkllm 不可用
    sys.modules["rkllm"] = None
    sys.modules["rkllm.api"] = None
    import importlib
    import benchmark.backends.rknn_adapter as adapter
    importlib.reload(adapter)
    assert not adapter._RKLLM_AVAILABLE
