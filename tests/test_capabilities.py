"""Typed positive capabilities on ModelConfig (arch review P0.3 / F3+F4)."""
from pathlib import Path

import run_benchmark as rb
from common import ModelConfig, _is_chat_capable, load_models

MODELS_YAML = Path(rb.__file__).parent / "models.yaml"


def test_load_models_derives_capabilities():
    by_name = {m.name: m for m in load_models(MODELS_YAML)}
    assert "chat" in by_name["qwen3-vl-8b-instruct"].capabilities
    assert "translation" in by_name["qwen3-30b-a3b-instruct-2507-fp8"].capabilities
    assert by_name["qwen3-embedding-0.6b"].capabilities == ("embedding",)
    assert "chat" not in by_name["bge-reranker-v2-m3"].capabilities      # rerank_native
    assert "chat" not in by_name["sensevoice-small"].capabilities        # asr
    assert "rerank" in by_name["qwen3-reranker-4b"].capabilities


def test_model_hint_uses_typed_capabilities_without_yaml_reread(monkeypatch):
    m = next(x for x in load_models(MODELS_YAML) if x.name == "qwen3-embedding-0.6b")
    import yaml
    monkeypatch.setattr(yaml, "safe_load",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("yaml re-read")))
    assert rb._model_hint(m, "embedding_capable") is True
    assert rb._model_hint(m, "asr_capable") is False
    assert rb._is_chat_capable(m) is False


def test_stub_without_capabilities_falls_back_to_hint_seam():
    class _Stub:
        name = "nope"
        rerank_native = False
    assert rb._is_chat_capable(_Stub()) is True   # 未知模型默认 chat(现行为)


# ── rerank_native gate regression (task-2 bug fix) ──────────────────────────

def make_model(**kwargs):
    defaults = dict(name="m", port=0, provider="ollama", capabilities=frozenset())
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def test_rerank_native_not_chat_capable():
    m = make_model(capabilities=frozenset({"rerank", "rerank_native"}))
    assert not _is_chat_capable(m)


def test_embedding_not_chat_capable():
    m = make_model(capabilities=frozenset({"embedding"}))
    assert not _is_chat_capable(m)


def test_ocr_not_chat_capable():
    m = make_model(capabilities=frozenset({"ocr"}))
    assert not _is_chat_capable(m)


def test_pure_chat_is_chat_capable():
    m = make_model(capabilities=frozenset())
    assert _is_chat_capable(m)


def test_vlm_is_chat_capable():
    m = make_model(capabilities=frozenset({"vlm"}))
    assert _is_chat_capable(m)
