"""Cross-platform provider routing, auth header, and effective model-id.

Tests written BEFORE implementation (RED phase); they will pass once
common.py gains the provider / api_key_env / model_id / base_url_override
fields on ModelConfig.
"""

from common import ModelConfig, load_models


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _mc(**kwargs) -> ModelConfig:
    return ModelConfig(name="test", **kwargs)


# ──────────────────────────────────────────────────────────────
# base_url routing by provider
# ──────────────────────────────────────────────────────────────

class TestBaseUrl:
    def test_local_vllm_uses_port(self):
        m = _mc(port=8001)
        assert m.base_url == "http://localhost:8001/v1"

    def test_ollama_default_port(self):
        m = _mc(provider="ollama")
        assert m.base_url == "http://localhost:11434/v1"

    def test_ollama_custom_port(self):
        m = _mc(provider="ollama", port=11435)
        assert m.base_url == "http://localhost:11435/v1"

    def test_deepseek(self):
        m = _mc(provider="deepseek")
        assert m.base_url == "https://api.deepseek.com/v1"

    def test_dashscope(self):
        m = _mc(provider="dashscope")
        assert m.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_openai(self):
        m = _mc(provider="openai")
        assert m.base_url == "https://api.openai.com/v1"

    def test_override_wins_over_provider(self):
        m = _mc(provider="deepseek",
                base_url_override="https://my-proxy.example.com/v1")
        assert m.base_url == "https://my-proxy.example.com/v1"

    def test_override_trailing_slash_stripped(self):
        m = _mc(base_url_override="https://proxy.example.com/v1/")
        assert m.base_url == "https://proxy.example.com/v1"

    def test_generic_provider_uses_override(self):
        m = _mc(provider="generic",
                base_url_override="http://192.168.1.10:8080/v1")
        assert m.base_url == "http://192.168.1.10:8080/v1"


# ──────────────────────────────────────────────────────────────
# auth_header
# ──────────────────────────────────────────────────────────────

class TestAuthHeader:
    def test_no_api_key_env_gives_bearer_empty(self):
        m = _mc()
        assert m.auth_header == "Bearer EMPTY"

    def test_api_key_env_set_and_env_populated(self, monkeypatch):
        monkeypatch.setenv("MY_BENCH_KEY", "sk-test-abc123")
        m = _mc(api_key_env="MY_BENCH_KEY")
        assert m.auth_header == "Bearer sk-test-abc123"

    def test_api_key_env_set_but_var_missing(self, monkeypatch):
        monkeypatch.delenv("MY_MISSING_KEY", raising=False)
        m = _mc(api_key_env="MY_MISSING_KEY")
        assert m.auth_header == "Bearer EMPTY"

    def test_api_key_env_set_but_var_empty(self, monkeypatch):
        monkeypatch.setenv("MY_EMPTY_KEY", "")
        m = _mc(api_key_env="MY_EMPTY_KEY")
        assert m.auth_header == "Bearer EMPTY"

    def test_auth_header_format(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-xxxxxxxxxxxx")
        m = _mc(provider="deepseek", api_key_env="DEEPSEEK_API_KEY")
        header = m.auth_header
        assert header.startswith("Bearer ")
        assert header == "Bearer sk-xxxxxxxxxxxx"


# ──────────────────────────────────────────────────────────────
# effective_model_id (what goes in the API payload "model" field)
# ──────────────────────────────────────────────────────────────

class TestEffectiveModelId:
    def test_model_id_takes_priority(self):
        m = _mc(hf_repo="Qwen/Qwen2-7B", model_id="qwen-plus")
        assert m.effective_model_id == "qwen-plus"

    def test_falls_back_to_hf_repo(self):
        m = _mc(hf_repo="Qwen/Qwen2-7B")
        assert m.effective_model_id == "Qwen/Qwen2-7B"

    def test_falls_back_to_name_when_no_hf_repo(self):
        m = ModelConfig(name="deepseek-chat", provider="deepseek")
        assert m.effective_model_id == "deepseek-chat"

    def test_ollama_model_id_overrides_pull_name(self):
        m = _mc(provider="ollama", model_id="qwen2.5:7b")
        assert m.effective_model_id == "qwen2.5:7b"


# ──────────────────────────────────────────────────────────────
# load_models: cloud entry from YAML
# ──────────────────────────────────────────────────────────────

class TestLoadModelsCloud:
    def test_cloud_model_loads_all_new_fields(self, tmp_path):
        yaml_content = """\
models:
  - name: deepseek-v3
    provider: deepseek
    model_id: deepseek-chat
    api_key_env: DEEPSEEK_API_KEY
    translation_capable: true
    task_type: text_only
"""
        (tmp_path / "m.yaml").write_text(yaml_content)
        models = load_models(tmp_path / "m.yaml")
        assert len(models) == 1
        m = models[0]
        assert m.name == "deepseek-v3"
        assert m.provider == "deepseek"
        assert m.model_id == "deepseek-chat"
        assert m.api_key_env == "DEEPSEEK_API_KEY"
        assert m.base_url == "https://api.deepseek.com/v1"
        assert "translation" in m.capabilities
        assert "chat" in m.capabilities

    def test_cloud_model_omits_optional_local_fields(self, tmp_path):
        """Cloud entries are valid even without port/hf_repo/vram."""
        yaml_content = """\
models:
  - name: qwen-plus
    provider: dashscope
    model_id: qwen-plus
    task_type: text_only
"""
        (tmp_path / "m.yaml").write_text(yaml_content)
        models = load_models(tmp_path / "m.yaml")
        m = models[0]
        assert m.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert m.port == 0
        assert m.hf_repo == ""

    def test_ollama_model_loads(self, tmp_path):
        yaml_content = """\
models:
  - name: qwen2.5-7b-ollama
    provider: ollama
    model_id: qwen2.5:7b
    port: 11434
    task_type: text_only
    translation_capable: true
"""
        (tmp_path / "m.yaml").write_text(yaml_content)
        models = load_models(tmp_path / "m.yaml")
        m = models[0]
        assert m.provider == "ollama"
        assert m.base_url == "http://localhost:11434/v1"
        assert m.effective_model_id == "qwen2.5:7b"

    def test_local_vllm_unchanged(self, tmp_path):
        """Existing local_vllm entries work without any new fields."""
        yaml_content = """\
models:
  - name: qwen3-vl-8b-instruct
    hf_repo: Qwen/Qwen3-VL-8B-Instruct
    port: 8001
    vram_estimate_gb: 20
    role: vlm_primary
"""
        (tmp_path / "m.yaml").write_text(yaml_content)
        models = load_models(tmp_path / "m.yaml")
        m = models[0]
        assert m.provider == "local_vllm"
        assert m.base_url == "http://localhost:8001/v1"
        assert m.auth_header == "Bearer EMPTY"
        assert m.effective_model_id == "Qwen/Qwen3-VL-8B-Instruct"
