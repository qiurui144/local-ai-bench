from common import load_targets, TargetConfig


def test_load_targets_local_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    targets = load_targets()
    assert "local" in targets
    assert targets["local"].is_local()


def test_ip_from_env(monkeypatch):
    monkeypatch.setenv("AMD_HOST", "192.168.100.201")
    cfg = TargetConfig(name="amd", platform="windows", arch="x86_64",
                       connection="ssh", ip_env="AMD_HOST")
    assert cfg.ip == "192.168.100.201"


def test_ip_missing_env_returns_empty():
    cfg = TargetConfig(name="t", platform="linux", arch="x86_64",
                       connection="ssh", ip_env="NONEXISTENT_ENV_VAR_XYZ")
    assert cfg.ip == ""


def test_local_target_is_local():
    cfg = TargetConfig(name="local", platform="linux", arch="x86_64", connection="local")
    assert cfg.is_local()
