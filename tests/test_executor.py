"""RemoteExecutor 单元测试 — mock subprocess，不需要真实 SSH。"""
import pytest
from unittest.mock import patch
from common import TargetConfig
from benchmark.executor import remote as remote_mod
from benchmark.executor.remote import RemoteExecutor


def _make_target(**kwargs):
    defaults = dict(name="test-target", platform="linux", arch="aarch64",
                    connection="ssh", ip_env=None, ssh_user_env=None, ssh_pass_env=None,
                    runtime="ollama", remote_workdir="/home/user/bench", python_cmd="python3")
    defaults.update(kwargs)
    cfg = TargetConfig(**defaults)
    return cfg


def test_remote_executor_check_env_raises_no_ip(monkeypatch):
    monkeypatch.delenv("NONEXISTENT_IP", raising=False)
    cfg = _make_target(ip_env="NONEXISTENT_IP")
    with pytest.raises(ValueError, match="requires env var"):
        RemoteExecutor(cfg)


def test_remote_executor_env_set(monkeypatch):
    monkeypatch.setenv("TEST_BENCH_IP", "10.0.0.1")
    monkeypatch.setenv("TEST_BENCH_USER", "user")
    monkeypatch.setenv("TEST_BENCH_PASS", "pass")
    cfg = _make_target(ip_env="TEST_BENCH_IP",
                       ssh_user_env="TEST_BENCH_USER", ssh_pass_env="TEST_BENCH_PASS")
    ex = RemoteExecutor(cfg)
    assert ex.target.ip == "10.0.0.1"


def test_sync_code_calls_rsync(monkeypatch, tmp_path):
    monkeypatch.setenv("T_IP", "1.2.3.4")
    monkeypatch.setenv("T_USER", "u")
    monkeypatch.setenv("T_PASS", "p")
    cfg = _make_target(ip_env="T_IP", ssh_user_env="T_USER", ssh_pass_env="T_PASS")
    ex = RemoteExecutor(cfg)
    with patch("subprocess.check_call") as mock_cc:
        ex.sync_code()
    assert mock_cc.called
    call_args = mock_cc.call_args[0][0]
    assert "rsync" in call_args[0]


def test_windows_run_remote_injects_env_overrides(monkeypatch):
    monkeypatch.setenv("T_IP", "1.2.3.4")
    monkeypatch.setenv("T_USER", "u")
    monkeypatch.setenv("T_PASS", "p")
    cfg = _make_target(
        platform="windows",
        arch="x86_64",
        ip_env="T_IP",
        ssh_user_env="T_USER",
        ssh_pass_env="T_PASS",
        remote_workdir=r"C:\Users\u\bench",
        python_cmd="python.exe",
        env_overrides={"OLLAMA_INTEL_WIN_BASE_URL": "http://localhost:11434/v1"},
    )
    ex = RemoteExecutor(cfg)
    with patch("subprocess.call", return_value=0) as mock_call:
        ex.run_remote("m1", ["--target", "intel-win-x86"])
    cmd = mock_call.call_args[0][0]
    joined = " ".join(cmd)
    assert "cmd /S /C" in joined
    assert 'set "OLLAMA_INTEL_WIN_BASE_URL=http://localhost:11434/v1"' in joined
    assert "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand" in joined
    assert "--local-only --target intel-win-x86" in joined


def test_windows_run_remote_skips_ollama_start_for_local_onnx(monkeypatch):
    monkeypatch.setenv("T_IP", "1.2.3.4")
    monkeypatch.setenv("T_USER", "u")
    monkeypatch.setenv("T_PASS", "p")
    cfg = _make_target(
        platform="windows",
        arch="x86_64",
        ip_env="T_IP",
        ssh_user_env="T_USER",
        ssh_pass_env="T_PASS",
        remote_workdir=r"C:\Users\u\bench",
        python_cmd="python.exe",
        env_overrides={"OLLAMA_INTEL_WIN_BASE_URL": "http://localhost:11434/v1"},
    )
    ex = RemoteExecutor(cfg)
    with patch("subprocess.call", return_value=0) as mock_call:
        ex.run_remote("rapidocr-intel-directml", ["--target", "intel-win-x86"])
    joined = " ".join(mock_call.call_args[0][0])
    assert 'set "OLLAMA_INTEL_WIN_BASE_URL=http://localhost:11434/v1"' in joined
    assert "EncodedCommand" not in joined


def test_windows_sync_code_uses_tar_allowlist(monkeypatch):
    monkeypatch.setenv("T_IP", "1.2.3.4")
    monkeypatch.setenv("T_USER", "u")
    monkeypatch.setenv("T_PASS", "p")
    cfg = _make_target(
        platform="windows",
        arch="x86_64",
        ip_env="T_IP",
        ssh_user_env="T_USER",
        ssh_pass_env="T_PASS",
        remote_workdir=r"C:\Users\u\bench",
        python_cmd="python.exe",
    )
    ex = RemoteExecutor(cfg)
    class _Stdout:
        def close(self):
            pass

    class _Proc:
        stdout = _Stdout()

        def wait(self, timeout=None):
            return 0

    with patch("subprocess.check_call") as mock_cc, \
            patch("subprocess.Popen", return_value=_Proc()) as mock_popen, \
            patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        ex.sync_code()

    mkdir_cmd = mock_cc.call_args_list[0][0][0]
    tar_cmd = mock_popen.call_args[0][0]
    ssh_cmd = mock_run.call_args[0][0]
    joined_tar = " ".join(tar_cmd)
    joined_ssh = " ".join(ssh_cmd)
    assert "cmd" in mkdir_cmd
    assert tar_cmd[:3] == ["tar", "-C", str(remote_mod.ROOT)]
    assert "-cf" in tar_cmd
    assert "run_benchmark.py" in joined_tar
    assert "--exclude=*/__pycache__" in tar_cmd
    assert "--exclude=*.pyc" in tar_cmd
    assert "reports" not in joined_tar
    assert ".ruff_cache" not in joined_tar
    assert ".coverage" not in joined_tar
    assert "tar -xf -" in joined_ssh


def test_windows_install_forces_directml_ort(monkeypatch):
    monkeypatch.setenv("T_IP", "1.2.3.4")
    monkeypatch.setenv("T_USER", "u")
    monkeypatch.setenv("T_PASS", "p")
    cfg = _make_target(
        platform="windows",
        arch="x86_64",
        ip_env="T_IP",
        ssh_user_env="T_USER",
        ssh_pass_env="T_PASS",
        remote_workdir=r"C:\Users\u\bench",
        python_cmd="python.exe",
    )
    ex = RemoteExecutor(cfg)
    with patch("subprocess.check_call") as mock_cc:
        ex.install_deps()
    joined = " ".join(mock_cc.call_args[0][0])
    assert "requirements-windows.txt" in joined
    assert "pip uninstall -y onnxruntime onnxruntime-directml" in joined
    assert "pip install openvino==2025.4.1 openvino-telemetry==2025.2.0" in joined
    assert "onnxruntime-directml rapidocr-onnxruntime rapidocr-openvino --no-deps" in joined
