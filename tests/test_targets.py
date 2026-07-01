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


def test_windows_targets_advertise_cpu_gpu_npu():
    targets = load_targets()
    amd = targets["amd-win-x86"]
    intel = targets["intel-win-x86"]

    assert amd.supports_accelerator("cpu")
    assert amd.supports_accelerator("vulkan")
    assert amd.supports_accelerator("directml")
    assert amd.supports_accelerator("amd-xdna")

    assert intel.supports_accelerator("cpu")
    assert intel.supports_accelerator("directml")
    assert intel.supports_accelerator("openvino-gpu")
    assert intel.supports_accelerator("intel-ai-boost-npu")


def test_rockchip_targets_split_rk3588_and_rk182x():
    targets = load_targets()
    assert targets["rk3588-linux"].supports_accelerator("rknn-npu")
    assert targets["rk182x-linux"].supports_accelerator("rk1820-npu")


def test_k3_targets_split_memory_sizes():
    targets = load_targets()
    assert "k3-riscv" in targets
    assert "k3-riscv-16g" in targets
    assert "k3-riscv-8g" in targets

    assert targets["k3-riscv-16g"].supports_accelerator("rvv")
    assert targets["k3-riscv-16g"].supports_accelerator("ime2")
    assert targets["k3-riscv-8g"].supports_accelerator("rvv")
    assert targets["k3-riscv-8g"].supports_accelerator("ime2")
