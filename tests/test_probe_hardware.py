from benchmark.probe.hardware import HardwareProbe, _CpuOnlyProbe, _RK182xProbe, _RKNNProbe

def test_cpu_only_probe_returns_dict():
    probe = _CpuOnlyProbe(None)
    result = probe.collect()
    assert "accelerator" in result
    assert result["accelerator"] == "cpu"
    assert "hostname_hash" in result

def test_hardware_probe_factory_cpu():
    class FakeCfg:
        accelerator = "cpu"
    p = HardwareProbe.for_target(FakeCfg())
    assert isinstance(p, _CpuOnlyProbe)

def test_hardware_probe_factory_rknn():
    class FakeCfg:
        accelerator = "rknn-npu"
    p = HardwareProbe.for_target(FakeCfg())
    assert isinstance(p, _RKNNProbe)

def test_hardware_probe_factory_rk182x():
    class FakeCfg:
        accelerator = "rk1820-npu"
        accelerator_profiles = ("cpu", "rk1820-npu", "rknn3")
        npu = "rk1820"
    p = HardwareProbe.for_target(FakeCfg())
    assert isinstance(p, _RK182xProbe)
    result = p.collect()
    assert result["accelerator"] == "rk1820-npu"
    assert result["extra"]["runtime_family"] == "rknn3"
    assert "rk1820-npu" in result["extra"]["accelerator_profiles"]

def test_rknn_probe_missing_sysfs_doesnt_crash():
    probe = _RKNNProbe(None)
    result = probe.collect()
    assert result["accelerator"] == "rknn-npu"

def test_hardware_probe_local_none_target():
    p = HardwareProbe.for_target(None)
    result = p.collect()
    assert isinstance(result, dict)
