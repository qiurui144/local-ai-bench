"""Report schema v1: every report carries version + hardware identity (arch review P0.1)."""
import run_benchmark as rb


def test_run_all_result_carries_schema_v1_fields(monkeypatch):
    monkeypatch.setattr(rb, "wait_model_ready", lambda *a, **kw: False)
    monkeypatch.setattr(rb, "get_vram_info", lambda *a, **kw: {})
    monkeypatch.setattr(rb, "get_hardware_profile", lambda m=None: {
        "gpu": "stub", "driver": "0", "cuda": "0", "vllm": "0", "hostname_hash": "abc"})

    class _M:
        name = "m1"
        hf_repo = "org/stub"
        quantization = None
        hardware_min = "n/a"
        port = 9999
    result = rb.run_all_for_model(_M(), {}, set(), {})
    assert result["schema_version"] == 1
    assert isinstance(result["harness_version"], str) and result["harness_version"]
    assert set(result["hardware_profile"]) == {"gpu", "driver", "cuda", "vllm", "hostname_hash"}
    assert result["condition"] == {"context_tokens": None, "cache_mode": None}


def test_get_hardware_profile_degrades_to_unknown_not_crash():
    prof = rb.get_hardware_profile(None)   # no GPU / no endpoint on CI
    # legacy fields still present; new probe adds accelerator/arch/cpu_model/total_memory_gb/extra
    required = {"gpu", "driver", "cuda", "vllm", "hostname_hash",
                "accelerator", "arch", "cpu_model", "total_memory_gb", "extra"}
    assert required.issubset(set(prof))
    # string fields must be non-empty strings
    for key in ("gpu", "driver", "cuda", "vllm", "hostname_hash", "accelerator", "arch"):
        assert isinstance(prof[key], str) and prof[key], f"{key!r} must be non-empty str"


def test_harness_version_is_short_string():
    v = rb.harness_version()
    assert isinstance(v, str) and 1 <= len(v) <= 40
