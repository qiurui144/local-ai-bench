from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "serve_ov_intel.py"
    spec = importlib.util.spec_from_file_location("serve_ov_intel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_llm_ov_config_uses_conservative_gpu_options():
    mod = _load_module()

    cfg = mod._llm_ov_config("GPU", enable_large_allocations=True)

    assert cfg["GPU_ENABLE_LARGE_ALLOCATIONS"] == "YES"
    assert cfg["PERFORMANCE_HINT"] == "LATENCY"
    assert cfg["NUM_STREAMS"] == "1"


def test_llm_ov_config_can_disable_large_allocations():
    mod = _load_module()

    cfg = mod._llm_ov_config("GPU", enable_large_allocations=False)

    assert "GPU_ENABLE_LARGE_ALLOCATIONS" not in cfg
    assert cfg["NUM_STREAMS"] == "1"


def test_gpu_resource_error_detection():
    mod = _load_module()

    assert mod._is_gpu_resource_error(RuntimeError("[GPU] CL_OUT_OF_RESOURCES exception."))
    assert mod._is_gpu_resource_error(RuntimeError("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"))
    assert mod._is_gpu_resource_error(RuntimeError("subsequent OpenCL API call may cause the application to hang"))
    assert not mod._is_gpu_resource_error(RuntimeError("ordinary model error"))


def test_sampling_probability_error_detection():
    mod = _load_module()

    assert mod._is_sampling_probability_error(
        RuntimeError("probability tensor contains either inf, nan or element < 0")
    )
    assert not mod._is_sampling_probability_error(RuntimeError("ordinary model error"))


def test_sampling_probability_error_retries_deterministic_decode():
    mod = _load_module()

    class Request:
        def __init__(self):
            self.reset_count = 0

        def reset_state(self):
            self.reset_count += 1

    class Model:
        def __init__(self):
            self.calls = []
            self.request = Request()

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("probability tensor contains either inf, nan or element < 0")
            return ["ok"]

    model = Model()
    mod._llm_model = model

    result = mod._generate_llm_with_sampling_fallback({"input_ids": "ids"}, max_new_tokens=32, temperature=0.7)

    assert result == ["ok"]
    assert model.request.reset_count == 1
    assert model.calls[0]["do_sample"] is True
    assert model.calls[0]["temperature"] == 0.7
    assert model.calls[1]["do_sample"] is False
    assert "temperature" not in model.calls[1]


def test_gpu_resource_error_does_not_retry_as_sampling_failure():
    mod = _load_module()

    class Model:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("CL_OUT_OF_RESOURCES probability tensor contains nan")

    model = Model()
    mod._llm_model = model

    with pytest.raises(RuntimeError, match="CL_OUT_OF_RESOURCES"):
        mod._generate_llm_with_sampling_fallback({"input_ids": "ids"}, max_new_tokens=32, temperature=0.7)

    assert len(model.calls) == 1
