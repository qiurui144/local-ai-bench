import yaml

from common import load_models


X86_PLATFORM_TARGETS = (
    "amd-win-x86",
    "amd-linux-x86",
    "intel-win-x86",
    "intel-linux",
)

SMALL_MODEL_MARKERS = (
    "0.5b",
    "0.6b",
    "1b",
    "1.5b",
    "1.7b",
    "3b",
    "3.5-mini",
    "4b",
)


def _models_by_target():
    grouped = {target: [] for target in X86_PLATFORM_TARGETS}
    for model in load_models():
        if model.target in grouped:
            grouped[model.target].append(model)
    return grouped


def test_intel_amd_windows_linux_have_full_role_coverage():
    grouped = _models_by_target()
    for target, models in grouped.items():
        assert models, f"{target} has no models"
        assert any(m.task_type == "text_only" and "chat" in m.capabilities for m in models), target
        assert any(m.task_type == "vlm" for m in models), target
        assert any("embedding" in m.capabilities for m in models), target
        assert any("rerank" in m.capabilities for m in models), target
        assert any("asr" in m.capabilities for m in models), target
        assert any("ocr" in m.capabilities for m in models), target


def test_k3_models_are_split_by_memory_size():
    raw = yaml.safe_load(open("models.yaml", encoding="utf-8"))
    k3_models = [m for m in raw["models"] if "k3" in m["name"] and "riscv" in m["name"]]

    assert k3_models
    assert all(m.get("target") != "k3-riscv" for m in k3_models)

    k3_16g = [m for m in k3_models if m.get("target") == "k3-riscv-16g"]
    k3_8g = [m for m in k3_models if m.get("target") == "k3-riscv-8g"]
    assert k3_16g
    assert k3_8g
    assert any("7b" in m["name"].lower() for m in k3_16g)
    assert any("3b" in m["name"].lower() for m in k3_16g)

    for model in k3_16g:
        assert "16GB" in model.get("hardware_min", ""), model["name"]
    for model in k3_8g:
        lower_name = model["name"].lower()
        assert "8GB" in model.get("hardware_min", ""), model["name"]
        assert "3b" not in lower_name
        assert "7b" not in lower_name


def test_intel_amd_small_models_are_hf_sourced():
    raw = yaml.safe_load(open("models.yaml", encoding="utf-8"))
    for target in X86_PLATFORM_TARGETS:
        hf_small = [
            m for m in raw["models"]
            if m.get("target") == target
            and m.get("hf_repo")
            and any(marker in m["name"].lower() for marker in SMALL_MODEL_MARKERS)
        ]
        assert len(hf_small) >= 4, (target, [m["name"] for m in hf_small])
        assert any("Qwen2.5" in m["hf_repo"] for m in hf_small), target
        assert any(
            "Llama-3.2" in m["hf_repo"] or "Phi-3.5" in m["hf_repo"]
            for m in hf_small
        ), target


def test_intel_amd_practical_models_use_platform_optimized_hf_sources():
    raw = yaml.safe_load(open("models.yaml", encoding="utf-8"))
    by_target = {
        target: [m for m in raw["models"] if m.get("target") == target]
        for target in X86_PLATFORM_TARGETS
    }

    for target in ("amd-win-x86", "amd-linux-x86"):
        models = by_target[target]
        assert any(m.get("asr_capable") and m.get("hf_repo", "").startswith("amd/") for m in models), target
        assert any(m.get("translation_capable") and m.get("hf_repo", "").startswith("amd/") for m in models), target
        assert any(
            m.get("ocr_capable")
            and m.get("hf_repo")
            and m.get("ocr_backend") in {"vitisai", "directml"}
            for m in models
        ), target

    for target in ("intel-win-x86", "intel-linux"):
        models = by_target[target]
        assert any(m.get("asr_capable") and m.get("hf_repo", "").startswith("Intel/") for m in models), target
        assert any(m.get("translation_capable") and m.get("hf_repo", "").startswith("OpenVINO/") for m in models), target
        assert any(
            m.get("ocr_capable")
            and "OpenVINO" in m.get("hf_repo", "")
            and m.get("ocr_backend") == "openvino"
            for m in models
        ), target
