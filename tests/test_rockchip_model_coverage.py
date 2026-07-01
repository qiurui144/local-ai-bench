"""Rockchip RKNN3 Model Zoo coverage in models.yaml."""

import yaml


RKNN3_V104_MODEL_NAMES = {
    # LLM directories/files under /RKNN3_SDK/rknn3_models/v1.0.4/llm
    "qwen2.5-0.5b-rk1820",
    "qwen2.5-1.5b-rk1820",
    "qwen2.5-3b-rk1820",
    "qwen2.5-7b-rk1820",
    "qwen3-0.6b-rk1820",
    "qwen3-1.7b-rk1820",
    "qwen3-4b-rk1820",
    "qwen3-8b-rk1820",
    "copaw-flash-4b-4k-rk1820",
    "copaw-flash-4b-8k-rk1820",
    "copaw-flash-4b-32k-rk1820",
    # VLM directories under /RKNN3_SDK/rknn3_models/v1.0.4/vlm
    "fastvlm-1.6b-rk1820",
    "internvl3-2b-rk1820",
    "internvl3.5-4b-rk1820",
    "janus-pro-1b-rk1820",
    "mimo-vl-7b-prune-rk1820",
    "minicpm-3o-rk1820",
    "minicpm-v-4-rk1820",
    "qwen2.5-omni-3b-rk1820",
    "qwen2.5-vl-3b-rk1820",
    "qwen2.5-vl-3b-prune-rk1820",
    "qwen2.5-vl-7b-prune-rk1820",
    "qwen3-vl-2b-rk1820",
    "qwen3-vl-4b-rk1820",
    "smolvlm-500m-rk1820",
    "smolvlm2-500m-rk1820",
    "ui-tars-2b-sft-rk1820",
    "gemma4-e2b-rk1820",
    "gemma4-e4b-rk1820",
    # Related OCR/VLM model under /RKNN3_SDK/rknn3_models/v1.0.4/others
    "paddleocr-vl-rk1820",
}


def test_rockchip_rknn3_v104_model_zoo_coverage():
    data = yaml.safe_load(open("models.yaml", encoding="utf-8"))
    models = {m["name"]: m for m in data["models"]}

    missing = sorted(RKNN3_V104_MODEL_NAMES - set(models))
    assert missing == []

    for name in RKNN3_V104_MODEL_NAMES:
        model = models[name]
        assert model["target"] == "rk182x-linux"
        assert model["provider"] == "generic"
        assert model["hardware_min"] == "rk1820-npu"


def test_rockchip_rknn3_entries_point_to_model_zoo_paths():
    data = yaml.safe_load(open("models.yaml", encoding="utf-8"))
    models = {m["name"]: m for m in data["models"]}

    pathless_allowlist = {
        # Active calibrated service name differs from the raw Model Zoo file layout.
        "qwen3-vl-2b-rk1820",
    }
    for name in RKNN3_V104_MODEL_NAMES - pathless_allowlist:
        path = models[name].get("rknn_model_path", "")
        assert path.startswith("/RKNN3_SDK/rknn3_models/v1.0.4/")
        assert path.endswith(".rknn")
