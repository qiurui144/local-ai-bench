"""OCR dimension tests — metrics, manifest loader, runner (injected recognizer)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from benchmark.ocr.metrics import cer, corpus_cer, corpus_ned, edit_distance, ned, wer
from benchmark.ocr.datasets import load_ocr_manifest
from benchmark.ocr.dimension import run_ocr_dimension
from benchmark.ocr.runner import build_recognizer, run_ocr


# ── metrics ────────────────────────────────────────────────────────────────

def test_edit_distance_identical():
    assert edit_distance("abc", "abc") == 0


def test_edit_distance_insert():
    assert edit_distance("abc", "abcd") == 1


def test_edit_distance_delete():
    assert edit_distance("abcd", "abc") == 1


def test_edit_distance_substitute():
    assert edit_distance("abc", "axc") == 1


def test_cer_perfect():
    assert cer("开放时间", "开放时间") == 0.0


def test_cer_empty_ref():
    assert cer("", "") == 0.0
    assert cer("", "abc") == 1.0


def test_cer_partial():
    # 1 substitution in 4 chars
    assert cer("开放时间", "开放实间") == pytest.approx(1 / 4)


def test_ned_zero():
    assert ned("hello", "hello") == 0.0


def test_ned_worst():
    # completely different, same length
    val = ned("abcd", "efgh")
    assert val == pytest.approx(1.0)


def test_ned_empty_both():
    assert ned("", "") == 0.0


def test_wer_perfect():
    assert wer("hello world", "hello world") == 0.0


def test_wer_one_error():
    assert wer("hello world", "hello earth") == pytest.approx(1 / 2)


def test_corpus_cer():
    refs = ["开放", "时间"]
    hyps = ["开放", "实间"]
    # 1 error / 4 total chars
    assert corpus_cer(refs, hyps) == pytest.approx(1 / 4)


def test_corpus_ned():
    refs = ["abc", "xyz"]
    hyps = ["abc", "xyz"]
    assert corpus_ned(refs, hyps) == 0.0


# ── manifest loader ─────────────────────────────────────────────────────────

def _make_manifest(tmp: Path, entries: list[dict]) -> tuple[Path, Path]:
    img_dir = tmp / "images"
    img_dir.mkdir()
    for e in entries:
        img = Image.new("RGB", (32, 32), color=(255, 255, 255))
        img.save(str(img_dir / e["image"]))
    mf = tmp / "manifest.jsonl"
    with mf.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return mf, img_dir


def test_load_manifest_basic(tmp_path):
    entries = [
        {"image": "a.png", "text": "你好", "uid": "a1", "source": "synthetic"},
        {"image": "b.png", "text": "world", "uid": "b1", "source": "curated"},
    ]
    mf, img_dir = _make_manifest(tmp_path, entries)
    samples = load_ocr_manifest(mf, image_root=img_dir)
    assert len(samples) == 2
    assert samples[0].text == "你好"
    assert samples[1].source == "curated"


def test_load_manifest_num_samples(tmp_path):
    entries = [{"image": f"{i}.png", "text": f"t{i}", "uid": f"u{i}"} for i in range(5)]
    mf, img_dir = _make_manifest(tmp_path, entries)
    samples = load_ocr_manifest(mf, image_root=img_dir, num_samples=3)
    assert len(samples) == 3


def test_load_manifest_missing(tmp_path):
    samples = load_ocr_manifest(tmp_path / "nonexistent.jsonl")
    assert samples == []


def test_load_manifest_missing_image(tmp_path):
    mf = tmp_path / "manifest.jsonl"
    mf.write_text('{"image": "ghost.png", "text": "x", "uid": "g1"}\n')
    samples = load_ocr_manifest(mf, image_root=tmp_path)
    assert samples == []


# ── runner ──────────────────────────────────────────────────────────────────

def _model_cfg(name="rapidocr-cpu"):
    cfg = MagicMock()
    cfg.name = name
    cfg.ocr_backend = "auto"
    return cfg


def test_run_ocr_blocked_no_manifest(tmp_path):
    result = run_ocr(_model_cfg(), manifest_path=None)
    assert result["verdict"] == "SKIP"
    assert result["status"] == "blocked"


def test_run_ocr_pass_perfect(tmp_path):
    entries = [
        {"image": "a.png", "text": "开放时间", "uid": "a1", "source": "synthetic"},
        {"image": "b.png", "text": "增值税", "uid": "b1", "source": "synthetic"},
    ]
    mf, img_dir = _make_manifest(tmp_path, entries)
    # Inject perfect recognizer
    texts = {"a.png": "开放时间", "b.png": "增值税"}
    def perfect(path: Path) -> str:
        return texts.get(path.name, "")
    result = run_ocr(_model_cfg(), manifest_path=mf, image_root=img_dir,
                     recognizer_fn=perfect)
    assert result["status"] == "ok"
    assert result["verdict"] == "PASS"
    assert result["aggregate"]["cer"] == 0.0


def test_run_ocr_fail_cer(tmp_path):
    entries = [{"image": "a.png", "text": "开放时间早上九点", "uid": "a1", "source": "synthetic"}]
    mf, img_dir = _make_manifest(tmp_path, entries)
    result = run_ocr(_model_cfg(), manifest_path=mf, image_root=img_dir,
                     recognizer_fn=lambda p: "x",
                     thresholds={"cer_max": 0.10})
    assert result["verdict"] == "FAIL"
    assert any("CER" in r for r in result["verdict_reasons"])


def test_run_ocr_empty_outputs(tmp_path):
    entries = [{"image": "a.png", "text": "你好", "uid": "a1", "source": "synthetic"}]
    mf, img_dir = _make_manifest(tmp_path, entries)
    result = run_ocr(_model_cfg(), manifest_path=mf, image_root=img_dir,
                     recognizer_fn=lambda p: "")
    assert result["aggregate"]["empty_output_count"] == 1
    assert result["verdict"] == "FAIL"


def test_run_ocr_backend_in_aggregate(tmp_path):
    entries = [{"image": "a.png", "text": "hello", "uid": "a1", "source": "synthetic"}]
    mf, img_dir = _make_manifest(tmp_path, entries)
    result = run_ocr(_model_cfg(), manifest_path=mf, image_root=img_dir,
                     recognizer_fn=lambda p: "hello")
    assert result["aggregate"]["backend"] == "injected"


def test_run_ocr_blocked_no_backend(tmp_path):
    entries = [{"image": "a.png", "text": "test", "uid": "a1", "source": "synthetic"}]
    mf, img_dir = _make_manifest(tmp_path, entries)
    # Force backend that won't be available
    result = run_ocr(_model_cfg(), manifest_path=mf, image_root=img_dir,
                     backend="vitisai")
    assert result["status"] == "blocked"
    assert result["verdict"] == "SKIP"


def test_build_recognizer_rejects_unknown_backend():
    recognizer, reason = build_recognizer("not-a-backend")
    assert recognizer is None
    assert "unsupported ocr backend" in reason


def test_directml_requires_dml_provider(monkeypatch):
    import benchmark.ocr.runner as runner

    monkeypatch.setattr(runner, "_ORT_AVAILABLE", True)
    monkeypatch.setattr(
        runner,
        "ort",
        MagicMock(get_available_providers=lambda: ["CPUExecutionProvider"]),
    )

    recognizer, reason = build_recognizer("directml")
    assert recognizer is None
    assert reason == "no ocr backend available"


def test_openvino_backend_uses_rapidocr_openvino(monkeypatch):
    import benchmark.ocr.runner as runner

    calls = []

    def fake_run_helper(helper, *, backend, image=None, timeout_s=180):
        calls.append({"helper": helper, "backend": backend, "image": image, "timeout_s": timeout_s})
        if image is None:
            return {"ok": True, "devices": ["CPU", "GPU", "NPU"]}
        return {"ok": True, "text": "hello"}

    helper = Path("scripts/ocr_rapidocr_subprocess.py")
    monkeypatch.setattr(runner, "_rapidocr_helper_path", lambda: helper)
    monkeypatch.setattr(runner, "_run_rapidocr_helper", fake_run_helper)

    recognizer, reason = build_recognizer("openvino")
    assert reason == "openvino"
    assert recognizer is not None
    assert recognizer(Path("x.png")) == "hello"
    assert calls == [
        {"helper": helper, "backend": "openvino", "image": None, "timeout_s": 180},
        {"helper": helper, "backend": "openvino", "image": Path("x.png"), "timeout_s": 180},
    ]


def test_openvino_backend_blocks_when_helper_missing(monkeypatch, tmp_path):
    import benchmark.ocr.runner as runner

    monkeypatch.setattr(runner, "_rapidocr_helper_path", lambda: tmp_path / "missing.py")

    recognizer, reason = build_recognizer("openvino")
    assert recognizer is None
    assert reason == "no ocr backend available"


def test_ocr_dimension_uses_model_backend(monkeypatch, tmp_path):
    mf, img_dir = _make_manifest(
        tmp_path,
        [{"image": "a.png", "text": "hello", "uid": "a1", "source": "synthetic"}],
    )
    seen = {}

    def fake_run_ocr(model_cfg, **kwargs):
        seen.update(kwargs)
        return {"benchmark": "ocr", "status": "blocked", "verdict": "SKIP"}

    cfg = _model_cfg()
    cfg.ocr_backend = "directml"
    monkeypatch.setattr("benchmark.ocr.dimension.run_ocr", fake_run_ocr)

    run_ocr_dimension(
        cfg,
        {
            "manifest": mf.name,
            "image_root": img_dir.name,
            "backend": "auto",
        },
        tmp_path,
    )

    assert seen["backend"] == "directml"


# ── models.yaml: ocr entries parse ──────────────────────────────────────────

def test_models_yaml_ocr_entries():
    import yaml
    with open("models.yaml") as f:
        data = yaml.safe_load(f)
    ocr_models = [m for m in data["models"] if m.get("ocr_capable")]
    assert len(ocr_models) >= 2, "Expected at least rapidocr-cpu and paddleocr-cpu"
    names = {m["name"] for m in ocr_models}
    assert "rapidocr-cpu" in names
    assert "paddleocr-cpu" in names
    assert "rapidocr-amd-npu" in names
    assert "rapidocr-amd-directml" in names
    assert "rapidocr-intel-directml" in names
    assert "rapidocr-intel-openvino" in names
    assert data["benchmarks"].get("ocr"), "ocr benchmark config missing"


def test_load_models_preserves_ocr_backend():
    from common import load_models

    models = load_models("models.yaml")
    directml = next(m for m in models if m.name == "rapidocr-amd-directml")
    npu = next(m for m in models if m.name == "rapidocr-amd-npu")

    assert directml.ocr_backend == "directml"
    assert npu.ocr_backend == "vitisai"


# ── VitisAI helper safety ───────────────────────────────────────────────────

def test_vitisai_requires_helper_probe_success(monkeypatch, tmp_path):
    import benchmark.ocr.runner as runner

    monkeypatch.setattr(runner, "_default_vitisai_python", lambda: Path("python.exe"))
    monkeypatch.setattr(runner, "_vitisai_helper_path", lambda: tmp_path / "missing.py")

    recognizer, reason = build_recognizer("vitisai")
    assert recognizer is None
    assert reason == "no ocr backend available"


def test_vitisai_helper_provider_and_text(monkeypatch, tmp_path):
    import benchmark.ocr.runner as runner

    helper = tmp_path / "ocr_vitisai_rapidocr.py"
    helper.write_text("# helper placeholder\n", encoding="utf-8")
    calls = []

    def fake_run_helper(python_exe, helper_path, *, image=None, model_args=None, timeout_s=180):
        calls.append({"image": image, "model_args": model_args})
        if image is None:
            return {"ok": True, "providers": ["VitisAIExecutionProvider", "CPUExecutionProvider"]}
        return {"ok": True, "text": "hello npu"}

    monkeypatch.setattr(runner, "_default_vitisai_python", lambda: Path("python.exe"))
    monkeypatch.setattr(runner, "_vitisai_helper_path", lambda: helper)
    monkeypatch.setattr(runner, "_run_vitisai_helper", fake_run_helper)

    recognizer, reason = build_recognizer("vitisai")
    assert reason == "vitisai"
    assert recognizer is not None
    assert recognizer(Path("sample.png")) == "hello npu"
    assert calls[0]["image"] is None
    assert calls[1]["image"] == Path("sample.png")


def test_vitisai_rejects_non_npu_provider(monkeypatch, tmp_path):
    import benchmark.ocr.runner as runner

    helper = tmp_path / "ocr_vitisai_rapidocr.py"
    helper.write_text("# helper placeholder\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_default_vitisai_python", lambda: Path("python.exe"))
    monkeypatch.setattr(runner, "_vitisai_helper_path", lambda: helper)
    monkeypatch.setattr(
        runner,
        "_run_vitisai_helper",
        lambda *args, **kwargs: {"ok": True, "providers": ["CPUExecutionProvider"]},
    )

    recognizer, reason = build_recognizer("vitisai")
    assert recognizer is None
    assert reason == "no ocr backend available"
