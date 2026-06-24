"""Tests for scripts/export_attune_catalog.py — bench→attune catalog exporter.

Offline-only (§1.6): exercises the parser + manifest builder on fixture / real matrix
markdown; runs no model and no benchmark.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "export_attune_catalog.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_attune_catalog", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve cls.__module__ during exec.
    sys.modules["export_attune_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


FIXTURE_MATRIX = """
## Matrix

| Model | Target | Provider | Role | Caps | Status | Verdicts | Key metrics | Latest report |
|---|---|---|---|---|---|---|---|---|
| rapidocr-amd-directml | amd-win-x86 | local_onnx | ocr_gpu_directml | ocr | PASS | ocr:PASS | ocr CER 7.04% p50 468ms | a.json |
| rapidocr-intel-directml | intel-win-x86 | local_onnx | ocr_gpu_directml | ocr | FAIL | ocr:FAIL | ocr CER 202.35% | b.json |
| rapidocr-intel-openvino | intel-win-x86 | local_onnx | ocr_openvino_probe | ocr | PASS | ocr:PASS | ocr CER 7.04% p50 797ms | c.json |
| qwen3-embedding-0.6b-amd | amd-win-x86 | ollama | embedding_primary | embedding | PASS | embedding:PASS | embed hit@1 1.000 p50 875ms | d.json |
| sensevoice-small-amd-win | amd-win-x86 | local_onnx | asr_amd_win | asr | PASS | asr:PASS | asr CER 7.69% RTF 0.073 | e.json |
| qwen3-embedding-0.6b | local/reference |  | embedding_primary | embedding | REGISTERED | - | - | - |
""".lstrip()


def _write_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "matrix.en.md"
    p.write_text(FIXTURE_MATRIX, encoding="utf-8")
    return p


def test_parse_matrix_captures_rows_and_line_numbers(tmp_path):
    rows = M.parse_matrix(_write_fixture(tmp_path))
    assert len(rows) == 6
    # Line numbers must be captured (source refs depend on them).
    assert all(r.line > 0 for r in rows)
    amd_ocr = next(r for r in rows if r.model == "rapidocr-amd-directml")
    assert amd_ocr.target == "amd-win-x86"


def test_fail_row_is_never_selected(tmp_path):
    """The Intel DirectML OCR FAIL (CER 202%) must NOT appear in the catalog."""
    rows = M.parse_matrix(_write_fixture(tmp_path))
    cat = M.build_model_catalog(rows, "matrix.en.md")
    intel_ocr = cat["tiers"]["intel-win"]["ocr"]
    # Intel OCR must be OpenVINO (the PASS row), never DirectML (the FAIL row).
    assert intel_ocr["ep"] == "openvino"
    assert "202" not in intel_ocr.get("metric", "")


def test_amd_ocr_directml_kept(tmp_path):
    rows = M.parse_matrix(_write_fixture(tmp_path))
    cat = M.build_model_catalog(rows, "matrix.en.md")
    assert cat["tiers"]["amd-win"]["ocr"]["ep"] == "directml"


def test_every_entry_has_verdict_and_source(tmp_path):
    rows = M.parse_matrix(_write_fixture(tmp_path))
    cat = M.build_model_catalog(rows, "matrix.en.md")
    for tier, roles in cat["tiers"].items():
        for role, entry in roles.items():
            assert entry.get("verdict"), f"{tier}.{role} missing verdict"
            assert entry.get("source"), f"{tier}.{role} missing source (§6.3)"
            assert ":" in entry["source"], "source must be file:line"


def test_pending_verify_marked(tmp_path):
    """REGISTERED-only (no measured) entries are PENDING-VERIFY, not PASS."""
    rows = M.parse_matrix(_write_fixture(tmp_path))
    cat = M.build_model_catalog(rows, "matrix.en.md")
    # cpu-fallback embedding comes from local/reference REGISTERED row.
    cpu = cat["tiers"].get("cpu-fallback", {})
    if "embedding" in cpu:
        assert cpu["embedding"]["verdict"] == "PENDING-VERIFY"


def test_schema_and_metadata_present(tmp_path):
    rows = M.parse_matrix(_write_fixture(tmp_path))
    cat = M.build_model_catalog(rows, "matrix.en.md")
    assert cat["schema_version"] == M.SCHEMA_VERSION
    assert cat["source_repo"] == "local-ai-bench"
    assert cat["generated_at"].endswith("Z")


def test_output_yaml_round_trips(tmp_path):
    rows = M.parse_matrix(_write_fixture(tmp_path))
    cat = M.build_model_catalog(rows, "matrix.en.md")
    text = yaml.safe_dump(cat, sort_keys=False, allow_unicode=True)
    reparsed = yaml.safe_load(text)
    assert reparsed["tiers"]["amd-win"]["ocr"]["ep"] == "directml"


def test_driver_catalog_scans_real_drivers_dir():
    """driver-catalog must pick up real vendor packages, skipping PDFs."""
    dc = M.build_driver_catalog()
    assert dc["schema_version"] == M.SCHEMA_VERSION
    # amd-win has ryzen-ai exe + NPU zips (real files in drivers/amd-win/).
    if "amd-npu-win" in dc["tiers"]:
        names = [p["file"] for p in dc["tiers"]["amd-npu-win"]]
        assert any(n.endswith(".exe") or n.endswith(".zip") for n in names)
        assert not any(n.endswith(".pdf") for n in names), "PDFs must be skipped"


def test_real_matrix_report_parses_if_present():
    """Smoke: the real bench matrix report parses without error (no model run)."""
    real = REPO_ROOT / M.DEFAULT_MATRIX
    if not real.exists():
        pytest.skip("real matrix report not present")
    rows = M.parse_matrix(real)
    assert len(rows) > 10
    cat = M.build_model_catalog(rows, M.DEFAULT_MATRIX)
    # Real report: Intel OCR must resolve to OpenVINO (DirectML FAIL excluded).
    assert cat["tiers"]["intel-win"]["ocr"]["ep"] == "openvino"
    assert cat["tiers"]["amd-win"]["ocr"]["ep"] == "directml"
