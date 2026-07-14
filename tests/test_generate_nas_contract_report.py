from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "generate_nas_contract_report.py"
    spec = importlib.util.spec_from_file_location("generate_nas_contract_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_llm_quality_reason_reports_combined_translation_and_terminology_failure():
    mod = _load_module()

    reason = mod._llm_quality_reason({
        "translation": {
            "verdict": "FAIL",
            "verdict_reasons": [
                "[en->zh] FAIL: chrF 33.5 < 35.0",
                "[en->zh] FAIL: term-match 77% < 80%",
            ],
        }
    })

    assert reason == "translation_quality_and_terminology_failed"


def test_llm_quality_reason_keeps_single_terminology_failure():
    mod = _load_module()

    reason = mod._llm_quality_reason({
        "translation": {
            "verdict": "FAIL",
            "verdict_reasons": ["[en->zh] FAIL: term-match 77% < 80%"],
        }
    })

    assert reason == "translation_l3_terminology_failed"


def test_load_report_for_row_falls_back_to_local_report_basename(tmp_path):
    mod = _load_module()

    report = mod.ROOT / "output" / "reports" / "unit-remote-report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"model": "unit-model", "benchmarks": {"accuracy": {"verdict": "PASS"}}}', encoding="utf-8")
    try:
        loaded = mod._load_report_for_row({
            "model": "unit-model",
            "report": r"C:\Users\happy\vlm-llm-benchmark\output\reports\unit-remote-report.json",
        })
    finally:
        report.unlink(missing_ok=True)

    assert loaded["benchmarks"]["accuracy"]["verdict"] == "PASS"


def test_intel_windows_ollama_rows_are_cpu_resource_class():
    mod = _load_module()

    class Model:
        role = "llm_intel_win_hf_small"
        provider = "ollama"

    assert mod._resource_class(Model(), {}, "intel-win-x86") == "cpu"
