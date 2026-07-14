from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "generate_quality_dimension_report.py"
    spec = importlib.util.spec_from_file_location("generate_quality_dimension_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quality_dimension_report_requires_chat_high_dims_but_not_specialized_dims(tmp_path):
    mod = _load_module()
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  - name: llm-a
    target: unit-target
    provider: ollama
    task_type: text_only
    translation_capable: true
  - name: emb-a
    target: unit-target
    provider: ollama
    task_type: text_only
    embedding_capable: true
""",
        encoding="utf-8",
    )
    llm_raw = _write_json(
        tmp_path / "llm.json",
        {
            "model": "llm-a",
            "timestamp": "2026-07-13T10:00:00",
            "benchmarks": {
                "accuracy": {"verdict": "PASS"},
                "translation": {"verdict": "FAIL"},
                "general_ability": {"verdict": "PASS"},
                "conditioned": {"verdict": "PASS"},
                "scenarios": {"verdict": "WARN"},
                "conversation_drift": {"verdict": "PASS"},
            },
        },
    )
    emb_raw = _write_json(
        tmp_path / "emb.json",
        {
            "model": "emb-a",
            "timestamp": "2026-07-13T10:00:00",
            "benchmarks": {"embedding": {"verdict": "PASS"}},
        },
    )

    report = mod.build_report(
        models_yaml=models_yaml,
        targets=["unit-target"],
        raw_reports=[llm_raw, emb_raw],
        summary_files=[],
        run_id="unit",
        require_chat_long_context=True,
    )

    rows = {row["model"]: row for row in report["model_quality_rows"]}
    assert rows["llm-a"]["quality_dimension_status"]["long_context"] == "missing"
    assert rows["llm-a"]["quality_dimension_status"]["translation"] == "failed"
    assert rows["emb-a"]["required_quality_dimensions"] == ["embedding"]
    assert rows["emb-a"]["quality_complete"] is True
    assert report["quality_coverage_status"] == "partial"


def test_quality_dimension_report_does_not_count_cpu_only_block_as_feasible_gap(tmp_path):
    mod = _load_module()
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  - name: cpu-vlm
    target: unit-target
    provider: ollama
    task_type: vlm
""",
        encoding="utf-8",
    )
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "results": [
                {
                    "model": "cpu-vlm",
                    "error": "cpu_only_llm_vlm_blocked",
                    "benchmarks": None,
                }
            ]
        },
    )

    report = mod.build_report(
        models_yaml=models_yaml,
        targets=["unit-target"],
        raw_reports=[],
        summary_files=[summary],
        run_id="unit",
        require_chat_long_context=True,
    )

    row = report["model_quality_rows"][0]
    assert row["eligibility_status"] == "cpu_only_blocked"
    assert report["gaps"]["feasible_model_dimension_gaps"] == []
    assert report["gaps"]["runtime_or_cpu_blocked_models"][0]["model"] == "cpu-vlm"
