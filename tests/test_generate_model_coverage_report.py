from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "generate_model_coverage_report.py"
    spec = importlib.util.spec_from_file_location("generate_model_coverage_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _row(model: str, item: str, task: str, resource: str, verdict: str = "sync_default") -> dict:
    return {
        "test_item_id": item,
        "task_class": task,
        "model_artifact_id": f"{model}@artifact",
        "model_profile": {"name": model},
        "runtime": {"resource_class": resource},
        "product_verdict": verdict,
        "product_verdict_reason": "unit",
    }


def test_model_coverage_report_distinguishes_inventory_from_contract_gaps(tmp_path):
    mod = _load_module()
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  - name: llm-a
    target: amd-win-x86
    provider: ollama
    task_type: text_only
  - name: vlm-a
    target: amd-win-x86
    provider: ollama
    task_type: vlm
  - name: emb-a
    target: amd-win-x86
    provider: openai
    task_type: text_only
    embedding_capable: true
  - name: rank-a
    target: amd-win-x86
    provider: openai
    task_type: text_only
    rerank_capable: true
  - name: ocr-gpu
    target: amd-win-x86
    provider: local_onnx
    task_type: text_only
    ocr_capable: true
  - name: ocr-npu
    target: amd-win-x86
    provider: local_onnx
    task_type: text_only
    ocr_capable: true
  - name: asr-npu
    target: amd-win-x86
    provider: local_onnx
    task_type: text_only
    asr_capable: true
  - name: extra-registered
    target: amd-win-x86
    provider: ollama
    task_type: text_only
""",
        encoding="utf-8",
    )
    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()
    rows = [
        _row("llm-a", "llm_chat_boundary", "llm_chat", "igpu", "not_recommended"),
        _row("vlm-a", "vlm_image_qa_boundary", "vlm_qa", "igpu", "not_recommended"),
        _row("emb-a", "embedding_retrieval", "embedding", "igpu"),
        _row("rank-a", "reranker_candidates", "reranker", "igpu"),
        _row("ocr-gpu", "ocr_pages", "ocr", "igpu"),
        _row("ocr-npu", "ocr_pages", "ocr", "npu"),
        _row("asr-npu", "asr_duration_concurrency", "asr", "npu", "not_recommended"),
    ]
    (contract_dir / "run-summary.json").write_text(
        json.dumps({
            "target": "amd-win-x86",
            "run_id": "unit",
            "status": "complete",
            "coverage_summary": {
                "planned_test_items": ["llm_chat_boundary"],
                "completed_test_items": ["llm_chat_boundary"],
                "blocked_test_items": [],
                "missing_required_profiles": [],
            },
        }),
        encoding="utf-8",
    )
    (contract_dir / "parameter-matrix.json").write_text(
        json.dumps({"target": "amd-win-x86", "rows": rows}),
        encoding="utf-8",
    )

    report = mod.build_report(
        models_yaml=models_yaml,
        contract_dirs=[contract_dir],
        targets=["amd-win-x86"],
        run_id="unit",
    )

    assert report["coverage_status"] == "complete"
    assert report["overall_status"] == "complete_with_quality_caveats"
    assert report["gaps"]["required_coverage_gaps"] == []
    assert report["gaps"]["contract_gaps"] == []
    unmeasured = {row["model"] for row in report["gaps"]["registered_not_in_contract_run"]}
    assert unmeasured == {"extra-registered"}

    paths = mod.write_artifacts(report, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["tsv"].exists()
    assert paths["markdown"].exists()
