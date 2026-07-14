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


def test_model_coverage_report_applies_amd_linux_and_intel_win_requirements(tmp_path):
    mod = _load_module()
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  - name: qwen-amd-linux
    target: amd-linux-x86
    provider: ollama
    task_type: text_only
  - name: vlm-amd-linux
    target: amd-linux-x86
    provider: ollama
    task_type: vlm
  - name: emb-amd-linux
    target: amd-linux-x86
    provider: ollama
    task_type: text_only
    embedding_capable: true
  - name: rank-amd-linux
    target: amd-linux-x86
    provider: ollama
    task_type: text_only
    rerank_capable: true
  - name: rapidocr-amd-linux-directml
    target: amd-linux-x86
    provider: local_onnx
    task_type: text_only
    ocr_capable: true
  - name: rapidocr-amd-linux-npu
    target: amd-linux-x86
    provider: local_onnx
    task_type: text_only
    ocr_capable: true
  - name: whisper-amd-linux-npu
    target: amd-linux-x86
    provider: local_onnx
    task_type: text_only
    asr_capable: true
  - name: qwen-igpu-intel-win
    target: intel-win-x86
    provider: openai
    task_type: text_only
  - name: llava-intel-win
    target: intel-win-x86
    provider: ollama
    task_type: vlm
  - name: bge-base-en-v1.5-igpu-intel-win
    target: intel-win-x86
    provider: openai
    task_type: text_only
    embedding_capable: true
  - name: bge-reranker-base-igpu-intel-win
    target: intel-win-x86
    provider: openai
    task_type: text_only
    rerank_capable: true
  - name: paddleocr-openvino-intel-win
    target: intel-win-x86
    provider: local_onnx
    task_type: text_only
    ocr_capable: true
  - name: whisper-tiny-openvino-intel-win
    target: intel-win-x86
    provider: openai
    task_type: text_only
    asr_capable: true
  - name: whisper-base-npu-intel-win
    target: intel-win-x86
    provider: local_onnx
    task_type: text_only
    asr_capable: true
""",
        encoding="utf-8",
    )

    amd_dir = tmp_path / "amd-linux-contract"
    amd_dir.mkdir()
    amd_rows = [
        _row("qwen-amd-linux", "llm_chat_boundary", "llm_chat", "igpu"),
        _row("vlm-amd-linux", "vlm_image_qa_boundary", "vlm_qa", "igpu"),
        _row("emb-amd-linux", "embedding_retrieval", "embedding", "igpu"),
        _row("rank-amd-linux", "reranker_candidates", "reranker", "igpu"),
        _row("rapidocr-amd-linux-directml", "ocr_pages", "ocr", "igpu"),
        _row("rapidocr-amd-linux-npu", "ocr_pages", "ocr", "npu"),
        _row("whisper-amd-linux-npu", "asr_duration_concurrency", "asr", "npu"),
    ]
    (amd_dir / "run-summary.json").write_text(
        json.dumps({
            "target": "amd-linux-x86",
            "run_id": "unit-amd",
            "status": "complete",
            "coverage_summary": {"blocked_test_items": [], "missing_required_profiles": []},
        }),
        encoding="utf-8",
    )
    (amd_dir / "parameter-matrix.json").write_text(
        json.dumps({"target": "amd-linux-x86", "rows": amd_rows}),
        encoding="utf-8",
    )

    intel_dir = tmp_path / "intel-win-contract"
    intel_dir.mkdir()
    intel_rows = [
        _row("qwen-igpu-intel-win", "llm_chat_boundary", "llm_chat", "igpu"),
        _row("llava-intel-win", "vlm_image_qa_boundary", "vlm_qa", "igpu"),
        _row("bge-base-en-v1.5-igpu-intel-win", "embedding_retrieval", "embedding", "igpu"),
        _row("bge-reranker-base-igpu-intel-win", "reranker_candidates", "reranker", "igpu"),
        _row("paddleocr-openvino-intel-win", "ocr_pages", "ocr", "mixed"),
        _row("whisper-tiny-openvino-intel-win", "asr_duration_concurrency", "asr", "igpu"),
        _row("whisper-base-npu-intel-win", "asr_duration_concurrency", "asr", "npu"),
    ]
    (intel_dir / "run-summary.json").write_text(
        json.dumps({
            "target": "intel-win-x86",
            "run_id": "unit-intel",
            "status": "complete",
            "coverage_summary": {"blocked_test_items": [], "missing_required_profiles": []},
        }),
        encoding="utf-8",
    )
    (intel_dir / "parameter-matrix.json").write_text(
        json.dumps({"target": "intel-win-x86", "rows": intel_rows}),
        encoding="utf-8",
    )

    report = mod.build_report(
        models_yaml=models_yaml,
        contract_dirs=[amd_dir, intel_dir],
        targets=["amd-linux-x86", "intel-win-x86"],
        run_id="unit",
    )

    assert report["coverage_status"] == "complete"
    by_target = {}
    for row in report["required_coverage"]:
        by_target.setdefault(row["target"], set()).add(row["requirement_id"])
        assert row["status"] == "covered"
    assert by_target["amd-linux-x86"] == {
        "llm_igpu",
        "vlm_igpu",
        "embedding_igpu",
        "reranker_igpu",
        "ocr_directml_igpu",
        "ocr_npu",
        "asr_any",
        "asr_npu",
    }
    assert by_target["intel-win-x86"] == {
        "llm_openvino_igpu",
        "vlm_igpu",
        "embedding_openvino_igpu",
        "reranker_openvino_igpu",
        "ocr_openvino",
        "asr_openvino",
        "asr_npu",
    }
