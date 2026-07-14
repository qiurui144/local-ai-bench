from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "generate_platform_hardware_reports.py"
    spec = importlib.util.spec_from_file_location("generate_platform_hardware_reports", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _row(model: str, task: str, resource: str, verdict: str = "sync_default") -> dict:
    return {
        "test_item_id": f"{task}_item",
        "task_class": task,
        "model_artifact_id": f"{model}@artifact",
        "model_profile": {"name": model},
        "runtime": {"resource_class": resource},
        "latency_profile": {"e2e_latency_ms": {"p95": 123.4}},
        "quality_profile": {"score": 0.9, "reason": "PASS"},
        "product_verdict": verdict,
        "product_verdict_reason": "unit_reason",
    }


def test_platform_hardware_reports_write_bilingual_pages_and_no_evidence_paths(tmp_path):
    mod = _load_module()
    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()
    (contract_dir / "parameter-matrix.json").write_text(
        json.dumps({
            "target": "amd-linux-x86",
            "run_id": "unit-run",
            "rows": [
                _row("emb", "embedding", "igpu"),
                _row("asr", "asr", "npu", "not_recommended"),
                _row("bad", "llm_chat", "blocked", "blocked"),
            ],
        }),
        encoding="utf-8",
    )
    (contract_dir / "run-summary.json").write_text(
        json.dumps({"target": "amd-linux-x86", "run_id": "unit-run", "status": "partial"}),
        encoding="utf-8",
    )
    for name in ("verdict-table.tsv", "model-profile.json", "scheduler-contract.json", "nas-contract-report.md"):
        (contract_dir / name).write_text("unit", encoding="utf-8")

    report = mod.build_report([contract_dir], targets=["amd-linux-x86"])
    paths = mod.write_reports(report, tmp_path / "reports" / "platforms")
    names = {path.name for path in paths}

    assert {"index.en.md", "index.zh.md", "cpu.en.md", "igpu.en.md", "npu.en.md", "blocked-runtime.en.md"}.issubset(names)
    index = (tmp_path / "reports" / "platforms" / "amd-linux" / "index.en.md").read_text(encoding="utf-8")
    assert "unit-run" in index
    assert "Blocked Runtime" in index
    assert "](../../../output/" not in index
    assert "local artifact dir" in index
    cpu = (tmp_path / "reports" / "platforms" / "amd-linux" / "cpu.en.md").read_text(encoding="utf-8")
    assert "No current contract rows" in cpu
    assert "](../../../output/" not in cpu
    zh = (tmp_path / "reports" / "platforms" / "amd-linux" / "npu.zh.md").read_text(encoding="utf-8")
    assert "该硬件条件有 1 条合同实测行" in zh
    assert "本地证据目录" in zh


def test_platform_hardware_reports_add_k3_contract_supplement_without_overwriting_index(tmp_path):
    mod = _load_module()
    contract_dir = tmp_path / "k3-contract"
    contract_dir.mkdir()
    (contract_dir / "parameter-matrix.json").write_text(
        json.dumps({
            "target": "k3-riscv-32g",
            "run_id": "k3-unit",
            "rows": [_row("qwen", "llm_chat", "x100_cpu", "async_default")],
        }),
        encoding="utf-8",
    )
    (contract_dir / "run-summary.json").write_text(
        json.dumps({"target": "k3-riscv-32g", "run_id": "k3-unit", "status": "complete"}),
        encoding="utf-8",
    )

    report = mod.build_report([contract_dir], targets=["k3-riscv-32g"])
    paths = mod.write_reports(report, tmp_path / "reports" / "platforms")
    names = {path.name for path in paths}

    assert names == {"contract.en.md", "contract.zh.md"}
    text = (tmp_path / "reports" / "platforms" / "k3-riscv-32g" / "contract.en.md").read_text(encoding="utf-8")
    assert "X100 CPU" in text
    assert "k3-unit" in text
    assert "](../../../output/" not in text
