#!/usr/bin/env python3
"""Generate a normalized model coverage report from contract artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import ModelConfig, _is_chat_capable, load_models  # noqa: E402


REQUIRED_COVERAGE: dict[str, list[tuple[str, str, Callable[[dict[str, Any]], bool]]]] = {
    "amd-win-x86": [
        ("llm_igpu", "LLM/RAG answer path on AMD iGPU", lambda r: _is_llm_row(r) and _resource(r) == "igpu"),
        ("vlm_igpu", "VLM path on AMD iGPU", lambda r: _is_vlm_row(r) and _resource(r) == "igpu"),
        ("embedding_igpu", "Embedding path on AMD iGPU", lambda r: r.get("task_class") == "embedding" and _resource(r) == "igpu"),
        ("reranker_igpu", "Reranker path on AMD iGPU", lambda r: r.get("task_class") == "reranker" and _resource(r) == "igpu"),
        ("ocr_igpu", "OCR path on AMD iGPU/DirectML", lambda r: r.get("task_class") == "ocr" and _resource(r) == "igpu"),
        ("ocr_npu", "OCR path on AMD NPU/VitisAI", lambda r: r.get("task_class") == "ocr" and _resource(r) == "npu"),
        ("asr_any", "ASR path present", lambda r: r.get("task_class") == "asr"),
        ("asr_npu", "ASR path on AMD NPU", lambda r: r.get("task_class") == "asr" and _resource(r) == "npu"),
    ],
    "intel-linux": [
        ("llm_openvino_igpu", "LLM/RAG answer path on Intel OpenVINO GPU", lambda r: _is_llm_row(r) and "openvino" in _model_name(r)),
        ("vlm_vulkan_igpu", "VLM path on Intel Vulkan/Ollama GPU", lambda r: _is_vlm_row(r) and _resource(r) == "igpu"),
        ("embedding_openvino_igpu", "Embedding path on Intel OpenVINO GPU", lambda r: r.get("task_class") == "embedding" and "igpu-intel-linux" in _model_name(r)),
        ("reranker_openvino_igpu", "Reranker path on Intel OpenVINO GPU", lambda r: r.get("task_class") == "reranker" and "igpu-intel-linux" in _model_name(r)),
        ("ocr_openvino", "OCR path on Intel OpenVINO", lambda r: r.get("task_class") == "ocr" and "openvino" in _model_name(r)),
        ("asr_openvino", "ASR path on Intel OpenVINO", lambda r: r.get("task_class") == "asr" and "openvino" in _model_name(r)),
    ],
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _model_name(row: dict[str, Any]) -> str:
    profile = row.get("model_profile") if isinstance(row.get("model_profile"), dict) else {}
    return str(profile.get("name") or row.get("model_artifact_id") or "").lower()


def _resource(row: dict[str, Any]) -> str:
    runtime = row.get("runtime") if isinstance(row.get("runtime"), dict) else {}
    return str(runtime.get("resource_class") or "unknown")


def _is_llm_row(row: dict[str, Any]) -> bool:
    return row.get("task_class") in {"llm_chat", "llm_summary", "rag_answer"}


def _is_vlm_row(row: dict[str, Any]) -> bool:
    return str(row.get("task_class") or "").startswith("vlm_")


def _model_kind(model: ModelConfig) -> str:
    caps = set(getattr(model, "capabilities", ()) or ())
    if "ocr" in caps:
        return "ocr"
    if "asr" in caps:
        return "asr"
    if "embedding" in caps:
        return "embedding"
    if "rerank" in caps or "rerank_native" in caps:
        return "reranker"
    if getattr(model, "task_type", "") == "vlm":
        return "vlm"
    if _is_chat_capable(model):
        return "llm"
    return "other"


def _capabilities(model: ModelConfig) -> list[str]:
    caps = list(getattr(model, "capabilities", ()) or ())
    if getattr(model, "task_type", "") == "vlm" and "vlm" not in caps:
        caps.append("vlm")
    return sorted(caps)


def _contract_dirs(paths: list[Path] | None) -> list[Path]:
    if paths:
        return paths
    root = ROOT / "output" / "reports" / "contract"
    return sorted(p for p in root.glob("*") if (p / "run-summary.json").exists())


def _load_contract(contract_dir: Path) -> dict[str, Any]:
    run_summary = _read_json(contract_dir / "run-summary.json")
    parameter_matrix = _read_json(contract_dir / "parameter-matrix.json")
    target = str(run_summary.get("target") or parameter_matrix.get("target") or "")
    return {
        "dir": contract_dir,
        "target": target,
        "run_summary": run_summary,
        "parameter_matrix": parameter_matrix,
        "rows": list(parameter_matrix.get("rows") or []),
    }


def _row_key(row: dict[str, Any]) -> str:
    return f"{row.get('test_item_id')}:{row.get('model_artifact_id')}"


def _contract_target_summary(contract: dict[str, Any]) -> dict[str, Any]:
    rs = contract["run_summary"]
    coverage = rs.get("coverage_summary") or {}
    rows = contract["rows"]
    verdict_counts = Counter(str(r.get("product_verdict")) for r in rows)
    resources = Counter(_resource(r) for r in rows)
    return {
        "target": contract["target"],
        "run_id": rs.get("run_id"),
        "contract_status": rs.get("status"),
        "contract_dir": str(contract["dir"]),
        "planned_test_items": coverage.get("planned_test_items") or [],
        "completed_test_items": coverage.get("completed_test_items") or [],
        "blocked_test_items": coverage.get("blocked_test_items") or [],
        "missing_required_profiles": coverage.get("missing_required_profiles") or [],
        "row_count": len(rows),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "resource_counts": dict(sorted(resources.items())),
    }


def _coverage_requirements(target: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [r for r in rows if r.get("product_verdict") != "blocked"]
    out: list[dict[str, Any]] = []
    for req_id, description, predicate in REQUIRED_COVERAGE.get(target, []):
        evidence = [r for r in usable if predicate(r)]
        out.append({
            "target": target,
            "requirement_id": req_id,
            "description": description,
            "status": "covered" if evidence else "missing",
            "evidence_models": sorted({_model_name(r) for r in evidence}),
            "evidence_test_items": sorted({str(r.get("test_item_id")) for r in evidence}),
            "evidence_verdicts": sorted({str(r.get("product_verdict")) for r in evidence}),
        })
    return out


def _models_by_target(models_yaml: Path, targets: set[str]) -> dict[str, list[ModelConfig]]:
    grouped: dict[str, list[ModelConfig]] = {target: [] for target in targets}
    for model in load_models(models_yaml):
        if model.target in grouped:
            grouped[model.target].append(model)
    return grouped


def _model_rows(
    grouped_models: dict[str, list[ModelConfig]],
    rows_by_target_model: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for target, models in sorted(grouped_models.items()):
        for model in sorted(models, key=lambda m: m.name):
            evidence = rows_by_target_model.get((target, model.name), [])
            non_blocked = [r for r in evidence if r.get("product_verdict") != "blocked"]
            if non_blocked:
                status = "measured"
            elif evidence:
                status = "blocked_evidence"
            else:
                status = "registered_not_in_contract_run"
            out.append({
                "target": target,
                "model": model.name,
                "provider": model.provider,
                "role": model.role,
                "kind": _model_kind(model),
                "capabilities": _capabilities(model),
                "coverage_status": status,
                "evidence_row_count": len(evidence),
                "resource_classes": sorted({_resource(r) for r in evidence}),
                "test_items": sorted({str(r.get("test_item_id")) for r in evidence}),
                "product_verdicts": sorted({str(r.get("product_verdict")) for r in evidence}),
            })
    return out


def build_report(
    *,
    models_yaml: Path,
    contract_dirs: list[Path] | None,
    targets: list[str] | None,
    run_id: str,
) -> dict[str, Any]:
    contracts = [_load_contract(path) for path in _contract_dirs(contract_dirs)]
    if targets:
        target_set = set(targets)
        contracts = [c for c in contracts if c["target"] in target_set]
    else:
        target_set = {c["target"] for c in contracts}
    if not contracts:
        raise SystemExit("no contract artifacts selected")

    rows_by_target_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for contract in contracts:
        target = contract["target"]
        for row in contract["rows"]:
            name = ""
            profile = row.get("model_profile")
            if isinstance(profile, dict):
                name = str(profile.get("name") or "")
            if name:
                rows_by_target_model[(target, name)].append(row)
            all_rows_by_target[target].append(row)

    grouped_models = _models_by_target(models_yaml, target_set)
    model_rows = _model_rows(grouped_models, rows_by_target_model)
    target_summaries = [_contract_target_summary(c) for c in contracts]
    requirements: list[dict[str, Any]] = []
    for target in sorted(target_set):
        requirements.extend(_coverage_requirements(target, all_rows_by_target.get(target, [])))

    contract_gaps = [
        {"target": s["target"], "blocked_test_items": s["blocked_test_items"], "missing_required_profiles": s["missing_required_profiles"]}
        for s in target_summaries
        if s["blocked_test_items"] or s["missing_required_profiles"] or s["contract_status"] != "complete"
    ]
    required_gaps = [r for r in requirements if r["status"] != "covered"]
    blocked_rows = [
        {
            "target": target,
            "model": (row.get("model_profile") or {}).get("name"),
            "test_item_id": row.get("test_item_id"),
            "reason": row.get("product_verdict_reason"),
        }
        for target, rows in all_rows_by_target.items()
        for row in rows
        if row.get("product_verdict") == "blocked"
    ]
    quality_caveats = [
        {
            "target": target,
            "model": (row.get("model_profile") or {}).get("name"),
            "test_item_id": row.get("test_item_id"),
            "verdict": row.get("product_verdict"),
            "reason": row.get("product_verdict_reason"),
        }
        for target, rows in all_rows_by_target.items()
        for row in rows
        if row.get("product_verdict") in {"not_recommended", "offline_only", "async_only"}
    ]
    registered_unmeasured = [r for r in model_rows if r["coverage_status"] == "registered_not_in_contract_run"]
    cpu_only_rows = [
        {
            "target": target,
            "model": (row.get("model_profile") or {}).get("name"),
            "test_item_id": row.get("test_item_id"),
        }
        for target, rows in all_rows_by_target.items()
        for row in rows
        if _resource(row) == "cpu"
    ]

    coverage_complete = not contract_gaps and not required_gaps and not cpu_only_rows
    overall_status = "complete"
    if not coverage_complete:
        overall_status = "partial"
    elif quality_caveats or blocked_rows:
        overall_status = "complete_with_quality_caveats"

    return {
        "$schema": "docs/model-coverage.schema.json",
        "schema_version": 1,
        "generated_at": _now(),
        "run_id": run_id,
        "models_yaml": str(models_yaml),
        "targets": sorted(target_set),
        "overall_status": overall_status,
        "coverage_status": "complete" if coverage_complete else "partial",
        "quality_status": "caveats" if quality_caveats or blocked_rows else "clean",
        "target_summaries": target_summaries,
        "required_coverage": requirements,
        "model_rows": model_rows,
        "gaps": {
            "contract_gaps": contract_gaps,
            "required_coverage_gaps": required_gaps,
            "cpu_only_rows": cpu_only_rows,
            "blocked_evidence_rows": blocked_rows,
            "quality_caveats": quality_caveats,
            "registered_not_in_contract_run": registered_unmeasured,
        },
    }


def _write_tsv(path: Path, report: dict[str, Any]) -> None:
    cols = [
        "target",
        "model",
        "provider",
        "role",
        "kind",
        "capabilities",
        "coverage_status",
        "evidence_row_count",
        "resource_classes",
        "test_items",
        "product_verdicts",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in report["model_rows"]:
            item = dict(row)
            for key in ("capabilities", "resource_classes", "test_items", "product_verdicts"):
                item[key] = ",".join(item.get(key) or [])
            writer.writerow({key: item.get(key, "") for key in cols})


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Model Coverage Report",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- overall_status: `{report['overall_status']}`",
        f"- coverage_status: `{report['coverage_status']}`",
        f"- quality_status: `{report['quality_status']}`",
        "",
        "## Contract Coverage",
        "",
        "| target | contract_status | rows | blocked_items | missing_profiles | verdict_counts | resources |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for item in report["target_summaries"]:
        lines.append(
            "| "
            + " | ".join([
                str(item["target"]),
                str(item["contract_status"]),
                str(item["row_count"]),
                ", ".join(item["blocked_test_items"]) or "-",
                ", ".join(item["missing_required_profiles"]) or "-",
                ", ".join(f"{k}:{v}" for k, v in item["verdict_counts"].items()) or "-",
                ", ".join(f"{k}:{v}" for k, v in item["resource_counts"].items()) or "-",
            ])
            + " |"
        )
    lines += [
        "",
        "## Required Platform Coverage",
        "",
        "| target | requirement | status | evidence_models | evidence_test_items | verdicts |",
        "|---|---|---:|---|---|---|",
    ]
    for req in report["required_coverage"]:
        lines.append(
            "| "
            + " | ".join([
                req["target"],
                req["requirement_id"],
                req["status"],
                ", ".join(req["evidence_models"]) or "-",
                ", ".join(req["evidence_test_items"]) or "-",
                ", ".join(req["evidence_verdicts"]) or "-",
            ])
            + " |"
        )
    lines += [
        "",
        "## Registered Model Inventory",
        "",
        "| target | model | kind | role | coverage_status | resources | verdicts | test_items |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for row in report["model_rows"]:
        lines.append(
            "| "
            + " | ".join([
                row["target"],
                row["model"],
                row["kind"],
                row["role"] or "-",
                row["coverage_status"],
                ", ".join(row["resource_classes"]) or "-",
                ", ".join(row["product_verdicts"]) or "-",
                ", ".join(row["test_items"]) or "-",
            ])
            + " |"
        )
    lines += [
        "",
        "## Gaps And Caveats",
        "",
        f"- contract_gaps: `{len(report['gaps']['contract_gaps'])}`",
        f"- required_coverage_gaps: `{len(report['gaps']['required_coverage_gaps'])}`",
        f"- cpu_only_rows: `{len(report['gaps']['cpu_only_rows'])}`",
        f"- blocked_evidence_rows: `{len(report['gaps']['blocked_evidence_rows'])}`",
        f"- quality_caveats: `{len(report['gaps']['quality_caveats'])}`",
        f"- registered_not_in_contract_run: `{len(report['gaps']['registered_not_in_contract_run'])}`",
        "",
        "Registered-but-unmeasured rows are inventory entries outside this contract baseline, not contract coverage failures.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "model-coverage.json",
        "tsv": output_dir / "model-coverage.tsv",
        "markdown": output_dir / "model-coverage.md",
    }
    _write_json(paths["json"], report)
    _write_tsv(paths["tsv"], report)
    _write_markdown(paths["markdown"], report)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-yaml", default=str(ROOT / "models.yaml"))
    parser.add_argument("--contract-dir", action="append", default=[], help="Contract artifact directory; repeatable. Defaults to all output/reports/contract/*")
    parser.add_argument("--target", action="append", default=[], help="Target to include; repeatable")
    parser.add_argument("--run-id", default=f"model-coverage-{dt.datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "reports" / "model-coverage" / "latest"))
    args = parser.parse_args()

    report = build_report(
        models_yaml=Path(args.models_yaml),
        contract_dirs=[Path(p) for p in args.contract_dir] or None,
        targets=args.target or None,
        run_id=args.run_id,
    )
    paths = write_artifacts(report, Path(args.output_dir))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0 if report["coverage_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
