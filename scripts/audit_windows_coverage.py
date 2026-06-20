#!/usr/bin/env python3
"""Audit model-type and benchmark-dimension coverage from existing reports."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = [
    "accuracy",
    "ttft",
    "throughput",
    "prefill_decode",
    "concurrency",
    "stability",
    "translation",
    "embedding",
    "rerank",
    "asr",
    "ocr",
    "general_ability",
    "conditioned",
    "scenarios",
    "conversation_drift",
]


def _load_models() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    return {m["name"]: m for m in data.get("models", [])}


def _classify(model: dict[str, Any]) -> str:
    if model.get("ocr_capable"):
        return "ocr"
    if model.get("asr_capable"):
        return "asr"
    if model.get("embedding_capable"):
        return "embedding"
    if model.get("rerank_capable") or model.get("rerank_native"):
        return "rerank"
    if model.get("is_vlm") or model.get("task_type") == "vlm":
        return "vlm"
    return "llm"


def _report_model_name(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("model") or "")
    except Exception:
        return ""


def _report_timestamp(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("timestamp") or data.get("started_at") or "")
    except Exception:
        return ""


def _reports_by_model(report_root: Path) -> dict[str, list[Path]]:
    reports: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(report_root.rglob("*.json")):
        # Skip seed detail and non-model probes for this model-dimension audit.
        if re.search(r"_seed\d+\.json$", path.name):
            continue
        model = _report_model_name(path)
        if not model:
            continue
        reports[model].append(path)
    return reports


def _benchmarks(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("benchmarks") or {}
    except Exception:
        return {}


def _latest_dimension_blocks(paths: list[Path]) -> tuple[dict[str, Any], dict[str, str]]:
    blocks: dict[str, tuple[str, Any, str]] = {}
    for path in paths:
        ts = _report_timestamp(path) or path.name
        for dim, block in _benchmarks(path).items():
            if dim not in DIMENSIONS:
                continue
            if dim not in blocks or ts > blocks[dim][0]:
                blocks[dim] = (ts, block, str(path))
    return {d: item[1] for d, item in blocks.items()}, {d: item[2] for d, item in blocks.items()}


def _coverage_status(block: Any) -> str:
    if not isinstance(block, dict):
        return "missing"
    verdict = block.get("verdict")
    if verdict in {"FAIL", "WARN", "BLOCKED", "SKIPPED", "PASS"}:
        return verdict
    if block:
        return "MEASURED"
    return "missing"


def build_audit(report_root: Path) -> dict[str, Any]:
    models = _load_models()
    reports_by_model = _reports_by_model(report_root)
    by_type: dict[str, list[str]] = defaultdict(list)
    model_rows: list[dict[str, Any]] = []
    dimension_rows: dict[str, list[dict[str, str]]] = {d: [] for d in DIMENSIONS}

    for name, model in sorted(models.items()):
        target = model.get("target") or "local"
        if target not in {"amd-win-x86", "intel-win-x86"}:
            continue
        typ = _classify(model)
        by_type[typ].append(name)
        report_paths = reports_by_model.get(name, [])
        benches, evidence_paths = _latest_dimension_blocks(report_paths)
        dims = {d: _coverage_status(benches.get(d)) for d in DIMENSIONS}
        model_rows.append({
            "model": name,
            "target": target,
            "type": typ,
            "reports": [str(p) for p in report_paths],
            "dimension_reports": evidence_paths,
            "dimensions": dims,
        })
        for dim, status in dims.items():
            if status != "missing":
                dimension_rows[dim].append({
                    "model": name,
                    "target": target,
                    "type": typ,
                    "status": status,
                    "report": evidence_paths.get(dim, ""),
                })

    return {
        "model_type_inventory": {k: sorted(v) for k, v in sorted(by_type.items())},
        "dimension_coverage": dimension_rows,
        "models": model_rows,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Windows Coverage Audit",
        "",
        "## Model Type Inventory",
        "",
    ]
    for typ, models in audit["model_type_inventory"].items():
        lines.append(f"- {typ}: {len(models)} model(s) - {', '.join(models)}")
    lines += ["", "## Dimension Coverage", ""]
    lines += ["| Dimension | Covered | Evidence |", "|---|---:|---|"]
    for dim in DIMENSIONS:
        rows = audit["dimension_coverage"][dim]
        evidence = "; ".join(
            f"{r['target']}:{r['model']}({r['status']})" for r in rows[:5]
        )
        if len(rows) > 5:
            evidence += f"; +{len(rows) - 5} more"
        lines.append(f"| {dim} | {len(rows)} | {evidence or 'missing'} |")
    lines += ["", "## Model x Dimension Matrix", ""]
    header = ["Model", "Target", "Type"] + DIMENSIONS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in audit["models"]:
        vals = [row["model"], row["target"], row["type"]]
        vals.extend(row["dimensions"][d] for d in DIMENSIONS)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default=str(ROOT / "output" / "reports"))
    parser.add_argument("--out-json", default=str(ROOT / "output" / "reports" / "windows_coverage_audit.json"))
    parser.add_argument("--out-md", default=str(ROOT / "reports" / "2026-06-19-windows-coverage-audit.md"))
    args = parser.parse_args()

    audit = build_audit(Path(args.report_root))
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(audit), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
