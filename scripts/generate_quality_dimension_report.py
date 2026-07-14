#!/usr/bin/env python3
"""Generate full quality-dimension coverage reports from benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_benchmark as rb  # noqa: E402
from common import ModelConfig, _is_chat_capable, load_models  # noqa: E402


SCHEMA_VERSION = 1
CHAT_QUALITY_DIMS = (
    "accuracy",
    "translation",
    "general_ability",
    "conditioned",
    "long_context",
    "scenarios",
    "conversation_drift",
)
PRIMARY_DIM_BY_FAMILY = {
    "llm": "translation",
    "vlm": "accuracy",
    "embedding": "embedding",
    "reranker": "rerank",
    "ocr": "ocr",
    "asr": "asr",
}
MEASURED_STATUSES = {"passed", "failed", "warning", "measured"}
INCOMPLETE_STATUSES = {"missing", "blocked", "skipped", "error"}
ERROR_BLOCKERS = {
    "cpu_only_llm_vlm_blocked": "cpu_only_blocked",
    "runtime_repair_failed": "runtime_blocked",
    "model_process_failed": "runtime_blocked",
    "model_not_ready": "runtime_blocked",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


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


def _required_dims(model: ModelConfig, require_chat_long_context: bool) -> list[str]:
    caps = set(getattr(model, "capabilities", ()) or ())
    benchmarks = getattr(model, "benchmarks", None) or {}
    skipped = set(benchmarks.get("skip") or [])
    dims: list[str] = []
    kind = _model_kind(model)
    if kind in {"llm", "vlm"}:
        for dim in CHAT_QUALITY_DIMS:
            if dim in skipped:
                continue
            if dim == "translation" and "translation" not in caps:
                continue
            if dim == "long_context" and not require_chat_long_context:
                if not rb.DIMENSIONS[dim].gate(model):
                    continue
            dims.append(dim)
    if "embedding" in caps:
        dims.append("embedding")
    if "rerank" in caps or "rerank_native" in caps:
        dims.append("rerank")
    if "ocr" in caps:
        dims.append("ocr")
    if "asr" in caps:
        dims.append("asr")
    return [dim for dim in rb.QUALITY_DIMS if dim in set(dims)]


def _target_models(models_yaml: Path, targets: set[str]) -> dict[str, ModelConfig]:
    out: dict[str, ModelConfig] = {}
    for model in load_models(models_yaml):
        if model.target in targets:
            out[model.name] = model
    return out


def _status_from_verdict(verdict: Any) -> str:
    v = str(verdict or "").upper()
    if v == "PASS":
        return "passed"
    if v == "FAIL":
        return "failed"
    if v == "WARN":
        return "warning"
    if v == "BLOCKED":
        return "blocked"
    if v in {"SKIP", "SKIPPED"}:
        return "skipped"
    if v == "MEASURED":
        return "measured"
    if v:
        return v.lower()
    return "measured"


def _status_from_block(block: Any) -> tuple[str, str | None]:
    if not isinstance(block, dict):
        return "missing", None
    if block.get("skipped"):
        return "skipped", str(block.get("reason") or "skipped")
    verdict = block.get("verdict")
    if verdict is not None:
        status = _status_from_verdict(verdict)
    elif block.get("status"):
        status = _status_from_verdict(block.get("status"))
    else:
        status = "measured"
    reasons = block.get("verdict_reasons")
    if isinstance(reasons, list) and reasons:
        reason = " ; ".join(str(r) for r in reasons[:3])
    else:
        reason = str(block.get("reason")) if block.get("reason") else None
    return status, reason


def _timestamp_value(path: Path, payload: dict[str, Any]) -> float:
    ts = payload.get("timestamp")
    if isinstance(ts, str):
        try:
            return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _collect_raw_evidence(raw_reports: list[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    evidence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in raw_reports:
        payload = _read_json(path)
        model = str(payload.get("model") or path.stem)
        ts_value = _timestamp_value(path, payload)
        for dim, block in (payload.get("benchmarks") or {}).items():
            if dim not in rb.QUALITY_DIMS:
                continue
            status, reason = _status_from_block(block)
            evidence[(model, dim)].append({
                "status": status,
                "reason": reason,
                "source": _rel(path),
                "source_type": "raw_report",
                "timestamp_order": ts_value,
            })
    return evidence


def _collect_summary_evidence(summary_files: list[Path]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    evidence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    model_state: dict[str, dict[str, Any]] = {}
    for path in summary_files:
        payload = _read_json(path)
        order = path.stat().st_mtime if path.exists() else 0.0
        for row in payload.get("results") or []:
            model = str(row.get("model") or "")
            if not model:
                continue
            error = row.get("error")
            state = model_state.setdefault(model, {"errors": [], "sources": []})
            if error:
                state["errors"].append(str(error))
            state["sources"].append(_rel(path))
            for dim, status_value in (row.get("benchmarks") or {}).items():
                if dim not in rb.QUALITY_DIMS:
                    continue
                evidence[(model, dim)].append({
                    "status": _status_from_verdict(status_value),
                    "reason": str(error) if error else None,
                    "source": _rel(path),
                    "source_type": "summary",
                    "timestamp_order": order,
                })
    return evidence, model_state


def _pick_evidence(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return sorted(items, key=lambda x: (float(x.get("timestamp_order") or 0.0), x.get("source_type") == "raw_report"))[-1]


def _eligibility(model: str, required: list[str], row_statuses: dict[str, str], state: dict[str, Any] | None) -> str:
    errors = list((state or {}).get("errors") or [])
    if any(status in MEASURED_STATUSES for status in row_statuses.values()):
        return "feasible"
    if any(status in {"blocked", "skipped", "error"} for status in row_statuses.values()):
        return "platform_blocked"
    if errors:
        return ERROR_BLOCKERS.get(errors[-1], "runtime_blocked")
    if required:
        return "not_run"
    return "not_applicable"


def build_report(
    *,
    models_yaml: Path,
    targets: list[str],
    raw_reports: list[Path],
    summary_files: list[Path],
    run_id: str,
    require_chat_long_context: bool = True,
) -> dict[str, Any]:
    target_set = set(targets)
    if not target_set:
        raise SystemExit("provide at least one --target")
    models = _target_models(models_yaml, target_set)
    raw_evidence = _collect_raw_evidence(raw_reports)
    summary_evidence, model_state = _collect_summary_evidence(summary_files)
    all_evidence = defaultdict(list)
    for key, items in summary_evidence.items():
        all_evidence[key].extend(items)
    for key, items in raw_evidence.items():
        all_evidence[key].extend(items)

    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    family_dim_status: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for model_name, model in sorted(models.items()):
        kind = _model_kind(model)
        if kind == "other":
            continue
        required = _required_dims(model, require_chat_long_context)
        dim_statuses: dict[str, str] = {}
        dim_sources: dict[str, str] = {}
        dim_reasons: dict[str, str] = {}
        for dim in required:
            picked = _pick_evidence(all_evidence.get((model_name, dim), []))
            if picked is None:
                status = "missing"
                source = ""
                reason = None
            else:
                status = str(picked["status"])
                source = str(picked["source"])
                reason = picked.get("reason")
            dim_statuses[dim] = status
            dim_sources[dim] = source
            if reason:
                dim_reasons[dim] = str(reason)
            if status in MEASURED_STATUSES:
                family_dim_status[(model.target or "", kind, dim)].add(model_name)
            if status in INCOMPLETE_STATUSES:
                gaps.append({
                    "target": model.target,
                    "model": model_name,
                    "family": kind,
                    "dimension": dim,
                    "status": status,
                    "reason": reason,
                    "source": source,
                })
        eligibility = _eligibility(model_name, required, dim_statuses, model_state.get(model_name))
        complete = bool(required) and eligibility == "feasible" and all(status in MEASURED_STATUSES for status in dim_statuses.values())
        rows.append({
            "target": model.target,
            "model": model_name,
            "family": kind,
            "provider": model.provider,
            "role": model.role,
            "capabilities": sorted(getattr(model, "capabilities", ()) or []),
            "eligibility_status": eligibility,
            "required_quality_dimensions": required,
            "quality_dimension_status": dim_statuses,
            "quality_dimension_sources": dim_sources,
            "quality_dimension_reasons": dim_reasons,
            "quality_complete": complete,
        })

    family_rows: list[dict[str, Any]] = []
    for target in sorted(target_set):
        for family in ("llm", "vlm", "embedding", "reranker", "ocr", "asr"):
            primary = PRIMARY_DIM_BY_FAMILY[family]
            applicable = [r for r in rows if r["target"] == target and r["family"] == family]
            feasible = [r for r in applicable if r["eligibility_status"] == "feasible"]
            measured = sorted(family_dim_status.get((target, family, primary), set()))
            family_rows.append({
                "target": target,
                "family": family,
                "applicable_models": len(applicable),
                "feasible_models": len(feasible),
                "primary_quality_dimension": primary,
                "measured_models": measured,
                "status": "covered" if measured else ("blocked" if applicable else "missing"),
            })

    feasible_rows = [r for r in rows if r["eligibility_status"] == "feasible"]
    feasible_incomplete = [
        r for r in feasible_rows if not all(s in MEASURED_STATUSES for s in r["quality_dimension_status"].values())
    ]
    family_gaps = [r for r in family_rows if r["status"] != "covered"]
    status_counts = Counter()
    for row in rows:
        for status in row["quality_dimension_status"].values():
            status_counts[status] += 1
    overall_status = "complete" if not feasible_incomplete and not family_gaps else "partial"
    if overall_status == "complete" and any("failed" in r["quality_dimension_status"].values() for r in rows):
        overall_status = "complete_with_quality_failures"

    return {
        "$schema": "docs/quality-dimension.schema.json",
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "run_id": run_id,
        "models_yaml": _rel(models_yaml),
        "targets": sorted(target_set),
        "quality_dimensions": list(rb.QUALITY_DIMS),
        "require_chat_long_context": require_chat_long_context,
        "overall_status": overall_status,
        "quality_coverage_status": "complete" if not feasible_incomplete else "partial",
        "family_coverage_status": "complete" if not family_gaps else "partial",
        "summary": {
            "model_count": len(rows),
            "feasible_model_count": len(feasible_rows),
            "quality_complete_feasible_models": sum(1 for r in feasible_rows if r["quality_complete"]),
            "status_counts": dict(sorted(status_counts.items())),
            "family_status_counts": dict(sorted(Counter(r["status"] for r in family_rows).items())),
        },
        "family_coverage": family_rows,
        "model_quality_rows": rows,
        "gaps": {
            "feasible_model_dimension_gaps": [
                gap for gap in gaps
                if next((r for r in rows if r["model"] == gap["model"]), {}).get("eligibility_status") == "feasible"
            ],
            "family_coverage_gaps": family_gaps,
            "runtime_or_cpu_blocked_models": [
                {
                    "target": r["target"],
                    "model": r["model"],
                    "family": r["family"],
                    "eligibility_status": r["eligibility_status"],
                    "required_quality_dimensions": r["required_quality_dimensions"],
                }
                for r in rows
                if r["eligibility_status"] not in {"feasible", "not_applicable"}
            ],
        },
    }


def _write_tsv(path: Path, report: dict[str, Any]) -> None:
    cols = [
        "target",
        "model",
        "family",
        "eligibility_status",
        "quality_complete",
        *report["quality_dimensions"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in report["model_quality_rows"]:
            item = {
                "target": row["target"],
                "model": row["model"],
                "family": row["family"],
                "eligibility_status": row["eligibility_status"],
                "quality_complete": row["quality_complete"],
            }
            item.update({dim: row["quality_dimension_status"].get(dim, "not_applicable") for dim in report["quality_dimensions"]})
            writer.writerow(item)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Quality Dimension Coverage Report",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- overall_status: `{report['overall_status']}`",
        f"- quality_coverage_status: `{report['quality_coverage_status']}`",
        f"- family_coverage_status: `{report['family_coverage_status']}`",
        f"- quality_dimensions: `{', '.join(report['quality_dimensions'])}`",
        "",
        "## Family Coverage",
        "",
        "| target | family | status | feasible_models | measured_models | primary_dim |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in report["family_coverage"]:
        lines.append(
            "| "
            + " | ".join([
                str(row["target"]),
                row["family"],
                row["status"],
                str(row["feasible_models"]),
                ", ".join(row["measured_models"]) or "-",
                row["primary_quality_dimension"],
            ])
            + " |"
        )
    lines += [
        "",
        "## Model Quality Matrix",
        "",
        "| target | model | family | eligibility | complete | required_dims | statuses |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in report["model_quality_rows"]:
        statuses = ", ".join(f"{dim}:{status}" for dim, status in row["quality_dimension_status"].items()) or "-"
        lines.append(
            "| "
            + " | ".join([
                str(row["target"]),
                f"`{row['model']}`",
                row["family"],
                row["eligibility_status"],
                str(row["quality_complete"]).lower(),
                ", ".join(row["required_quality_dimensions"]) or "-",
                statuses,
            ])
            + " |"
        )
    lines += [
        "",
        "## Gaps",
        "",
        f"- feasible_model_dimension_gaps: `{len(report['gaps']['feasible_model_dimension_gaps'])}`",
        f"- family_coverage_gaps: `{len(report['gaps']['family_coverage_gaps'])}`",
        f"- runtime_or_cpu_blocked_models: `{len(report['gaps']['runtime_or_cpu_blocked_models'])}`",
        "",
        "A `failed` quality dimension is still measured evidence. `blocked`, `skipped`, and `missing` are coverage gaps for feasible models.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "quality-dimensions.json",
        "tsv": output_dir / "quality-dimensions.tsv",
        "markdown": output_dir / "quality-dimensions.md",
    }
    _write_json(paths["json"], report)
    _write_tsv(paths["tsv"], report)
    _write_markdown(paths["markdown"], report)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-yaml", type=Path, default=ROOT / "models.yaml")
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--raw-report", action="append", type=Path, default=[])
    parser.add_argument("--raw-dir", action="append", type=Path, default=[])
    parser.add_argument("--summary", action="append", type=Path, default=[])
    parser.add_argument("--run-id", default=f"quality-dimensions-{dt.datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "reports" / "quality-dimensions" / "latest")
    parser.add_argument(
        "--no-require-chat-long-context",
        action="store_true",
        help="use run_benchmark's default long-context gate instead of full-quality chat/VLM coverage",
    )
    args = parser.parse_args()

    raw_reports = list(args.raw_report)
    for raw_dir in args.raw_dir:
        raw_reports.extend(sorted(raw_dir.glob("*.json")))
    report = build_report(
        models_yaml=args.models_yaml,
        targets=args.target,
        raw_reports=sorted(raw_reports),
        summary_files=sorted(args.summary),
        run_id=args.run_id,
        require_chat_long_context=not args.no_require_chat_long_context,
    )
    paths = write_artifacts(report, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0 if report["quality_coverage_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
