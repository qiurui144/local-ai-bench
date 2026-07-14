#!/usr/bin/env python3
"""Generate canonical platform/hardware reports from NAS contract artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

VERDICT_ORDER = {
    "blocked": 0,
    "not_recommended": 1,
    "offline_only": 2,
    "async_only": 3,
    "async_default": 4,
    "sync_bounded": 5,
    "sync_default": 6,
}
USABLE_VERDICTS = {"sync_default", "sync_bounded", "async_default", "async_only", "offline_only"}
EXPECTED_X86_PATHS = ["cpu", "igpu", "npu", "mixed", "blocked-runtime"]

PLATFORMS: dict[str, dict[str, Any]] = {
    "amd-win-x86": {
        "slug": "amd-windows",
        "title_en": "AMD Windows",
        "title_zh": "AMD Windows",
        "expected_paths": EXPECTED_X86_PATHS,
        "scope_en": (
            "AMD Windows contract reporting is split by CPU, Radeon 780M iGPU, "
            "XDNA NPU, mixed runtime, and blocked-runtime rows."
        ),
        "scope_zh": "AMD Windows 合同报告按 CPU、Radeon 780M iGPU、XDNA NPU、混合运行时和阻塞运行时行拆分。",
    },
    "intel-win-x86": {
        "slug": "intel-windows",
        "title_en": "Intel Windows",
        "title_zh": "Intel Windows",
        "expected_paths": EXPECTED_X86_PATHS,
        "scope_en": (
            "Intel Windows contract reporting is split by CPU, Intel Arc iGPU, "
            "AI Boost NPU, mixed OpenVINO/runtime rows, and blocked-runtime rows."
        ),
        "scope_zh": "Intel Windows 合同报告按 CPU、Intel Arc iGPU、AI Boost NPU、OpenVINO/混合运行时和阻塞运行时行拆分。",
    },
    "amd-linux-x86": {
        "slug": "amd-linux",
        "title_en": "AMD Linux",
        "title_zh": "AMD Linux",
        "expected_paths": EXPECTED_X86_PATHS,
        "scope_en": (
            "AMD Linux contract reporting is split by CPU fallback/tool rows, "
            "Radeon 780M iGPU/Vulkan rows, Linux NPU probe rows, mixed rows, "
            "and blocked-runtime rows."
        ),
        "scope_zh": "AMD Linux 合同报告按 CPU fallback/tool、Radeon 780M iGPU/Vulkan、Linux NPU probe、混合运行时和阻塞运行时行拆分。",
    },
    "intel-linux": {
        "slug": "intel-linux",
        "title_en": "Intel Linux",
        "title_zh": "Intel Linux",
        "expected_paths": EXPECTED_X86_PATHS,
        "scope_en": (
            "Intel Linux contract reporting is split by CPU fallback rows, "
            "OpenVINO/iGPU rows, NPU rows where present, mixed OpenVINO rows, "
            "and blocked-runtime rows."
        ),
        "scope_zh": "Intel Linux 合同报告按 CPU fallback、OpenVINO/iGPU、可用 NPU、OpenVINO 混合运行时和阻塞运行时行拆分。",
    },
    "k3-riscv-32g": {
        "slug": "k3-riscv-32g",
        "title_en": "K3 RISC-V 32G",
        "title_zh": "K3 RISC-V 32G",
        "expected_paths": ["x100-cpu", "a100-ime2", "mixed", "blocked-runtime"],
        "contract_page_only": True,
        "scope_en": (
            "K3 32G contract reporting is added as a contract supplement to the "
            "existing K3 platform reports."
        ),
        "scope_zh": "K3 32G 合同报告作为现有 K3 平台报告的合同补充页。",
    },
}

HARDWARE: dict[str, dict[str, str]] = {
    "cpu": {
        "file": "cpu",
        "en": "CPU",
        "zh": "CPU",
        "scope_en": "Rows whose contract runtime resource class is `cpu`.",
        "scope_zh": "合同 runtime resource class 为 `cpu` 的行。",
    },
    "igpu": {
        "file": "igpu",
        "en": "iGPU",
        "zh": "iGPU",
        "scope_en": "Rows whose contract runtime resource class is `igpu`.",
        "scope_zh": "合同 runtime resource class 为 `igpu` 的行。",
    },
    "npu": {
        "file": "npu",
        "en": "NPU",
        "zh": "NPU",
        "scope_en": "Rows whose contract runtime resource class is `npu`.",
        "scope_zh": "合同 runtime resource class 为 `npu` 的行。",
    },
    "mixed": {
        "file": "mixed",
        "en": "Mixed Runtime",
        "zh": "混合运行时",
        "scope_en": "Rows whose contract runtime resource class is `mixed`.",
        "scope_zh": "合同 runtime resource class 为 `mixed` 的行。",
    },
    "blocked-runtime": {
        "file": "blocked-runtime",
        "en": "Blocked Runtime",
        "zh": "阻塞运行时",
        "scope_en": "Rows that did not reach a concrete hardware runtime and are grouped as blocked runtime evidence.",
        "scope_zh": "未进入明确硬件运行时的行，统一归为阻塞运行时证据。",
    },
    "x100-cpu": {
        "file": "x100-cpu",
        "en": "X100 CPU",
        "zh": "X100 CPU",
        "scope_en": "Rows whose contract runtime resource class is `x100_cpu`.",
        "scope_zh": "合同 runtime resource class 为 `x100_cpu` 的行。",
    },
    "a100-ime2": {
        "file": "a100-ime2",
        "en": "A100 IME2",
        "zh": "A100 IME2",
        "scope_en": "Rows whose contract runtime resource class is `a100_ime2`.",
        "scope_zh": "合同 runtime resource class 为 `a100_ime2` 的行。",
    },
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _date() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rel(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_dir).replace(os.sep, "/")


def _safe(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt_float(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_ms(value: Any) -> str:
    formatted = _fmt_float(value)
    return "-" if formatted == "-" else f"{formatted}ms"


def _fmt_score(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _model_name(row: dict[str, Any]) -> str:
    profile = row.get("model_profile") if isinstance(row.get("model_profile"), dict) else {}
    return str(profile.get("name") or row.get("model_artifact_id") or "unknown")


def _latency_p95(row: dict[str, Any]) -> Any:
    latency = row.get("latency_profile") if isinstance(row.get("latency_profile"), dict) else {}
    e2e = latency.get("e2e_latency_ms") if isinstance(latency.get("e2e_latency_ms"), dict) else {}
    return e2e.get("p95")


def _quality_score(row: dict[str, Any]) -> Any:
    quality = row.get("quality_profile") if isinstance(row.get("quality_profile"), dict) else {}
    return quality.get("score")


def _quality_reason(row: dict[str, Any]) -> str:
    quality = row.get("quality_profile") if isinstance(row.get("quality_profile"), dict) else {}
    return str(quality.get("reason") or row.get("product_verdict_reason") or "not_recorded")


def _params_identity(row: dict[str, Any]) -> str:
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    data = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _params_brief(row: dict[str, Any]) -> str:
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    keys = [
        "context_tokens",
        "target_context_tokens",
        "max_output_tokens",
        "startup_state",
        "finish_reason",
        "concurrency",
        "document_pages",
        "image_count",
    ]
    parts = []
    for key in keys:
        value = params.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    if parts:
        return ", ".join(parts)
    return f"params_hash={_params_identity(row)}"


def _hardware_key(row: dict[str, Any]) -> str:
    runtime = row.get("runtime") if isinstance(row.get("runtime"), dict) else {}
    raw = str(runtime.get("resource_class") or "mixed").strip().lower().replace("_", "-")
    if raw == "blocked":
        return "blocked-runtime"
    if raw in HARDWARE:
        return raw
    return re.sub(r"[^a-z0-9-]+", "-", raw).strip("-") or "mixed"


def _artifact_links(contract_dir: Path, from_dir: Path, lang: str) -> str:
    labels = {
        "en": {
            "parameter-matrix.json": "Parameter matrix",
            "run-summary.json": "Run summary",
            "verdict-table.tsv": "Verdict table",
            "model-profile.json": "Model profile",
            "scheduler-contract.json": "Scheduler contract",
            "nas-contract-report.md": "Contract report",
        },
        "zh": {
            "parameter-matrix.json": "参数矩阵",
            "run-summary.json": "运行摘要",
            "verdict-table.tsv": "verdict 表",
            "model-profile.json": "模型画像",
            "scheduler-contract.json": "scheduler 合同",
            "nas-contract-report.md": "合同报告",
        },
    }[lang]
    links = []
    for name, label in labels.items():
        path = contract_dir / name
        if path.exists():
            links.append(f"[{label}]({_rel(from_dir, path)})")
    return ", ".join(links) if links else "-"


def discover_contract_dirs(root: Path = ROOT) -> list[Path]:
    dirs = []
    contract_root = root / "output" / "reports" / "contract"
    if contract_root.exists():
        dirs.extend(path for path in sorted(contract_root.iterdir()) if (path / "parameter-matrix.json").exists())
    k3_contract = root / "output" / "reports" / "k3-riscv-32g" / "contract-qwen30b-20260708_221818" / "contract"
    if (k3_contract / "parameter-matrix.json").exists():
        dirs.append(k3_contract)
    return dirs


def load_contracts(contract_dirs: list[Path], targets: set[str] | None = None) -> dict[str, dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}
    for source_order, contract_dir in enumerate(contract_dirs):
        matrix_path = contract_dir / "parameter-matrix.json"
        if not matrix_path.exists():
            continue
        matrix = _read_json(matrix_path)
        summary_path = contract_dir / "run-summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        target = str(matrix.get("target") or summary.get("target") or "")
        if not target or (targets and target not in targets):
            continue
        item = by_target.setdefault(target, {"contracts": [], "rows": []})
        run_id = str(matrix.get("run_id") or summary.get("run_id") or contract_dir.name)
        generated_at = str(summary.get("generated_at") or matrix.get("generated_at") or "")
        item["contracts"].append({
            "dir": contract_dir,
            "run_id": run_id,
            "summary": summary,
            "generated_at": generated_at,
            "source_order": source_order,
        })
        for row in matrix.get("rows") or []:
            if not isinstance(row, dict):
                continue
            cloned = dict(row)
            cloned["_source_contract_dir"] = str(contract_dir)
            cloned["_source_run_id"] = run_id
            cloned["_source_order"] = source_order
            cloned["_source_generated_at"] = generated_at
            item["rows"].append(cloned)
    for target, item in by_target.items():
        item["rows"] = _dedupe_rows(item["rows"])
        item["contracts"] = sorted(item["contracts"], key=lambda c: (c.get("generated_at") or "", c.get("run_id") or ""))
    return by_target


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _hardware_key(row),
        str(row.get("test_item_id") or ""),
        str(row.get("model_artifact_id") or _model_name(row)),
        _params_identity(row),
    )


def _row_preference(row: dict[str, Any]) -> tuple[int, int, int, int]:
    verdict_rank = VERDICT_ORDER.get(str(row.get("product_verdict")), 0)
    quality_measured = 1 if _quality_score(row) is not None or _quality_reason(row) not in {"not_measured", "not_recorded"} else 0
    latency_measured = 1 if _latency_p95(row) is not None else 0
    source_order = int(row.get("_source_order") or 0)
    return (verdict_rank, quality_measured, latency_measured, source_order)


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        current = selected.get(key)
        if current is None or _row_preference(row) >= _row_preference(current):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda r: (_hardware_key(r), str(r.get("task_class") or ""), _model_name(r), str(r.get("test_item_id") or "")),
    )


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in VERDICT_ORDER}
    for row in rows:
        verdict = str(row.get("product_verdict") or "blocked")
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def _verdict_mix(rows: list[dict[str, Any]]) -> str:
    counts = _counts(rows)
    parts = [f"{verdict}={counts[verdict]}" for verdict in sorted(counts, key=lambda v: VERDICT_ORDER.get(v, 0), reverse=True) if counts[verdict]]
    return ", ".join(parts) if parts else "-"


def _workloads(rows: list[dict[str, Any]]) -> str:
    values = sorted({str(row.get("task_class") or row.get("test_item_id") or "unknown") for row in rows})
    return ", ".join(values) if values else "-"


def _target_status(contracts: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    statuses = {str((contract.get("summary") or {}).get("status") or "complete") for contract in contracts}
    if "invalidated" in statuses:
        return "invalidated"
    if "blocked" in statuses:
        return "blocked"
    if "partial" in statuses or any(str(row.get("product_verdict")) == "blocked" for row in rows):
        return "partial"
    return "complete"


def build_report(contract_dirs: list[Path], targets: list[str] | None = None) -> dict[str, Any]:
    target_set = set(targets) if targets else None
    data = load_contracts(contract_dirs, target_set)
    platforms = []
    for target in sorted(data):
        if target not in PLATFORMS:
            continue
        item = data[target]
        rows = item["rows"]
        contracts = item["contracts"]
        actual_paths = sorted({_hardware_key(row) for row in rows})
        expected_paths = list(PLATFORMS[target]["expected_paths"])
        path_order = expected_paths + [path for path in actual_paths if path not in expected_paths]
        path_summaries = []
        for path_key in path_order:
            path_rows = [row for row in rows if _hardware_key(row) == path_key]
            path_summaries.append({
                "path": path_key,
                "rows": path_rows,
                "row_count": len(path_rows),
                "usable_rows": sum(1 for row in path_rows if str(row.get("product_verdict")) in USABLE_VERDICTS),
                "verdict_mix": _verdict_mix(path_rows),
                "workloads": _workloads(path_rows),
            })
        platforms.append({
            "target": target,
            "meta": PLATFORMS[target],
            "contracts": contracts,
            "rows": rows,
            "status": _target_status(contracts, rows),
            "counts": _counts(rows),
            "path_summaries": path_summaries,
        })
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "platforms": platforms,
    }


def _platform_dir(output_root: Path, platform: dict[str, Any]) -> Path:
    return output_root / str(platform["meta"]["slug"])


def _path_filename(path_key: str, lang: str) -> str:
    meta = HARDWARE.get(path_key, {"file": path_key})
    return f"{meta['file']}.{lang}.md"


def _source_runs(contracts: list[dict[str, Any]]) -> str:
    return ", ".join(f"`{contract['run_id']}`" for contract in contracts) or "-"


def _write_platform_index(platform: dict[str, Any], output_root: Path, lang: str) -> Path:
    meta = platform["meta"]
    title = meta[f"title_{lang}"]
    out_dir = _platform_dir(output_root, platform)
    filename = "index.en.md" if lang == "en" else "index.zh.md"
    peer = "index.zh.md" if lang == "en" else "index.en.md"
    peer_label = "Chinese version" if lang == "en" else "英文版本"
    scope = meta[f"scope_{lang}"]
    report_title = f"# {title}"
    updated = "**Last updated:**" if lang == "en" else "**最后更新：**"
    scope_header = "## Scope" if lang == "en" else "## 范围"
    baseline_header = "## Contract Baseline" if lang == "en" else "## 合同基线"
    path_header = "## Hardware Path Summary" if lang == "en" else "## 硬件路径摘要"
    decision_header = "## Decision" if lang == "en" else "## 结论"
    evidence_header = "## Evidence" if lang == "en" else "## 证据"
    if lang == "en":
        note = (
            "Rows are grouped by `runtime.resource_class` from the contract matrix. "
            "`failed` or `not_recommended` rows are measured evidence, not missing reports. "
            "Rows with no concrete runtime are kept under Blocked Runtime."
        )
        decision = _platform_decision_en(platform)
        path_cols = "| Path | Rows | Usable rows | Workloads | Verdict mix | Report |\n|---|---:|---:|---|---|---|"
        evidence_cols = "| Run ID | Artifacts |\n|---|---|"
        baseline_rows = [
            ("target", platform["target"]),
            ("source_runs", _source_runs(platform["contracts"])),
            ("status", platform["status"]),
            ("row_count", str(len(platform["rows"]))),
        ]
    else:
        note = (
            "行按合同矩阵里的 `runtime.resource_class` 分组。"
            "`failed` 或 `not_recommended` 是已有实测证据，不是报告缺失。"
            "没有进入明确运行时的行保留在阻塞运行时页。"
        )
        decision = _platform_decision_zh(platform)
        path_cols = "| 路径 | 行数 | 可用行 | 工作负载 | Verdict 分布 | 报告 |\n|---|---:|---:|---|---|---|"
        evidence_cols = "| Run ID | 产物 |\n|---|---|"
        baseline_rows = [
            ("target", platform["target"]),
            ("source_runs", _source_runs(platform["contracts"])),
            ("status", platform["status"]),
            ("row_count", str(len(platform["rows"]))),
        ]
    for verdict in sorted(VERDICT_ORDER, key=lambda v: VERDICT_ORDER[v], reverse=True):
        count = platform["counts"].get(verdict, 0)
        if count:
            baseline_rows.append((verdict, str(count)))

    lines = [
        report_title,
        "",
        f"{updated} {_date()}",
        f"**{peer_label}:** [{peer}]({peer})",
        "",
        scope_header,
        "",
        scope,
        "",
        note,
        "",
        baseline_header,
        "",
        "| Item | Value |" if lang == "en" else "| 项目 | 值 |",
        "|---|---|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in baseline_rows)
    lines += ["", path_header, "", path_cols]
    for path_summary in platform["path_summaries"]:
        path_key = path_summary["path"]
        path_meta = HARDWARE.get(path_key, {"en": path_key, "zh": path_key})
        path_label = path_meta.get(lang, path_key)
        link_file = _path_filename(path_key, lang)
        lines.append(
            f"| [{path_label}]({link_file}) | {path_summary['row_count']} | {path_summary['usable_rows']} | "
            f"{_safe(path_summary['workloads'])} | {_safe(path_summary['verdict_mix'])} | [{link_file}]({link_file}) |"
        )
    lines += ["", decision_header, "", decision, "", evidence_header, "", evidence_cols]
    for contract in platform["contracts"]:
        contract_dir = Path(str(contract["dir"]))
        lines.append(f"| `{contract['run_id']}` | {_artifact_links(contract_dir, out_dir, lang)} |")
    _write(out_dir / filename, "\n".join(lines) + "\n")
    return out_dir / filename


def _write_hardware_page(platform: dict[str, Any], path_summary: dict[str, Any], output_root: Path, lang: str) -> Path:
    meta = platform["meta"]
    path_key = path_summary["path"]
    path_meta = HARDWARE.get(path_key, {"file": path_key, "en": path_key, "zh": path_key, "scope_en": "", "scope_zh": ""})
    out_dir = _platform_dir(output_root, platform)
    filename = _path_filename(path_key, lang)
    peer = _path_filename(path_key, "zh" if lang == "en" else "en")
    title = f"{meta[f'title_{lang}']} {path_meta.get(lang, path_key)}"
    peer_label = "Chinese version" if lang == "en" else "英文版本"
    updated = "**Last updated:**" if lang == "en" else "**最后更新：**"
    scope_header = "## Scope" if lang == "en" else "## 范围"
    results_header = "## Workload Results" if lang == "en" else "## 工作负载结果"
    decision_header = "## Decision" if lang == "en" else "## 结论"
    evidence_header = "## Evidence" if lang == "en" else "## 证据"
    source_label = "Contract source runs" if lang == "en" else "合同来源运行"
    if lang == "en":
        table_header = "| Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |\n|---|---|---|---:|---:|---|---|"
        no_rows = "_No current contract rows for this hardware condition._"
        decision = _hardware_decision_en(path_summary)
    else:
        table_header = "| 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |\n|---|---|---|---:|---:|---|---|"
        no_rows = "_当前没有该硬件条件下的合同实测行。_"
        decision = _hardware_decision_zh(path_summary)
    lines = [
        f"# {title}",
        "",
        f"{updated} {_date()}",
        f"**{peer_label}:** [{peer}]({peer})",
        f"**{source_label}:** {_source_runs(platform['contracts'])}",
        "",
        scope_header,
        "",
        path_meta.get(f"scope_{lang}", ""),
        "",
        results_header,
        "",
        table_header,
    ]
    rows = path_summary["rows"]
    if rows:
        for row in rows:
            latency = _fmt_ms(_latency_p95(row))
            score = _fmt_score(_quality_score(row))
            lines.append(
                f"| `{_safe(row.get('task_class') or row.get('test_item_id'))}` | `{_safe(_model_name(row))}` | "
                f"{_safe(_params_brief(row))} | "
                f"{latency} | {score} | `{_safe(row.get('product_verdict'))}` | {_safe(row.get('product_verdict_reason') or _quality_reason(row))} |"
            )
    else:
        lines.append(f"| - | - | - | - | - | - | {no_rows} |")
    lines += ["", decision_header, "", decision, "", evidence_header, ""]
    if lang == "en":
        lines.append("| Run ID | Artifacts |")
    else:
        lines.append("| Run ID | 产物 |")
    lines.append("|---|---|")
    for contract in platform["contracts"]:
        contract_dir = Path(str(contract["dir"]))
        lines.append(f"| `{contract['run_id']}` | {_artifact_links(contract_dir, out_dir, lang)} |")
    _write(out_dir / filename, "\n".join(lines) + "\n")
    return out_dir / filename


def _platform_decision_en(platform: dict[str, Any]) -> str:
    usable = sum(1 for row in platform["rows"] if str(row.get("product_verdict")) in USABLE_VERDICTS)
    not_rec = platform["counts"].get("not_recommended", 0)
    blocked = platform["counts"].get("blocked", 0)
    return (
        f"{usable} contract rows are product-usable under the current verdict policy. "
        f"{not_rec} rows are measured but not recommended, and {blocked} rows remain blocked. "
        "Use the hardware subreports for the concrete path decision instead of mixing CPU, iGPU, NPU, and mixed-runtime evidence."
    )


def _platform_decision_zh(platform: dict[str, Any]) -> str:
    usable = sum(1 for row in platform["rows"] if str(row.get("product_verdict")) in USABLE_VERDICTS)
    not_rec = platform["counts"].get("not_recommended", 0)
    blocked = platform["counts"].get("blocked", 0)
    return (
        f"当前 verdict 口径下有 {usable} 行可作为产品可用证据。"
        f"{not_rec} 行已有实测但不推荐，{blocked} 行仍为 blocked。"
        "具体选型必须看对应硬件子报告，不要混用 CPU、iGPU、NPU 和混合运行时证据。"
    )


def _hardware_decision_en(path_summary: dict[str, Any]) -> str:
    rows = path_summary["rows"]
    if not rows:
        return "This hardware condition has no current contract evidence. It must not be reported as covered."
    usable = path_summary["usable_rows"]
    return (
        f"This hardware condition has {len(rows)} contract rows and {usable} product-usable rows. "
        f"Verdict mix: {path_summary['verdict_mix']}."
    )


def _hardware_decision_zh(path_summary: dict[str, Any]) -> str:
    rows = path_summary["rows"]
    if not rows:
        return "该硬件条件当前没有合同实测证据，不能写成已覆盖。"
    usable = path_summary["usable_rows"]
    return f"该硬件条件有 {len(rows)} 条合同实测行，其中 {usable} 条为产品可用行。Verdict 分布：{path_summary['verdict_mix']}。"


def _write_contract_supplement(platform: dict[str, Any], output_root: Path, lang: str) -> Path:
    meta = platform["meta"]
    out_dir = _platform_dir(output_root, platform)
    filename = "contract.en.md" if lang == "en" else "contract.zh.md"
    peer = "contract.zh.md" if lang == "en" else "contract.en.md"
    peer_label = "Chinese version" if lang == "en" else "英文版本"
    title = f"{meta[f'title_{lang}']} Contract Supplement" if lang == "en" else f"{meta[f'title_{lang}']} 合同补充"
    lines = [
        f"# {title}",
        "",
        f"{'**Last updated:**' if lang == 'en' else '**最后更新：**'} {_date()}",
        f"**{peer_label}:** [{peer}]({peer})",
        "",
        "## Scope" if lang == "en" else "## 范围",
        "",
        meta[f"scope_{lang}"],
        "",
        "## Contract Rows" if lang == "en" else "## 合同行",
        "",
        "| Hardware | Workload | Model/path | Params | p95 latency | Quality score | Verdict | Reason |"
        if lang == "en"
        else "| 硬件 | 工作负载 | 模型/路径 | 参数 | p95 延迟 | 质量分 | Verdict | 原因 |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in platform["rows"]:
        path_key = _hardware_key(row)
        path_label = HARDWARE.get(path_key, {}).get(lang, path_key)
        lines.append(
            f"| {path_label} | `{_safe(row.get('task_class') or row.get('test_item_id'))}` | `{_safe(_model_name(row))}` | "
            f"{_safe(_params_brief(row))} | "
            f"{_fmt_ms(_latency_p95(row))} | {_fmt_score(_quality_score(row))} | "
            f"`{_safe(row.get('product_verdict'))}` | {_safe(row.get('product_verdict_reason') or _quality_reason(row))} |"
        )
    lines += ["", "## Evidence" if lang == "en" else "## 证据", ""]
    lines.append("| Run ID | Artifacts |" if lang == "en" else "| Run ID | 产物 |")
    lines.append("|---|---|")
    for contract in platform["contracts"]:
        contract_dir = Path(str(contract["dir"]))
        lines.append(f"| `{contract['run_id']}` | {_artifact_links(contract_dir, out_dir, lang)} |")
    _write(out_dir / filename, "\n".join(lines) + "\n")
    return out_dir / filename


def write_reports(report: dict[str, Any], output_root: Path) -> list[Path]:
    written: list[Path] = []
    for platform in report["platforms"]:
        if platform["meta"].get("contract_page_only"):
            written.append(_write_contract_supplement(platform, output_root, "en"))
            written.append(_write_contract_supplement(platform, output_root, "zh"))
            continue
        for lang in ("en", "zh"):
            written.append(_write_platform_index(platform, output_root, lang))
            for path_summary in platform["path_summaries"]:
                written.append(_write_hardware_page(platform, path_summary, output_root, lang))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", action="append", type=Path, default=[])
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports" / "platforms")
    args = parser.parse_args(argv)

    contract_dirs = args.contract_dir or discover_contract_dirs(ROOT)
    if not contract_dirs:
        raise SystemExit("no contract directories found")
    report = build_report(contract_dirs=contract_dirs, targets=args.target or None)
    paths = write_reports(report, args.output_root)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
