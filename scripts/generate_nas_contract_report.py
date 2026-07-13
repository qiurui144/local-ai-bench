#!/usr/bin/env python3
"""Generate NAS evaluation-contract artifacts from target runner outputs.

The Windows/Linux runners still emit legacy per-model benchmark reports. This
script converts those reports into the contract artifacts declared in
docs/evaluation-contract.json without inventing measurements: missing metrics
are represented as null with low confidence.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import ModelConfig, _is_chat_capable, load_models  # noqa: E402


CONTRACT_ID = "vlm-llm-nas-evaluation-contract"
CONTRACT_VERSION = 1
TARGETS = {
    "amd-win-x86",
    "intel-win-x86",
    "amd-linux-x86",
    "intel-linux",
    "k3-riscv-16g",
    "k3-riscv-32g",
}
VERDICTS = {
    "sync_default",
    "sync_bounded",
    "async_default",
    "async_only",
    "offline_only",
    "not_recommended",
    "blocked",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    return Path(raw)


def _summary(values: dict[str, Any] | None = None) -> dict[str, float | None]:
    values = values or {}
    return {
        "p50": _num(values.get("p50")),
        "p90": _num(values.get("p90")),
        "p95": _num(values.get("p95")),
        "p99": _num(values.get("p99")),
        "max": _num(values.get("max")),
    }


def _single_summary(value: Any) -> dict[str, float | None]:
    numeric = _num(value)
    return {
        "p50": numeric,
        "p90": numeric,
        "p95": numeric,
        "p99": numeric,
        "max": numeric,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) * q) + 0.999999) - 1))
    return ordered[idx]


def _summary_from_values(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_dict(*items: Any) -> dict[str, Any] | None:
    for item in items:
        if isinstance(item, dict):
            return item
    return None


def _dig(obj: dict[str, Any], *path: str) -> Any:
    cur: Any = obj
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _stats_from_report(report: dict[str, Any], *paths: tuple[str, ...]) -> dict[str, Any] | None:
    for path in paths:
        value = _dig(report, *path)
        if isinstance(value, dict):
            return value
    return None


def _load_models_by_name() -> dict[str, ModelConfig]:
    return {m.name: m for m in load_models(ROOT / "models.yaml")}


def _task_items_for_model(model: ModelConfig) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    caps = set(getattr(model, "capabilities", ()) or ())
    role = getattr(model, "role", "") or ""
    is_vlm = getattr(model, "task_type", "") != "text_only" or "vlm" in role
    if _is_chat_capable(model) and not is_vlm:
        items.append(("llm_chat_boundary", "llm_chat"))
        items.append(("llm_summary_boundary", "llm_summary"))
        items.append(("rag_answer_defaults", "rag_answer"))
    if "embedding" in caps:
        items.append(("embedding_retrieval", "embedding"))
        items.append(("rag_search_only_fallback", "rag_search_only"))
    if "rerank" in caps or "rerank_native" in caps:
        items.append(("reranker_candidates", "reranker"))
    if is_vlm:
        items.append(("vlm_image_qa_boundary", "vlm_qa"))
        items.append(("vlm_document_extract_boundary", "vlm_doc_extract"))
    if "ocr" in caps:
        items.append(("ocr_pages", "ocr"))
    if "asr" in caps:
        items.append(("asr_duration_concurrency", "asr"))
    return items or [("llm_chat_boundary", "llm_chat")]


def _resource_class(model: ModelConfig, row: dict[str, Any]) -> str:
    role = (getattr(model, "role", "") or "").lower()
    provider = (getattr(model, "provider", "") or "").lower()
    if "npu" in role or "xdna" in role or "vitis" in role:
        return "npu"
    if "igpu" in role or "directml" in role or "gpu" in role or provider in {"ollama", "openai"}:
        return "igpu"
    if "cpu" in role:
        return "cpu"
    if row.get("error"):
        return "blocked"
    return "mixed"


def _memory_tier(target: str) -> str:
    if target == "k3-riscv-16g":
        return "k3_16g"
    if target == "k3-riscv-32g":
        return "k3_32g"
    if target in {"amd-win-x86", "intel-win-x86", "amd-linux-x86", "intel-linux"}:
        return "x86_host"
    return "unknown"


def _environment(target: str) -> dict[str, Any]:
    memory_gb: float | None = None
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory_gb = round(status.ullTotalPhys / (1024**3), 2)
        else:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            memory_gb = round((pages * page_size) / (1024**3), 2)
    except Exception:
        memory_gb = None

    return {
        "os": platform.system().lower() or "unknown",
        "arch": platform.machine() or "unknown",
        "kernel_or_build": platform.version() or None,
        "hostname": socket.gethostname(),
        "memory_gb": memory_gb,
        "accelerators": [
            {
                "name": "target_default",
                "provider": target,
                "driver_version": None,
                "available": True,
            }
        ],
        "storage": {
            "system_disk": "unknown",
            "data_disk": "unknown",
            "data_fs": "unknown",
            "direct_io_supported": None,
        },
        "runtime_stack": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def _model_profile(model: ModelConfig) -> dict[str, Any]:
    return {
        "name": model.name,
        "format": getattr(model, "provider", None) or "unknown",
        "quantization": getattr(model, "quantization", None),
        "artifact_size_bytes": None,
        "artifact_hash": getattr(model, "model_id", None) or getattr(model, "hf_repo", None),
        "load_mode": "provider_default",
    }


def _dataset_profile(report: dict[str, Any], task_class: str) -> dict[str, Any]:
    bench = report.get("benchmarks") or {}
    source = "not_measured"
    sample_count: int | None = None
    if task_class == "embedding":
        source = _dig(bench, "embedding", "aggregate", "data_source") or "builtin"
        sample_count = _int_or_none(_dig(bench, "embedding", "aggregate", "num_queries"))
    elif task_class == "reranker":
        source = "builtin"
        sample_count = _int_or_none(_dig(bench, "rerank", "aggregate", "num_queries"))
    elif task_class == "ocr":
        source = "builtin"
        sample_count = _int_or_none(
            _dig(bench, "ocr", "aggregate", "num_samples") or _dig(bench, "ocr", "num_samples") or _dig(bench, "ocr", "samples")
        )
    elif task_class == "asr":
        source = "builtin"
        sample_count = _int_or_none(
            _dig(bench, "asr", "aggregate", "num_samples") or _dig(bench, "asr", "num_samples") or _dig(bench, "asr", "samples")
        )
    elif task_class == "vlm_qa":
        source = "builtin"
        sample_count = _int_or_none(_dig(bench, "accuracy", "aggregate", "total_cases"))
    elif task_class == "vlm_doc_extract":
        source = "builtin"
        sample_count = _int_or_none(_dig(bench, "scenarios", "scenarios", "vlm_document_extraction", "n_cases"))
    return {
        "dataset_id": f"{task_class}.{source}",
        "dataset_version": None,
        "sample_count": sample_count,
        "seed": _dig(report, "full_matrix", "tag"),
    }


def _latency_profile(report: dict[str, Any], task_class: str) -> dict[str, Any]:
    bench = report.get("benchmarks") or {}
    ttft = _stats_from_report(report, ("benchmarks", "ttft", "ttft_ms_stats"))
    total = _first_dict(
        _stats_from_report(report, ("benchmarks", "accuracy", "aggregate", "latency_stats_ms")),
        _stats_from_report(report, ("benchmarks", "ttft", "total_latency_ms_stats")),
        _stats_from_report(report, ("benchmarks", "throughput", "latency_stats_ms")),
        _stats_from_report(report, ("benchmarks", "embedding", "performance", "latency", "single_query_latency_ms_stats")),
        _stats_from_report(report, ("benchmarks", "rerank", "performance", "latency", "single_query_latency_ms_stats")),
        _stats_from_report(report, ("benchmarks", "rerank", "aggregate", "query_rerank_latency_ms_stats")),
        _stats_from_report(report, ("benchmarks", "rerank", "aggregate", "single_pair_latency_ms_stats")),
        _stats_from_report(report, ("benchmarks", "ocr", "aggregate", "latency_ms_stats")),
        _stats_from_report(report, ("benchmarks", "ocr", "latency_ms_stats")),
        _stats_from_report(report, ("benchmarks", "asr", "aggregate", "latency_ms_stats")),
        _stats_from_report(report, ("benchmarks", "asr", "latency_ms_stats")),
    )
    decode = _stats_from_report(report, ("benchmarks", "throughput", "per_request_tps_stats"))
    prefill = _stats_from_report(report, ("benchmarks", "prefill_decode", "prompt_tps_stats"))
    tpot = _stats_from_report(report, ("benchmarks", "prefill_decode", "tpot_ms_stats"))
    if task_class in {"embedding", "reranker", "ocr", "asr"} and not ttft:
        ttft = total
    if not decode:
        decode = {}
    return {
        "ttft_ms": _summary(ttft),
        "prefill_tps": _summary(prefill),
        "decode_tps": _summary(decode),
        "tpot_ms": _summary(tpot),
        "e2e_latency_ms": _summary(total),
    }


def _memory_profile(report: dict[str, Any]) -> dict[str, Any]:
    bench = report.get("benchmarks") or {}
    memory = (
        _dig(bench, "embedding", "performance", "memory")
        or _dig(bench, "rerank", "performance", "memory")
        or _dig(bench, "ocr", "memory")
        or _dig(bench, "asr", "memory")
        or {}
    )
    return {
        "rss_peak_mb": _num(memory.get("rss_peak_mb") or memory.get("peak_rss_mb")),
        "mem_available_min_mb": _num(memory.get("mem_available_min_mb")),
        "cma_free_min_kb": None,
        "tcm_state_before": None,
        "tcm_state_after": None,
        "oom": False,
        "worker_restarted": False,
    }


def _llm_quality_reason(bench: dict[str, Any]) -> str:
    """Return a stable LLM quality reason instead of a bare benchmark verdict."""
    translation = bench.get("translation") if isinstance(bench.get("translation"), dict) else {}
    scenarios = bench.get("scenarios") if isinstance(bench.get("scenarios"), dict) else {}
    general = bench.get("general_ability") if isinstance(bench.get("general_ability"), dict) else {}

    translation_verdict = translation.get("verdict")
    if translation_verdict in {"FAIL", "BLOCKED"}:
        reasons = " ".join(str(r) for r in translation.get("verdict_reasons") or [])
        has_terminology_failure = "term-match" in reasons
        has_quality_failure = "chrF" in reasons or "BLEU" in reasons or "bleu" in reasons
        if has_terminology_failure and has_quality_failure:
            return "translation_quality_and_terminology_failed"
        if has_terminology_failure:
            return "translation_l3_terminology_failed"
        if has_quality_failure:
            return "translation_quality_failed"
        return f"translation_{str(translation_verdict).lower()}"
    if scenarios.get("verdict") in {"FAIL", "BLOCKED"}:
        return f"scenarios_{str(scenarios.get('verdict')).lower()}"
    if general.get("verdict") in {"FAIL", "BLOCKED"}:
        return f"general_ability_{str(general.get('verdict')).lower()}"
    verdicts = [
        general.get("verdict"),
        translation.get("verdict"),
        scenarios.get("verdict"),
    ]
    return ",".join(str(v) for v in verdicts if v) or "not_measured"


def _quality_profile(report: dict[str, Any], task_class: str) -> dict[str, Any]:
    bench = report.get("benchmarks") or {}
    metric_name: str | None = None
    score: float | None = None
    threshold: float | None = None
    passed = False
    reason = "not_measured"
    evidence_status: str | None = None
    if task_class == "embedding":
        score = _num(_dig(bench, "embedding", "aggregate", "ndcg@10"))
        metric_name = "ndcg@10"
        threshold = _num(_dig(bench, "embedding", "thresholds", "recall_at_10_min"))
        verdict = _dig(bench, "embedding", "verdict")
        passed = verdict == "PASS" or (score is not None and score >= 0.75)
        reason = str(verdict or "measured")
    elif task_class == "reranker":
        score = _num(_dig(bench, "rerank", "aggregate", "ndcg@10") or _dig(bench, "rerank", "aggregate", "ndcg@5"))
        metric_name = "ndcg"
        verdict = _dig(bench, "rerank", "verdict")
        passed = verdict == "PASS" or (score is not None and score >= 0.65)
        reason = str(verdict or "measured")
    elif task_class == "rag_search_only":
        emb_score = _num(_dig(bench, "embedding", "aggregate", "ndcg@10"))
        rank_score = _num(_dig(bench, "rerank", "aggregate", "ndcg@10") or _dig(bench, "rerank", "aggregate", "ndcg@5"))
        score = emb_score if emb_score is not None else rank_score
        metric_name = "retrieval_ndcg"
        verdict = _dig(bench, "embedding", "verdict") or _dig(bench, "rerank", "verdict")
        passed = verdict == "PASS" or (score is not None and score >= 0.65)
        reason = str(verdict or "measured")
    elif task_class == "ocr":
        cer = _num(_dig(bench, "ocr", "cer") or _dig(bench, "ocr", "cer_avg") or _dig(bench, "ocr", "aggregate", "cer"))
        score = None if cer is None else max(0.0, 1.0 - cer)
        metric_name = "1-cer"
        verdict = _dig(bench, "ocr", "verdict")
        passed = verdict == "PASS" or (cer is not None and cer <= 0.10)
        reason = str(verdict or "measured")
    elif task_class == "asr":
        cer = _num(_dig(bench, "asr", "cer") or _dig(bench, "asr", "cer_avg") or _dig(bench, "asr", "aggregate", "cer"))
        score = None if cer is None else max(0.0, 1.0 - cer)
        metric_name = "1-cer"
        verdict = _dig(bench, "asr", "verdict")
        passed = verdict == "PASS" or (cer is not None and cer <= 0.18)
        reason = str(verdict or "measured")
    elif task_class.startswith("vlm"):
        if task_class == "vlm_doc_extract":
            doc = _dig(bench, "scenarios", "scenarios", "vlm_document_extraction") or {}
            verdict = doc.get("verdict") or _dig(bench, "scenarios", "verdict")
            score = _num(_dig(doc, "l1", "field_accuracy"))
            metric_name = "field_accuracy"
            passed = verdict in {"PASS", "WARN"} and (score is None or score >= 0.70)
            if verdict == "FAIL" and score is not None and score < 0.70:
                reason = "vlm_document_field_accuracy_failed"
            else:
                reason = str(verdict or "not_measured")
        else:
            verdict = _dig(bench, "accuracy", "verdict") or _dig(bench, "scenarios", "verdict")
            score = _num(
                _dig(bench, "accuracy", "score")
                or _dig(bench, "accuracy", "aggregate", "category_precision")
                or _dig(bench, "accuracy", "aggregate", "entity_recall")
            )
            metric_name = "category_precision"
            passed = verdict == "PASS"
            reasons = " ".join(str(r) for r in (_dig(bench, "accuracy", "verdict_reasons") or []))
            if verdict == "FAIL" and ("entity recall" in reasons or "实体 recall" in reasons):
                reason = "vlm_entity_recall_failed"
            elif verdict == "FAIL" and ("category" in reasons or "分类" in reasons):
                reason = "vlm_category_precision_failed"
            else:
                reason = str(verdict or "not_measured")
        evidence_status = "measured" if score is not None else "not_measured"
    else:
        verdicts = [
            _dig(bench, "general_ability", "verdict"),
            _dig(bench, "translation", "verdict"),
            _dig(bench, "scenarios", "verdict"),
        ]
        bad = [v for v in verdicts if v in {"FAIL", "BLOCKED"}]
        good = [v for v in verdicts if v == "PASS"]
        passed = bool(good) and not bad
        metric_name = "composite_quality"
        score = 1.0 if passed else None
        reason = _llm_quality_reason(bench)
    return {
        "score": score,
        "metric_name": metric_name,
        "threshold": threshold,
        "passed": passed,
        "reason": reason,
        "evidence_status": evidence_status,
    }


def _summary_has_value(item: dict[str, Any]) -> bool:
    return any(item.get(k) is not None for k in ("p50", "p90", "p95", "p99", "max"))


def _product_verdict(report: dict[str, Any], row_error: str | None, task_class: str, quality: dict[str, Any], latency: dict[str, Any]) -> tuple[str, str, str]:
    if row_error:
        return "blocked", row_error, "low"
    if not quality.get("passed"):
        reason = str(quality.get("reason") or "quality_not_passed")
        return ("blocked" if reason == "not_measured" else "not_recommended"), reason, "low"
    e2e = latency.get("e2e_latency_ms") or {}
    p95 = _num(e2e.get("p95"))
    if p95 is None:
        return "async_default", "quality_passed_latency_missing", "low"
    if task_class in {"embedding", "reranker", "ocr", "asr"} and p95 <= 5000:
        return "sync_default", "quality_and_latency_within_tool_budget", "medium"
    if p95 <= 30000:
        return "sync_bounded", "quality_passed_sync_bounded_by_latency", "medium"
    return "async_default", "quality_passed_latency_exceeds_sync_default", "medium"


def _params_for_item(test_item_id: str, report: dict[str, Any]) -> dict[str, Any]:
    condition = report.get("condition") or {}
    defaults: dict[str, Any] = {
        "context_tokens": condition.get("context_tokens"),
        "max_output_tokens": None,
        "startup_state": "warm_process",
    }
    if test_item_id == "embedding_retrieval":
        defaults.update({"embedding_batch_size": None, "query_length_tokens": None, "document_chunk_tokens": None})
    elif test_item_id == "reranker_candidates":
        defaults.update({"rerank_candidate_count": None, "document_chunk_tokens": None})
    elif test_item_id == "rag_answer_defaults":
        defaults.update({
            "chunk_tokens": None,
            "chunk_overlap_tokens": None,
            "retrieve_top_k": None,
            "rerank_top_k": None,
            "evidence_chunks": None,
            "answer_context_budget_tokens": condition.get("context_tokens"),
            "answer_output_tokens": None,
        })
    elif test_item_id.startswith("vlm"):
        defaults.update({"image_count": None, "image_resize_px": None, "vision_token_budget": None})
    elif test_item_id == "ocr_pages":
        defaults.update({"document_pages": None, "image_resize_px": None, "output_schema": None})
    elif test_item_id == "asr_duration_concurrency":
        defaults.update({"audio_duration_s": None, "concurrency": None, "language": None})
    return defaults


def _row_from_result(
    target: str,
    result_row: dict[str, Any],
    model: ModelConfig,
    report: dict[str, Any],
    test_item_id: str,
    task_class: str,
) -> dict[str, Any]:
    row_error = result_row.get("error") or report.get("error")
    latency = _latency_profile(report, task_class)
    quality = _quality_profile(report, task_class)
    verdict, verdict_reason, confidence = _product_verdict(report, row_error, task_class, quality, latency)
    resource_class = _resource_class(model, result_row)
    return {
        "test_item_id": test_item_id,
        "task_class": task_class,
        "priority_class": "interactive",
        "deadline_ms": None,
        "sync_requested": True,
        "hot_set_id": "none",
        "memory_tier": _memory_tier(target),
        "storage_state": "unknown",
        "nas_pressure": "unknown",
        "fallback_allowed": True,
        "model_artifact_id": f"{model.name}@{getattr(model, 'model_id', None) or getattr(model, 'hf_repo', None) or 'local'}",
        "model_profile": _model_profile(model),
        "runtime": {
            "name": getattr(model, "provider", None) or "unknown",
            "version": "unknown",
            "resource_class": resource_class,
        },
        "params": _params_for_item(test_item_id, report),
        "dataset_profile": _dataset_profile(report, task_class),
        "latency_profile": latency,
        "memory_profile": _memory_profile(report),
        "quality_profile": quality,
        "startup_profile": {
            "startup_state": "warm_process",
            "startup_wait_ms": _summary(),
            "model_io_ms": _summary(),
            "storage_state": "unknown",
        },
        "queue_profile": {
            "queue_wait_ms": _summary(),
            "queue_depth_p95": None,
            "deadline_hit_rate": None,
            "async_cutover_reason": None if verdict.startswith("sync") else "latency",
        },
        "resource_profile": {
            "resource_hold_ms": latency["e2e_latency_ms"],
            "resource_class": resource_class,
            "lease_conflict_rate": None,
            "preempted": False,
        },
        "error_profile": {
            "error_class": "none" if not row_error else "runtime_error",
            "retryable": None if not row_error else True,
            "blocked_reason": None if not row_error else str(row_error),
        },
        "product_verdict": verdict,
        "product_verdict_reason": verdict_reason,
        "confidence": confidence,
    }


def _row_preference(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    confidence_rank = {"low": 0, "medium": 1, "high": 2}.get(str(row.get("confidence")), 0)
    quality = row.get("quality_profile") or {}
    latency = row.get("latency_profile") or {}
    e2e = latency.get("e2e_latency_ms") if isinstance(latency.get("e2e_latency_ms"), dict) else {}
    verdict_rank = {
        "blocked": 0,
        "not_recommended": 1,
        "offline_only": 2,
        "async_only": 3,
        "async_default": 4,
        "sync_bounded": 5,
        "sync_default": 6,
    }.get(str(row.get("product_verdict")), 0)
    quality_measured = 1 if quality.get("score") is not None or quality.get("evidence_status") == "measured" else 0
    quality_passed = 1 if quality.get("passed") else 0
    latency_measured = 1 if isinstance(e2e, dict) and _summary_has_value(e2e) else 0
    return (verdict_rank, quality_passed, quality_measured, latency_measured, confidence_rank)


def _dedupe_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["test_item_id"], row["model_artifact_id"])
        current = selected.get(key)
        if current is None or _row_preference(row) >= _row_preference(current):
            selected[key] = row
    return list(selected.values())


def _load_report_for_row(result_row: dict[str, Any]) -> dict[str, Any]:
    report_path = _as_path(result_row.get("report"))
    if report_path and report_path.exists():
        return _read_json(report_path)
    return {
        "model": result_row.get("model"),
        "error": result_row.get("error") or "report_not_found",
        "benchmarks": {},
    }


def _params_hash(params: dict[str, Any]) -> str:
    data = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _generate_parameter_matrix(target: str, run_id: str, rows: list[dict[str, Any]], env: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "generated_at": _now(),
        "run_id": run_id,
        "target": target,
        "environment": env,
        "rows": rows,
    }


def _generate_run_summary(target: str, run_id: str, env: dict[str, Any], matrix_rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    planned = sorted({r["test_item_id"] for r in matrix_rows})
    completed = sorted({r["test_item_id"] for r in matrix_rows if r["product_verdict"] != "blocked"})
    blocked = sorted(set(planned) - set(completed))
    blocked_items = set(blocked)
    counts = {v: 0 for v in sorted(VERDICTS)}
    for row in matrix_rows:
        counts[row["product_verdict"]] = counts.get(row["product_verdict"], 0) + 1
    missing_profiles: list[str] = []
    for row in matrix_rows:
        for profile in ("latency_profile", "memory_profile", "quality_profile", "startup_profile", "queue_profile", "resource_profile"):
            if profile not in row:
                missing_profiles.append(f"{row['test_item_id']}:{profile}")
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "generated_at": _now(),
        "target": target,
        "run_id": run_id,
        "status": "complete" if not blocked else "partial",
        "started_at": None,
        "finished_at": _now(),
        "environment": env,
        "coverage_summary": {
            "planned_test_items": planned,
            "completed_test_items": completed,
            "blocked_test_items": blocked,
            "row_count": len(matrix_rows),
            "missing_required_profiles": missing_profiles,
        },
        "aggregate_verdict_counts": counts,
        "artifact_paths": {
            "parameter_matrix": str(out / "parameter-matrix.json"),
            "verdict_table": str(out / "verdict-table.tsv"),
            "model_profile": str(out / "model-profile.json"),
            "scheduler_contract": str(out / "scheduler-contract.json"),
        },
        "blockers": [
            {
                "test_item_id": row["test_item_id"],
                "reason": row["product_verdict_reason"],
                "retryable": row["error_profile"].get("retryable"),
            }
            for row in matrix_rows
            if row["test_item_id"] in blocked_items and row["product_verdict"] == "blocked"
        ],
    }


def _generate_model_profile(
    target: str,
    run_id: str,
    matrix_rows: list[dict[str, Any]],
    source_runs: list[str] | None = None,
) -> dict[str, Any]:
    first = matrix_rows[0] if matrix_rows else {}
    source_runs = source_runs or [run_id]
    task_profiles: dict[str, Any] = {}
    for row in matrix_rows:
        key = f"{row['test_item_id']}:{row['model_artifact_id']}"
        e2e = row["latency_profile"]["e2e_latency_ms"]
        task_profiles[key] = {
            "test_item_id": row["test_item_id"],
            "task_class": row["task_class"],
            "verdict": row["product_verdict"],
            "max_context_tokens_sync": _int_or_none(row["params"].get("context_tokens")),
            "max_output_tokens_sync": _int_or_none(row["params"].get("max_output_tokens")),
            "sync_deadline_ms": row.get("deadline_ms"),
            "sync_queue_wait_budget_ms": None,
            "sync_resource_hold_budget_ms": None,
            "estimated_runtime_ms": {
                "p50": e2e.get("p50"),
                "p95": e2e.get("p95"),
                "p99": e2e.get("p99"),
                "confidence": row["confidence"],
            },
            "eta_weights": {
                "queue_weight": 1.0,
                "startup_weight": 1.15,
                "prefill_weight": 1.15,
                "decode_weight": 1.25,
                "resource_weight": 1.0,
                "io_weight": 1.1,
                "safety_margin_ms": 1500,
            },
            "quality_profile": row["quality_profile"],
            "failure_profile": row["error_profile"],
            "fallback": {
                "on_timeout": "async_job",
                "on_queue_full": "search_only",
                "on_low_memory": "smaller_model",
            },
        }
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "generated_at": _now(),
        "target": target,
        "source_runs": source_runs,
        "model_artifact_id": first.get("model_artifact_id", "unknown"),
        "model_profile": first.get("model_profile", {
            "name": "unknown",
            "format": "unknown",
            "quantization": None,
            "artifact_size_bytes": None,
            "artifact_hash": None,
            "load_mode": "unknown",
        }),
        "runtime": first.get("runtime", {"name": "unknown", "version": "unknown", "resource_class": "blocked"}),
        "task_profiles": task_profiles,
        "hot_sets": [
            {
                "hot_set_id": f"{target}_default",
                "models": sorted({row["model_artifact_id"] for row in matrix_rows}),
                "resident_policy": "opportunistic",
                "warm_ttl_s": None,
                "resident_gb_budget": None,
                "prompt_cache_mb": None,
                "mutex_groups": [],
                "preemptible": True,
            }
        ],
    }


def _generate_scheduler_contract(
    target: str,
    run_id: str,
    matrix_rows: list[dict[str, Any]],
    source_runs: list[str] | None = None,
) -> dict[str, Any]:
    source_runs = source_runs or [run_id]
    hot_sets = [
        {
            "hot_set_id": f"{target}_default",
            "models": sorted({row["model_artifact_id"] for row in matrix_rows}),
            "resident_policy": "opportunistic",
            "warm_ttl_s": None,
            "resident_gb_budget": None,
            "prompt_cache_mb": None,
            "mutex_groups": [],
            "preemptible": True,
        }
    ]
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "generated_at": _now(),
        "target": target,
        "source_model_profiles": source_runs,
        "endpoints": {
            "/capacity": {
                "hot_sets": hot_sets,
                "queue_depth": None,
                "resource_leases": [],
                "memory_tier": _memory_tier(target),
                "degraded_reason": None,
            },
            "/benchmark/contract": {
                "model_artifacts": sorted({row["model_artifact_id"] for row in matrix_rows}),
                "task_profiles": sorted({row["test_item_id"] for row in matrix_rows}),
                "sync_limits": {},
                "eta_buckets": {},
                "verdicts": {row["test_item_id"]: row["product_verdict"] for row in matrix_rows},
            },
            "/v1/chat/completions": {
                "sync_admission": "profile_required",
                "async_cutover": "eta_or_boundary_exceeded",
                "timeout_reason": "returned_in_error_profile",
                "fallback_reason": "returned_in_error_profile",
            },
            "/kb/tasks": {
                "rag_defaults": {},
                "job_timeout_ms": None,
                "poll_interval_ms": None,
                "cancel_policy": "supported",
            },
        },
        "ui_timeout_profile": {
            "ui_soft_timeout_ms": None,
            "ui_hard_timeout_ms": None,
            "api_hard_timeout_ms": None,
            "proxy_timeout_ms": None,
            "async_poll_initial_ms": None,
            "async_poll_backoff_ms": None,
        },
        "rag_defaults": {
            "chunk_tokens": None,
            "chunk_overlap_tokens": None,
            "retrieve_top_k": None,
            "rerank_top_k": None,
            "evidence_chunks": None,
            "answer_context_budget_tokens": None,
            "answer_output_tokens": None,
        },
        "vlm_sync_boundary": {
            "vlm_sync_max_images": None,
            "vlm_sync_max_pages": None,
            "vlm_sync_max_resize_px": None,
            "vlm_sync_output_schema": None,
            "vlm_async_boundary": "profile_required",
        },
        "hot_set_profile": hot_sets,
    }


def _write_verdict_table(path: Path, matrix_rows: list[dict[str, Any]], target: str) -> None:
    cols = [
        "target",
        "test_item_id",
        "task_class",
        "model_artifact_id",
        "params_hash",
        "latency_p95_ms",
        "memory_peak_mb",
        "quality_score",
        "startup_p95_ms",
        "queue_p95_ms",
        "resource_hold_p95_ms",
        "product_verdict",
        "product_verdict_reason",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in matrix_rows:
            writer.writerow({
                "target": target,
                "test_item_id": row["test_item_id"],
                "task_class": row["task_class"],
                "model_artifact_id": row["model_artifact_id"],
                "params_hash": _params_hash(row["params"]),
                "latency_p95_ms": row["latency_profile"]["e2e_latency_ms"].get("p95"),
                "memory_peak_mb": row["memory_profile"].get("rss_peak_mb"),
                "quality_score": row["quality_profile"].get("score"),
                "startup_p95_ms": row["startup_profile"]["startup_wait_ms"].get("p95"),
                "queue_p95_ms": row["queue_profile"]["queue_wait_ms"].get("p95"),
                "resource_hold_p95_ms": row["resource_profile"]["resource_hold_ms"].get("p95"),
                "product_verdict": row["product_verdict"],
                "product_verdict_reason": row["product_verdict_reason"],
                "confidence": row["confidence"],
            })


def _write_markdown(path: Path, target: str, run_id: str, matrix_rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in matrix_rows:
        counts[row["product_verdict"]] = counts.get(row["product_verdict"], 0) + 1
    lines = [
        f"# NAS Contract Report: {target}",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_at: `{_now()}`",
        f"- contract: `{CONTRACT_ID}@{CONTRACT_VERSION}`",
        "",
        "## Verdict Counts",
        "",
    ]
    for verdict in sorted(VERDICTS):
        lines.append(f"- `{verdict}`: {counts.get(verdict, 0)}")
    lines += [
        "",
        "## Rows",
        "",
        "| test_item_id | task_class | model | verdict | confidence | reason |",
        "|:---|:---|:---|:---|:---|:---|",
    ]
    for row in matrix_rows:
        lines.append(
            f"| `{row['test_item_id']}` | `{row['task_class']}` | `{row['model_artifact_id']}` | "
            f"`{row['product_verdict']}` | `{row['confidence']}` | {row['product_verdict_reason']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifacts(
    out_dir: Path,
    target: str,
    run_id: str,
    source_runs: list[str],
    matrix_rows: list[dict[str, Any]],
    env: dict[str, Any],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "parameter_matrix": out_dir / "parameter-matrix.json",
        "run_summary": out_dir / "run-summary.json",
        "verdict_table": out_dir / "verdict-table.tsv",
        "model_profile": out_dir / "model-profile.json",
        "scheduler_contract": out_dir / "scheduler-contract.json",
        "markdown": out_dir / "nas-contract-report.md",
    }
    _write_json(artifacts["parameter_matrix"], _generate_parameter_matrix(target, run_id, matrix_rows, env))
    _write_json(artifacts["run_summary"], _generate_run_summary(target, run_id, env, matrix_rows, out_dir))
    _write_verdict_table(artifacts["verdict_table"], matrix_rows, target)
    _write_json(artifacts["model_profile"], _generate_model_profile(target, run_id, matrix_rows, source_runs))
    _write_json(artifacts["scheduler_contract"], _generate_scheduler_contract(target, run_id, matrix_rows, source_runs))
    _write_markdown(artifacts["markdown"], target, run_id, matrix_rows)
    return artifacts


def _summary_target(summary: dict[str, Any]) -> str | None:
    manifest = summary.get("manifest") or []
    return next((m.get("target") for m in manifest if isinstance(m, dict) and m.get("target")), None)


def _summary_run_id(summary: dict[str, Any], path: Path) -> str:
    manifest = summary.get("manifest") or []
    return next((m.get("tag") for m in manifest if isinstance(m, dict) and m.get("tag")), None) or path.stem


def _merged_run_id(target: str, source_runs: list[str]) -> str:
    if len(source_runs) == 1:
        return source_runs[0]
    digest = hashlib.sha256("|".join(source_runs).encode("utf-8")).hexdigest()[:10]
    return f"{target}-contract-merged-{digest}"


def generate_from_summaries(
    summary_paths: list[Path],
    out_dir: Path,
    target: str | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    summaries = [_read_json(path) for path in summary_paths]
    detected_targets = [t for t in (_summary_target(summary) for summary in summaries) if t]
    target = target or (detected_targets[0] if detected_targets else None)
    if target not in TARGETS:
        raise SystemExit(f"unknown or missing target: {target!r}")
    mismatched = sorted({t for t in detected_targets if t != target})
    if mismatched:
        raise SystemExit(f"summary target mismatch: expected {target}, got {mismatched}")
    source_runs = [_summary_run_id(summary, path) for summary, path in zip(summaries, summary_paths)]
    run_id = run_id or _merged_run_id(target, source_runs)
    models = _load_models_by_name()
    env = _environment(target)
    matrix_rows: list[dict[str, Any]] = []
    for summary_path, summary in zip(summary_paths, summaries):
        for result_row in summary.get("results") or []:
            name = result_row.get("model")
            if not name:
                continue
            model = models.get(name)
            if not model:
                continue
            report = _load_report_for_row(result_row)
            for test_item_id, task_class in _task_items_for_model(model):
                matrix_rows.append(_row_from_result(target, result_row, model, report, test_item_id, task_class))
        if not (summary.get("results") or []):
            print(f"warning: summary has no results: {summary_path}", file=sys.stderr)
    if not matrix_rows:
        raise SystemExit(f"no matrix rows generated from {[str(p) for p in summary_paths]}")
    matrix_rows = _dedupe_matrix_rows(matrix_rows)
    return _write_artifacts(out_dir, target, run_id, source_runs, matrix_rows, env)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _parse_meminfo_kb(meminfo: str, key: str) -> int | None:
    match = re.search(rf"^{re.escape(key)}:\s+(\d+)\s+kB", meminfo or "", flags=re.M)
    return int(match.group(1)) if match else None


def _k3_environment(preflight: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    uname = str(preflight.get("uname") or "")
    parts = uname.split()
    mem_total_kb = _parse_meminfo_kb(str(preflight.get("meminfo") or ""), "MemTotal")
    memory_gb = round(mem_total_kb / (1024**2), 2) if mem_total_kb else None
    hostname = parts[1] if len(parts) > 1 else "k3"
    kernel = parts[2] if len(parts) > 2 else None
    arch = next((p for p in parts if p in {"riscv64", "x86_64", "aarch64"}), "riscv64")
    tcm_available = "runtime=available" in str(preflight.get("tcm") or "")
    return {
        "os": "linux",
        "arch": arch,
        "kernel_or_build": kernel,
        "hostname": hostname,
        "memory_gb": memory_gb,
        "accelerators": [
            {
                "name": "spacemit-x100-cpu",
                "provider": "cpu",
                "driver_version": None,
                "available": True,
            },
            {
                "name": "a100-ime2-tcm",
                "provider": "tcm",
                "driver_version": None,
                "available": tcm_available,
            },
        ],
        "storage": {
            "system_disk": "emmc",
            "data_disk": "sata_or_model_store",
            "data_fs": "unknown",
            "direct_io_supported": None,
        },
        "runtime_stack": {
            "runner": "scripts/run_k3_32g_realistic_stress.py",
            "llm_ctx": run_config.get("llm_ctx"),
            "llm_contexts": run_config.get("llm_contexts"),
            "scheduler_ready": preflight.get("scheduler_ready"),
            "versions": preflight.get("versions"),
        },
    }


def _parse_k3_resource_profile(path: Path) -> dict[str, Any]:
    rss_values: list[float] = []
    mem_available_values: list[float] = []
    tcm_states: list[str] = []
    for item in _read_jsonl(path):
        stdout = str(item.get("stdout") or "")
        ps_match = re.search(r"^\s*\d+\s+\d+\s+\S+\s+(\d+)\s+\d+\s+", stdout, flags=re.M)
        if ps_match:
            rss_values.append(float(ps_match.group(1)) / 1024.0)
        mem_match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", stdout, flags=re.M)
        if mem_match:
            mem_available_values.append(float(mem_match.group(1)) / 1024.0)
        tcm_match = re.search(r"available_blocks=(\d+/\d+)", stdout)
        if tcm_match:
            tcm_states.append(f"available_blocks={tcm_match.group(1)}")
    return {
        "rss_peak_mb": max(rss_values) if rss_values else None,
        "mem_available_min_mb": min(mem_available_values) if mem_available_values else None,
        "cma_free_min_kb": None,
        "tcm_state_before": tcm_states[0] if tcm_states else None,
        "tcm_state_after": tcm_states[-1] if tcm_states else None,
        "oom": False,
        "worker_restarted": False,
    }


def _k3_model_profile(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_meta = next((row for row in trace_rows if row.get("case") == "models"), {})
    model_data = _dig(model_meta, "models", "data")
    first = model_data[0] if isinstance(model_data, list) and model_data else {}
    meta = first.get("meta") if isinstance(first, dict) else {}
    return {
        "name": str(first.get("id") or "Qwen3-30B-A3B-Q4_0"),
        "format": "gguf",
        "quantization": "Q4_0",
        "artifact_size_bytes": _int_or_none(meta.get("size") if isinstance(meta, dict) else None),
        "artifact_hash": None,
        "load_mode": "mmap",
        "n_ctx": _int_or_none(meta.get("n_ctx") if isinstance(meta, dict) else None),
        "n_params": _int_or_none(meta.get("n_params") if isinstance(meta, dict) else None),
    }


def _k3_quality(case_id: str, item: dict[str, Any]) -> dict[str, Any]:
    ok = bool(item.get("ok"))
    if case_id.startswith("context_"):
        passed = ok and bool(item.get("needle_recall_any"))
        return {
            "score": 1.0 if passed else 0.0,
            "metric_name": "needle_recall",
            "threshold": 1.0,
            "passed": passed,
            "reason": "needle_recall_passed" if passed else "needle_recall_failed",
            "evidence_status": "measured",
        }
    content = str(item.get("content") or item.get("reasoning_content") or "")
    passed = ok and bool(content.strip())
    return {
        "score": 1.0 if passed else 0.0,
        "metric_name": "response_smoke",
        "threshold": 1.0,
        "passed": passed,
        "reason": "response_returned" if passed else "response_missing",
        "evidence_status": "measured",
    }


def _k3_verdict(item: dict[str, Any], quality: dict[str, Any], elapsed_ms: float | None) -> tuple[str, str, str]:
    if not item.get("ok"):
        reason = str(item.get("error") or f"http_status_{item.get('status_code')}")
        error_class = "timeout" if "timeout" in reason.lower() else "runtime_error"
        return "blocked", error_class, "medium"
    if not quality.get("passed"):
        return "not_recommended", str(quality.get("reason") or "quality_failed"), "medium"
    if elapsed_ms is None:
        return "async_default", "quality_passed_latency_missing", "low"
    if elapsed_ms <= 5000:
        return "sync_default", "quality_and_latency_within_default_budget", "medium"
    if elapsed_ms <= 30000:
        return "sync_bounded", "quality_passed_sync_bounded_by_latency", "medium"
    if elapsed_ms <= 300000:
        return "async_default", "quality_passed_latency_requires_async", "medium"
    return "async_only", "quality_passed_latency_exceeds_interactive_async_default", "medium"


def _k3_row_from_trace(
    item: dict[str, Any],
    model_profile: dict[str, Any],
    memory_profile: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any] | None:
    case_id = str(item.get("case") or "")
    if case_id == "stream_ttft":
        test_item_id = "llm_chat_boundary"
        task_class = "llm_chat"
        max_output_tokens = 32
    elif case_id == "decode_128":
        test_item_id = "llm_summary_boundary"
        task_class = "llm_summary"
        max_output_tokens = 128
    elif case_id.startswith("context_"):
        test_item_id = "llm_chat_boundary"
        task_class = "llm_chat"
        max_output_tokens = 32
    else:
        return None
    elapsed_ms = None if item.get("elapsed_s") is None else float(item["elapsed_s"]) * 1000.0
    ttft_ms = None if item.get("ttft_s") is None else float(item["ttft_s"]) * 1000.0
    usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
    completion_tokens = _num(usage.get("completion_tokens"))
    decode_tps = completion_tokens / float(item["elapsed_s"]) if completion_tokens and item.get("elapsed_s") else None
    quality = _k3_quality(case_id, item)
    verdict, verdict_reason, confidence = _k3_verdict(item, quality, elapsed_ms)
    error_class = "none"
    blocked_reason = None
    if not item.get("ok"):
        blocked_reason = str(item.get("error") or f"http_status_{item.get('status_code')}")
        error_class = "timeout" if "timeout" in blocked_reason.lower() else "runtime_error"
    target_context = _int_or_none(item.get("target_context_tokens") or usage.get("prompt_tokens"))
    latency = {
        "ttft_ms": _single_summary(ttft_ms),
        "prefill_tps": _summary(),
        "decode_tps": _single_summary(decode_tps),
        "tpot_ms": _summary(),
        "e2e_latency_ms": _single_summary(elapsed_ms),
    }
    model_artifact_id = f"{model_profile['name']}@gguf-q4_0"
    return {
        "test_item_id": test_item_id,
        "task_class": task_class,
        "priority_class": "interactive",
        "deadline_ms": 30000,
        "sync_requested": True,
        "hot_set_id": "k3_qwen30b_default",
        "memory_tier": "k3_32g",
        "storage_state": "unknown",
        "nas_pressure": "idle",
        "fallback_allowed": True,
        "model_artifact_id": model_artifact_id,
        "model_profile": model_profile,
        "runtime": {
            "name": "llama.cpp",
            "version": "unknown",
            "resource_class": "x100_cpu",
            "server_ctx": run_config.get("llm_ctx"),
        },
        "params": {
            "context_tokens": target_context,
            "max_output_tokens": max_output_tokens,
            "startup_state": "warm_process",
            "target_context_tokens": _int_or_none(item.get("target_context_tokens")),
            "finish_reason": item.get("finish_reason"),
        },
        "dataset_profile": {
            "dataset_id": f"k3_realistic.{case_id}",
            "dataset_version": None,
            "sample_count": 1,
            "seed": run_config.get("remote_run_dir") or run_config.get("out_dir"),
        },
        "latency_profile": latency,
        "memory_profile": memory_profile,
        "quality_profile": quality,
        "startup_profile": {
            "startup_state": "warm_process",
            "startup_wait_ms": _summary(),
            "model_io_ms": _summary(),
            "storage_state": "unknown",
        },
        "queue_profile": {
            "queue_wait_ms": _summary(),
            "queue_depth_p95": None,
            "deadline_hit_rate": None,
            "async_cutover_reason": None if verdict.startswith("sync") else "latency",
        },
        "resource_profile": {
            "resource_hold_ms": latency["e2e_latency_ms"],
            "resource_class": "x100_cpu",
            "lease_conflict_rate": None,
            "preempted": False,
        },
        "error_profile": {
            "error_class": error_class,
            "retryable": None if error_class == "none" else True,
            "blocked_reason": blocked_reason,
        },
        "product_verdict": verdict,
        "product_verdict_reason": verdict_reason,
        "confidence": confidence,
    }


def generate_from_k3_realistic_dir(
    k3_dir: Path,
    out_dir: Path,
    target: str | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    target = target or "k3-riscv-32g"
    if target not in TARGETS:
        raise SystemExit(f"unknown or missing target: {target!r}")
    run_id = run_id or k3_dir.name
    run_config = _read_json_if_exists(k3_dir / "run-config.json")
    preflight = _read_json_if_exists(k3_dir / "preflight.json")
    trace_rows = _read_jsonl(k3_dir / "trace.jsonl")
    model_profile = _k3_model_profile(trace_rows)
    memory_profile = _parse_k3_resource_profile(k3_dir / "resource" / "llm-qwen3-30b.jsonl")
    env = _k3_environment(preflight, run_config)
    matrix_rows = [
        row
        for row in (_k3_row_from_trace(item, model_profile, memory_profile, run_config) for item in trace_rows)
        if row is not None
    ]
    if not matrix_rows:
        raise SystemExit(f"no K3 matrix rows generated from {k3_dir}")
    return _write_artifacts(out_dir, target, run_id, [run_id], matrix_rows, env)


def generate(summary_paths: list[Path] | Path, out_dir: Path, target: str | None = None, run_id: str | None = None) -> dict[str, Path]:
    if isinstance(summary_paths, Path):
        summary_paths = [summary_paths]
    return generate_from_summaries(summary_paths, out_dir, target, run_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary", nargs="+", type=Path)
    source.add_argument("--k3-realistic-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target", choices=sorted(TARGETS))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.k3_realistic_dir:
        artifacts = generate_from_k3_realistic_dir(args.k3_realistic_dir, args.output_dir, args.target, args.run_id)
    else:
        artifacts = generate_from_summaries(args.summary, args.output_dir, args.target, args.run_id)
    for key, value in artifacts.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
