#!/usr/bin/env python3
"""Generate model-adjusted quality diagnosis from raw benchmark reports."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import load_models  # noqa: E402


SCHEMA_VERSION = "1.0"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _dig(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _model_name(report: dict[str, Any], path: Path) -> str:
    name = report.get("model")
    if name:
        return str(name)
    for block in (report.get("benchmarks") or {}).values():
        if isinstance(block, dict) and block.get("model"):
            return str(block["model"])
    return path.stem


def _parameter_size_b(model_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    return float(match.group(1)) if match else None


def _contract_rows(contract_dir: Path | None) -> list[dict[str, Any]]:
    if not contract_dir:
        return []
    path = contract_dir / "parameter-matrix.json"
    if not path.exists():
        return []
    data = _read_json(path)
    return list(data.get("rows") or [])


def _contract_model(row: dict[str, Any]) -> str:
    profile = row.get("model_profile") if isinstance(row.get("model_profile"), dict) else {}
    return str(profile.get("name") or row.get("model_artifact_id") or "")


def _model_inventory(models_yaml: Path, target: str | None) -> dict[str, dict[str, Any]]:
    try:
        models = load_models(models_yaml)
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for model in models:
        if target and model.target != target:
            continue
        out[model.name] = {
            "target": model.target,
            "provider": model.provider,
            "role": model.role,
            "task_type": model.task_type,
            "notes": model.notes,
            "hardware_min": model.hardware_min,
        }
    return out


def _contract_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        quality = row.get("quality_profile") if isinstance(row.get("quality_profile"), dict) else {}
        latency = row.get("latency_profile") if isinstance(row.get("latency_profile"), dict) else {}
        e2e = latency.get("e2e_latency_ms") if isinstance(latency.get("e2e_latency_ms"), dict) else {}
        out.append({
            "test_item_id": row.get("test_item_id"),
            "task_class": row.get("task_class"),
            "product_verdict": row.get("product_verdict"),
            "product_verdict_reason": row.get("product_verdict_reason"),
            "quality_metric": quality.get("metric_name"),
            "quality_score": quality.get("score"),
            "quality_reason": quality.get("reason"),
            "latency_p95_ms": e2e.get("p95"),
            "resource_class": _dig(row, "runtime", "resource_class"),
        })
    return out


def _metric(
    *,
    dimension: str,
    metric: str,
    observed: Any,
    threshold: Any = None,
    verdict: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "metric": metric,
        "observed": observed,
        "threshold": threshold,
        "verdict": verdict,
        "detail": detail,
    }


def _translation_metrics(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    tr = _dig(report, "benchmarks", "translation") or {}
    metrics: list[dict[str, Any]] = []
    failures: list[str] = []
    flags = {
        "translation_verdict": tr.get("verdict"),
        "l1_pass": False,
        "l1_failed": False,
        "l3_failed": False,
        "min_l3_term_match": None,
        "max_l1_shortfall": None,
        "builtin_l1_sources": False,
    }
    l1_verdicts: list[str] = []
    l3_term_rates: list[float] = []
    l1_shortfalls: list[float] = []

    for direction, block in sorted((tr.get("directions") or {}).items()):
        if not isinstance(block, dict):
            continue
        for level_name, result in sorted(block.items()):
            if not isinstance(result, dict):
                continue
            aggregate = result.get("aggregate") if isinstance(result.get("aggregate"), dict) else {}
            verdict = str(result.get("verdict") or "")
            data_source = aggregate.get("data_source")
            level = str(aggregate.get("level") or level_name)
            prefix = f"translation.{direction}.{level}"
            bleu = _num(aggregate.get("bleu"))
            chrf = _num(aggregate.get("chrf"))
            term = _num(_dig(aggregate, "terminology", "term_match_rate"))
            metrics.append(_metric(
                dimension=prefix,
                metric="bleu",
                observed=bleu,
                threshold=None,
                verdict=verdict,
                detail=f"pairs={aggregate.get('num_pairs')}; source={data_source}",
            ))
            metrics.append(_metric(
                dimension=prefix,
                metric="chrf",
                observed=chrf,
                threshold=40.0 if level == "l3" and direction == "en->zh" else None,
                verdict=verdict,
                detail=f"pairs={aggregate.get('num_pairs')}; source={data_source}",
            ))
            if term is not None:
                l3_term_rates.append(term)
                metrics.append(_metric(
                    dimension=prefix,
                    metric="term_match_rate",
                    observed=term,
                    threshold=0.80,
                    verdict=verdict,
                    detail=f"matched={_dig(aggregate, 'terminology', 'matched_terms')}; total={_dig(aggregate, 'terminology', 'total_terms')}",
                ))
            if level_name == "l1_flores":
                l1_verdicts.append(verdict)
                if verdict == "FAIL":
                    flags["l1_failed"] = True
                if data_source == "builtin":
                    flags["builtin_l1_sources"] = True
            if level_name == "l3_terminology" and verdict == "FAIL":
                flags["l3_failed"] = True
            for reason in result.get("verdict_reasons") or []:
                match = re.search(r"<\s*(\d+(?:\.\d+)?)", str(reason))
                observed = re.search(r"(?:chrF|BLEU|bleu)\s+(\d+(?:\.\d+)?)", str(reason))
                if level_name == "l1_flores" and match and observed:
                    l1_shortfalls.append(float(match.group(1)) - float(observed.group(1)))
                failures.append(f"{direction}/{level}: {reason}")

    for direction, perf in sorted((_dig(tr, "performance", "directions") or {}).items()):
        if not isinstance(perf, dict):
            continue
        ttft = perf.get("ttft") if isinstance(perf.get("ttft"), dict) else {}
        throughput = perf.get("throughput") if isinstance(perf.get("throughput"), dict) else {}
        metrics.append(_metric(
            dimension=f"translation.{direction}.runtime",
            metric="ttft_error_rate",
            observed=_num(ttft.get("error_rate")),
            threshold=0.0,
            verdict="FAIL" if _num(ttft.get("error_rate")) else "PASS",
            detail=f"errors={ttft.get('errors')}; samples={ttft.get('samples')}",
        ))
        metrics.append(_metric(
            dimension=f"translation.{direction}.runtime",
            metric="aggregate_tps",
            observed=_num(throughput.get("aggregate_tps")),
            threshold=None,
            verdict=None,
            detail=f"requests={throughput.get('requests')}; errors={throughput.get('errors')}",
        ))

    flags["l1_pass"] = bool(l1_verdicts) and all(v == "PASS" for v in l1_verdicts)
    flags["min_l3_term_match"] = min(l3_term_rates) if l3_term_rates else None
    flags["max_l1_shortfall"] = max(l1_shortfalls) if l1_shortfalls else None
    for reason in tr.get("verdict_reasons") or []:
        if "FAIL:" in str(reason):
            failures.append(str(reason))
    return metrics, sorted(set(failures)), flags


def _general_metrics(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    general = _dig(report, "benchmarks", "general_ability") or {}
    metrics: list[dict[str, Any]] = []
    failures: list[str] = []
    flags = {"general_verdict": general.get("verdict"), "general_pass": general.get("verdict") == "PASS"}
    for task, block in sorted((general.get("tasks") or {}).items()):
        if not isinstance(block, dict):
            continue
        metrics.append(_metric(
            dimension=f"general_ability.{task}",
            metric="accuracy",
            observed=_num(block.get("accuracy")),
            threshold=None,
            verdict=block.get("verdict"),
            detail=f"n={block.get('n')}; errors={block.get('errors')}",
        ))
        for reason in block.get("verdict_reasons") or []:
            failures.append(f"{task}: {reason}")
    for reason in general.get("verdict_reasons") or []:
        failures.append(str(reason))
    return metrics, sorted(set(failures)), flags


def _vlm_metrics(reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    failures: list[str] = []
    flags = {
        "category_precision": None,
        "entity_recall": None,
        "fact_recall": None,
        "doc_field_accuracy": None,
    }
    for report in reports:
        acc = _dig(report, "benchmarks", "accuracy")
        if isinstance(acc, dict):
            aggregate = acc.get("aggregate") if isinstance(acc.get("aggregate"), dict) else {}
            category_precision = _num(aggregate.get("category_precision"))
            entity_recall = _num(aggregate.get("entity_recall"))
            fact_recall = _num(aggregate.get("fact_recall"))
            flags["category_precision"] = category_precision
            flags["entity_recall"] = entity_recall
            flags["fact_recall"] = fact_recall
            metrics.extend([
                _metric(dimension="vlm.image_qa", metric="category_precision", observed=category_precision, threshold=0.80, verdict=acc.get("verdict")),
                _metric(dimension="vlm.image_qa", metric="entity_recall", observed=entity_recall, threshold=0.60, verdict=acc.get("verdict")),
                _metric(dimension="vlm.image_qa", metric="fact_recall", observed=fact_recall, threshold=None, verdict=acc.get("verdict")),
                _metric(dimension="vlm.image_qa", metric="latency_p95_ms", observed=_dig(aggregate, "latency_stats_ms", "p95"), threshold=30000, verdict=acc.get("verdict")),
            ])
            for reason in acc.get("verdict_reasons") or []:
                failures.append(f"accuracy: {reason}")
        doc = _dig(report, "benchmarks", "scenarios", "scenarios", "vlm_document_extraction")
        if isinstance(doc, dict):
            field_accuracy = _num(_dig(doc, "l1", "field_accuracy"))
            flags["doc_field_accuracy"] = field_accuracy
            metrics.append(_metric(
                dimension="vlm.document_extraction",
                metric="field_accuracy",
                observed=field_accuracy,
                threshold=0.75,
                verdict=doc.get("verdict"),
                detail=f"cases={doc.get('n_cases')}; provenance={doc.get('provenance')}",
            ))
            for reason in doc.get("verdict_reasons") or []:
                failures.append(f"vlm_document_extraction: {reason}")
    return metrics, sorted(set(failures)), flags


def _diagnose_llm(model: str, reports: list[dict[str, Any]]) -> tuple[str, str, str, list[str], list[str], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    failures: list[str] = []
    flags: dict[str, Any] = {}
    for report in reports:
        m, f, fl = _translation_metrics(report)
        metrics.extend(m)
        failures.extend(f)
        flags.update(fl)
        m, f, fl = _general_metrics(report)
        metrics.extend(m)
        failures.extend(f)
        flags.update(fl)

    size_b = _parameter_size_b(model)
    if flags.get("general_pass") and flags.get("l1_failed") and flags.get("l3_failed"):
        term_match = flags.get("min_l3_term_match")
        if size_b is not None and size_b >= 3.0 and (term_match is None or term_match >= 0.75):
            standard = "general_llm_candidate_translation_and_terminology_caveat"
            status = "candidate_with_quality_caveat"
            recommendation = "candidate_for_general_llm_with_translation_and_terminology_caveat"
            bottlenecks = [
                "General ability passes, but L1 translation quality and L3 terminology gates both fail",
                f"L3 terminology term-match min={_pct(term_match)}, below 80.0%",
                "Use as a general LLM candidate only; do not certify bidirectional translation or terminology-sensitive RAG until retest passes",
            ]
            resolution = (
                "Keep the strict translation and terminology gates visible in the contract. "
                "For production translation/RAG certification, retest with a larger corpus and/or glossary-constrained decoding."
            )
        else:
            standard = "small_llm_translation_and_terminology_limited"
            status = "limited_suitability"
            recommendation = "not_recommended_for_translation_or_terminology_rag"
            bottlenecks = [
                "General ability passes, but translation quality and terminology preservation fail for this declared role",
                f"L3 terminology term-match min={_pct(term_match)}, below 80.0%",
            ]
            resolution = (
                "Do not relax the production gate; limit the model to non-translation smoke/general-chat evidence or retest with a stronger model."
            )
    elif flags.get("general_pass") and not flags.get("l3_failed") and flags.get("l1_failed"):
        shortfall = flags.get("max_l1_shortfall")
        if shortfall is not None and shortfall <= 1.0:
            standard = "general_llm_candidate_translation_l1_caveat"
            status = "candidate_with_quality_caveat"
            recommendation = "candidate_for_general_llm_not_translation_certified"
            bottlenecks = [
                f"General ability and L3 terminology pass, but one L1 translation metric is below threshold by {shortfall:.1f}",
                "Use for general chat/RAG candidate evaluation; do not certify as strict bidirectional translation without retest or calibrated threshold",
            ]
            resolution = (
                "Keep the strict translation gate visible in the contract, but classify the model-adjusted standard as a general LLM candidate. "
                "For translation certification, rerun a larger sample or calibrate the en->zh chrF threshold with the accepted corpus."
            )
        else:
            standard = "general_llm_candidate_translation_quality_caveat"
            status = "candidate_with_quality_caveat"
            recommendation = "candidate_for_general_llm_with_translation_caveat"
            bottlenecks = [
                "General ability passes, but L1 translation quality has a failing gate",
            ]
            resolution = "Keep the translation caveat and certify only non-translation LLM usage until retest."
    elif flags.get("l1_pass") and flags.get("l3_failed"):
        term_match = flags.get("min_l3_term_match")
        if size_b is not None and size_b >= 3.0 and term_match is not None and term_match >= 0.75:
            standard = "general_llm_candidate_terminology_caveat"
            status = "candidate_with_quality_caveat"
            recommendation = "candidate_for_general_llm_with_terminology_caveat"
            bottlenecks = [
                f"L1 translation passes, but L3 terminology term-match min={_pct(term_match)}, below 80.0%",
                "Terminology gate is close to threshold; keep the caveat until a larger sample or stricter glossary-constrained decoding passes",
            ]
            if flags.get("general_verdict") == "BLOCKED":
                bottlenecks.append("General ability was BLOCKED by missing/rejected benchmark data, so this run cannot certify general reasoning quality")
            resolution = (
                "Classify as an LLM candidate with terminology caveat, not as a production-certified terminology/RAG model. "
                "Rerun with official general-ability datasets and the full translation corpus before promotion."
            )
        elif size_b is not None and size_b <= 0.8:
            standard = "micro_llm_smoke_or_latency_only"
            status = "limited_suitability"
            recommendation = "not_recommended_for_production_llm_or_rag"
            bottlenecks = [
                f"L3 terminology term-match min={_pct(flags.get('min_l3_term_match'))}, below 80.0%",
                "Model repeats glossary/template fragments on technical prompts",
                "Use only as OpenVINO micro-model smoke/performance evidence",
            ]
        else:
            standard = "small_llm_basic_translation_only"
            status = "limited_suitability"
            recommendation = "not_recommended_for_terminology_rag_answer"
            bottlenecks = [
                f"L1 basic translation passes, but L3 terminology term-match min={_pct(flags.get('min_l3_term_match'))}, below 80.0%",
                "Prompt adherence and terminology preservation are insufficient for RAG answer quality",
            ]
        resolution = (
            "Keep production terminology/RAG gate unchanged; classify this model under the adjusted standard above. "
            "For production LLM/RAG quality, retest with a larger instruction/translation model or add domain-constrained decoding/post-processing."
        )
    elif flags.get("general_pass") and flags.get("translation_verdict") in {"FAIL", "BLOCKED"}:
        standard = "general_llm_candidate_translation_quality_caveat"
        status = "candidate_with_quality_caveat"
        recommendation = "candidate_for_general_llm_with_translation_caveat"
        bottlenecks = [
            "General ability passes, but translation quality has a failing or blocked gate",
        ]
        resolution = "Keep the translation caveat and certify only non-translation LLM usage until retest."
    elif reports:
        standard = "production_llm_quality_gate"
        translation_failed = flags.get("translation_verdict") in {"FAIL", "BLOCKED"} or flags.get("l1_failed") or flags.get("l3_failed")
        if (any(_dig(r, "benchmarks", "translation", "verdict") == "PASS" for r in reports) or flags.get("general_pass")) and not translation_failed:
            status = "quality_passed"
            recommendation = "candidate"
            bottlenecks = []
            resolution = "No LLM quality bottleneck detected in provided raw reports."
        else:
            status = "not_suitable"
            recommendation = "not_recommended_for_production_llm_or_rag"
            bottlenecks = failures or ["No passing LLM quality evidence"]
            resolution = "Do not relax the production gate; replace or retest with a stronger model."
    else:
        standard = "unstable_runtime_no_quality_evidence"
        status = "runtime_blocked"
        recommendation = "not_recommended_until_runtime_stable"
        bottlenecks = ["Contract rows are blocked and no raw quality report is available"]
        resolution = "Stabilize the runtime first, then rerun LLM quality dimensions."
    return status, standard, recommendation, bottlenecks, sorted(set(failures)), metrics


def _diagnose_vlm(reports: list[dict[str, Any]]) -> tuple[str, str, str, list[str], list[str], list[dict[str, Any]]]:
    metrics, failures, flags = _vlm_metrics(reports)
    category = flags.get("category_precision")
    entity = flags.get("entity_recall")
    fact = flags.get("fact_recall")
    field = flags.get("doc_field_accuracy")
    if category is not None and category >= 0.80 and ((entity is not None and entity < 0.60) or (field is not None and field < 0.75)):
        status = "limited_suitability"
        standard = "coarse_vlm_classification_only"
        recommendation = "not_recommended_for_evidence_or_document_extraction"
        bottlenecks = [
            f"Category precision is usable at {_pct(category)}, but entity recall is {_pct(entity)} and fact recall is {_pct(fact)}",
            f"Document field accuracy is {_pct(field)}, below 75.0%",
            "Model describes screenshots generically and misses exact entities/facts/fields",
        ]
        resolution = (
            "Keep document/evidence extraction gates unchanged; route document tasks to OCR plus LLM post-processing, "
            "or replace with a stronger document-capable VLM before recommending this path."
        )
    elif reports:
        status = "not_suitable"
        standard = "production_vlm_quality_gate"
        recommendation = "not_recommended_for_vlm_quality"
        bottlenecks = failures or ["No passing VLM quality evidence"]
        resolution = "Retest with a stronger VLM or reduce the declared model role to smoke coverage."
    else:
        status = "runtime_blocked"
        standard = "unstable_runtime_no_quality_evidence"
        recommendation = "not_recommended_until_runtime_stable"
        bottlenecks = ["No raw VLM quality report is available"]
        resolution = "Stabilize the runtime first, then rerun VLM quality dimensions."
    return status, standard, recommendation, bottlenecks, failures, metrics


def _family(model: str, reports: list[dict[str, Any]], rows: list[dict[str, Any]], inventory: dict[str, Any]) -> str:
    if any(_dig(r, "benchmarks", "translation") for r in reports):
        return "llm"
    if any(_dig(r, "benchmarks", "accuracy") or _dig(r, "benchmarks", "scenarios", "scenarios", "vlm_document_extraction") for r in reports):
        return "vlm"
    if any(str(row.get("task_class") or "").startswith("vlm_") for row in rows):
        return "vlm"
    if any(str(row.get("task_class") or "").startswith("llm_") or row.get("task_class") == "rag_answer" for row in rows):
        return "llm"
    return "vlm" if inventory.get("task_type") == "vlm" else "llm"


def build_report(
    *,
    raw_reports: list[Path],
    contract_dir: Path | None,
    models_yaml: Path,
    target: str | None,
    run_id: str,
) -> dict[str, Any]:
    inventory = _model_inventory(models_yaml, target)
    rows = _contract_rows(contract_dir)
    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task_class") or "")
        if task.startswith("llm_") or task == "rag_answer" or task.startswith("vlm_"):
            rows_by_model[_contract_model(row)].append(row)

    reports_by_model: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in raw_reports:
        report = _read_json(path)
        reports_by_model[_model_name(report, path)].append((path, report))

    model_names = set(reports_by_model) | set(rows_by_model)
    diagnostics: list[dict[str, Any]] = []
    for model in sorted(model_names):
        report_items = reports_by_model.get(model, [])
        reports = [r for _, r in report_items]
        model_rows = rows_by_model.get(model, [])
        info = inventory.get(model, {})
        family = _family(model, reports, model_rows, info)
        if family == "vlm":
            status, standard, recommendation, bottlenecks, failures, metrics = _diagnose_vlm(reports)
        else:
            status, standard, recommendation, bottlenecks, failures, metrics = _diagnose_llm(model, reports)
        if not reports and any(row.get("product_verdict") == "blocked" for row in model_rows):
            status = "runtime_blocked"
            standard = "unstable_runtime_no_quality_evidence"
            recommendation = "not_recommended_until_runtime_stable"
            bottlenecks = sorted({str(row.get("product_verdict_reason") or _dig(row, "error_profile", "blocked_reason") or "blocked") for row in model_rows})
            failures = bottlenecks
            metrics = []
        diagnostics.append({
            "model": model,
            "target": info.get("target") or target,
            "task_family": family,
            "provider": info.get("provider"),
            "role": info.get("role"),
            "quality_status": status,
            "adjusted_quality_standard": standard,
            "production_recommendation": recommendation,
            "bottlenecks": bottlenecks,
            "failed_gates": failures,
            "observed_metrics": metrics,
            "contract_rows": _contract_summary(model_rows),
            "evidence_files": [_rel(path) for path, _ in report_items],
            "model_notes": info.get("notes"),
        })

    status_counts = Counter(d["quality_status"] for d in diagnostics)
    limited = [
        d
        for d in diagnostics
        if d["quality_status"] in {"candidate_with_quality_caveat", "limited_suitability", "runtime_blocked", "not_suitable"}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": _now(),
        "target": target,
        "contract_dir": _rel(contract_dir) if contract_dir else None,
        "models_yaml": _rel(models_yaml),
        "overall_status": "diagnosed_with_model_adjusted_standards",
        "quality_status": "caveats" if limited else "pass",
        "standards_decision": {
            "production_quality_gate": "unchanged",
            "model_adjusted_reporting": "enabled",
            "threshold_policy": "Do not lower production gates to make unsuitable models pass; classify limited models by intended use and bottleneck.",
        },
        "summary": {
            "model_count": len(diagnostics),
            "status_counts": dict(sorted(status_counts.items())),
            "production_recommended_models": [d["model"] for d in diagnostics if d["production_recommendation"] == "candidate"],
            "limited_or_blocked_models": [d["model"] for d in limited],
            "conclusion": (
                f"{target or 'Selected target'} LLM/VLM coverage is present. Model-adjusted diagnosis is required before production use "
                "because at least one measured model has quality caveats for translation, RAG, evidence extraction, or document field extraction."
            ),
        },
        "model_diagnostics": diagnostics,
    }


def _write_tsv(path: Path, report: dict[str, Any]) -> None:
    cols = [
        "target",
        "model",
        "task_family",
        "quality_status",
        "adjusted_quality_standard",
        "production_recommendation",
        "bottlenecks",
        "failed_gates",
        "evidence_files",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in report["model_diagnostics"]:
            writer.writerow({
                "target": row.get("target"),
                "model": row["model"],
                "task_family": row["task_family"],
                "quality_status": row["quality_status"],
                "adjusted_quality_standard": row["adjusted_quality_standard"],
                "production_recommendation": row["production_recommendation"],
                "bottlenecks": " ; ".join(row["bottlenecks"]),
                "failed_gates": " ; ".join(row["failed_gates"]),
                "evidence_files": " ; ".join(row["evidence_files"]),
            })


def _metric_text(metric: dict[str, Any]) -> str:
    observed = metric.get("observed")
    if isinstance(observed, float) and metric.get("metric", "").endswith("rate"):
        observed_text = _pct(observed)
    elif isinstance(observed, float) and metric.get("metric") in {"category_precision", "entity_recall", "fact_recall", "field_accuracy"}:
        observed_text = _pct(observed)
    else:
        observed_text = "-" if observed is None else f"{observed}"
    threshold = metric.get("threshold")
    if isinstance(threshold, float) and threshold <= 1.0:
        threshold_text = _pct(threshold)
    else:
        threshold_text = "-" if threshold is None else f"{threshold}"
    return f"{metric['dimension']} `{metric['metric']}`={observed_text} / threshold={threshold_text}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Quality Diagnosis: {report['target']}",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- overall_status: `{report['overall_status']}`",
        f"- quality_status: `{report['quality_status']}`",
        f"- production_quality_gate: `{report['standards_decision']['production_quality_gate']}`",
        "",
        "## Summary",
        "",
        report["summary"]["conclusion"],
        "",
        "| model | family | adjusted standard | status | recommendation |",
        "|---|---|---|---|---|",
    ]
    for item in report["model_diagnostics"]:
        lines.append(
            f"| `{item['model']}` | `{item['task_family']}` | `{item['adjusted_quality_standard']}` | "
            f"`{item['quality_status']}` | `{item['production_recommendation']}` |"
        )
    for item in report["model_diagnostics"]:
        lines += [
            "",
            f"## {item['model']}",
            "",
            f"- role: `{item.get('role') or '-'}`",
            f"- adjusted_quality_standard: `{item['adjusted_quality_standard']}`",
            f"- production_recommendation: `{item['production_recommendation']}`",
            f"- evidence_files: {', '.join(f'`{p}`' for p in item['evidence_files']) or '-'}",
            "",
            "Bottlenecks:",
        ]
        for bottleneck in item["bottlenecks"]:
            lines.append(f"- {bottleneck}")
        lines += ["", "Key Metrics:"]
        for metric in item["observed_metrics"][:12]:
            lines.append(f"- {_metric_text(metric)}")
        if item["failed_gates"]:
            lines += ["", "Failed Gates:"]
            for failure in item["failed_gates"][:12]:
                lines.append(f"- {failure}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "quality-diagnosis.json",
        "tsv": output_dir / "quality-diagnosis.tsv",
        "markdown": output_dir / "quality-diagnosis.md",
    }
    _write_json(paths["json"], report)
    _write_tsv(paths["tsv"], report)
    _write_markdown(paths["markdown"], report)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report", action="append", type=Path, default=[], help="Raw benchmark JSON; repeatable")
    parser.add_argument("--raw-dir", action="append", type=Path, default=[], help="Directory containing raw benchmark JSON files; repeatable")
    parser.add_argument("--contract-dir", type=Path)
    parser.add_argument("--models-yaml", type=Path, default=ROOT / "models.yaml")
    parser.add_argument("--target")
    parser.add_argument("--run-id", default=f"quality-diagnosis-{dt.datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "reports" / "quality-diagnosis" / "latest")
    args = parser.parse_args()

    raw_reports = list(args.raw_report)
    for raw_dir in args.raw_dir:
        raw_reports.extend(sorted(raw_dir.glob("*.json")))
    if not raw_reports and not args.contract_dir:
        raise SystemExit("provide --raw-report/--raw-dir and/or --contract-dir")

    report = build_report(
        raw_reports=sorted(raw_reports),
        contract_dir=args.contract_dir,
        models_yaml=args.models_yaml,
        target=args.target,
        run_id=args.run_id,
    )
    paths = write_artifacts(report, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
