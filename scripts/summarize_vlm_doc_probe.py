#!/usr/bin/env python3
"""Summarize VLM document-extraction probe logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from decimal import Decimal, InvalidOperation
from pathlib import Path


def normalize_value(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("人民币", "").replace("元", "")
    return re.sub(r"[\s\t\r\n:：,，.。;；/\\|_\-—*]+", "", text)


def expected_variants(value: object) -> set[str]:
    text = str(value or "").strip()
    variants = {normalize_value(text)}
    numeric = text.replace("人民币", "").replace("元", "").replace(",", "").replace("，", "")
    numeric = numeric.strip()
    suffix = "%" if numeric.endswith("%") else ""
    if suffix:
        numeric = numeric[:-1]
    try:
        dec = Decimal(numeric)
    except (InvalidOperation, ValueError):
        return {v for v in variants if v}
    fixed = format(dec, "f")
    trimmed = fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    for candidate in {fixed, trimmed, f"{trimmed}{suffix}", f"{fixed}{suffix}"}:
        variants.add(normalize_value(candidate))
    return {v for v in variants if v}


def extract_jsonish(text: str) -> object | None:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def load_cases(path: Path) -> dict[str, dict]:
    cases = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        cases[case["id"]] = case["payload"]
    return cases


def model_name_from_path(path: Path) -> str:
    suffix = ".api-probe.stdout.log"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    parts = path.parts
    if len(parts) >= 3 and parts[-2] == "private-spacemit":
        return parts[-3]
    return path.parent.name


def summarize_log(path: Path, cases: dict[str, dict]) -> dict:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    doc_results = []
    for event in events:
        if event.get("case") != "vlm-doc-case":
            continue
        case_id = event.get("id")
        payload = cases.get(case_id, {})
        fields = payload.get("fields", [])
        golden = payload.get("golden", {})
        content = event.get("content", "")
        parsed = extract_jsonish(content)
        haystack = normalize_value(content)
        if parsed is not None:
            haystack += normalize_value(json.dumps(parsed, ensure_ascii=False))

        field_scores = {}
        for field in fields:
            variants = expected_variants(golden.get(field, ""))
            field_scores[field] = any(needle in haystack for needle in variants)
        hits = sum(1 for ok in field_scores.values() if ok)
        total = len(fields)
        doc_results.append(
            {
                "id": case_id,
                "document_type": payload.get("document_type", event.get("document_type")),
                "elapsed_s": event.get("elapsed_s"),
                "json_parse_ok": parsed is not None,
                "field_count": total,
                "field_hits": hits,
                "field_accuracy": round(hits / total, 4) if total else 0.0,
                "case_pass": total > 0 and hits == total,
                "field_scores": field_scores,
            }
        )

    latencies = sorted(float(r["elapsed_s"]) for r in doc_results if r.get("elapsed_s") is not None)
    field_count = sum(r["field_count"] for r in doc_results)
    field_hits = sum(r["field_hits"] for r in doc_results)
    case_pass = sum(1 for r in doc_results if r["case_pass"])
    failed = [
        {
            "id": r["id"],
            "document_type": r["document_type"],
            "missed": [k for k, ok in r["field_scores"].items() if not ok],
        }
        for r in doc_results
        if not r["case_pass"]
    ]

    def first_event(name: str) -> dict:
        return next((e for e in events if e.get("case") == name), {})

    context_1024 = first_event("context-1024")
    raw_aggregate = first_event("vlm-doc-aggregate")
    summary = {
        "model": model_name_from_path(path),
        "path": str(path),
        "cases": len(doc_results),
        "case_pass": case_pass,
        "case_pass_rate": round(case_pass / len(doc_results), 4) if doc_results else 0.0,
        "field_count": field_count,
        "field_hits": field_hits,
        "field_accuracy": round(field_hits / field_count, 4) if field_count else 0.0,
        "json_parse_rate": round(sum(1 for r in doc_results if r["json_parse_ok"]) / len(doc_results), 4)
        if doc_results
        else 0.0,
        "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95_s": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3)
        if latencies
        else None,
        "ttft_s": first_event("stream-ttft").get("ttft_s"),
        "decode128_s": first_event("decode-128").get("elapsed_s"),
        "context_1024_s": context_1024.get("elapsed_s"),
        "context_1024_pass": context_1024.get("needle_recall"),
        "vlm_image_s": first_event("vlm-image").get("elapsed_s"),
        "failed": failed,
    }
    if raw_aggregate:
        summary["raw_aggregate"] = raw_aggregate
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    for log in args.logs:
        print(json.dumps(summarize_log(log, cases), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
