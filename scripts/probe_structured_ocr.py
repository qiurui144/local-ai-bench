#!/usr/bin/env python3
"""Evaluate structured OCR extraction on the synthetic OCR manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.ocr.datasets import load_ocr_manifest  # noqa: E402
from benchmark.ocr.metrics import corpus_cer, corpus_ned  # noqa: E402
from benchmark.ocr.runner import build_recognizer  # noqa: E402
from common import ModelConfig, summarize_latencies  # noqa: E402


FIELD_PATTERNS = {
    "contract_no": [r"合同编号[:：]?\s*([A-Z0-9-]+)"],
    "invoice_no": [r"Invoice\s+No\.?\s*([A-Z0-9-]+)"],
    "date": [r"Date[:：]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"],
    "amount": [r"Amount[:：]?\s*([$¥]?\s*[0-9,]+(?:\.[0-9]{2})?)", r"金额[:：]?\s*([¥￥]?\s*[0-9,]+(?:\.[0-9]{2})?)"],
    "amount_upper": [r"总金额[（(]大写[）)][:：]?\s*([一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾佰仟万元整]+)"],
    "party_a": [r"甲方[:：]?\s*([^\s]+)"],
    "payee": [r"收款人[:：]?\s*([^\s]+)"],
    "customer_name": [r"Customer\s+Name[:：]?\s*([^\s]+)", r"客户姓名\s*/\s*Customer\s+Name[:：]?\s*([^\s]+)"],
    "address": [r"地址[:：]?\s*([^\s]+)"],
    "phone": [r"手机号码[:：]?\s*([0-9-]+)"],
    "time_window": [r"(早上\s*9\s*点\s*至\s*下午\s*5\s*点|9\s*点\s*至\s*下午\s*5\s*点)"],
    "note": [r"备注[:：]?\s*(.+)$"],
    "document_title": [r"(增值税专用发票)"],
}


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value or "").replace("￥", "¥").lower()


def extract_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, patterns in FIELD_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                fields[key] = match.group(1).strip()
                break
    return fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["rapidocr", "directml", "openvino", "vitisai", "paddleocr"])
    parser.add_argument("--model-name", default="")
    parser.add_argument("--manifest", default="datasets/ocr/manifest.jsonl")
    parser.add_argument("--image-root", default="datasets/ocr")
    parser.add_argument("--out", default="")
    parser.add_argument("--num-samples", type=int, default=0)
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    samples = load_ocr_manifest(
        manifest,
        image_root=ROOT / args.image_root,
        num_samples=args.num_samples or None,
    )
    model_cfg = ModelConfig(
        name=args.model_name or f"structured-ocr-{args.backend}",
        provider="local_onnx",
        port=0,
        ocr_backend=args.backend,
    )
    recognizer, backend_name = build_recognizer(args.backend)
    if recognizer is None:
        report = {
            "benchmark": "structured_ocr",
            "model": model_cfg.name,
            "backend": args.backend,
            "status": "blocked",
            "reason": backend_name,
            "verdict": "BLOCKED",
        }
    else:
        refs: list[str] = []
        hyps: list[str] = []
        latencies: list[float] = []
        per_sample: list[dict] = []
        total_fields = matched_fields = expected_samples = 0
        errors = 0
        for sample in samples:
            t0 = time.monotonic()
            try:
                hyp = recognizer(sample.image)
            except Exception as exc:
                hyp = ""
                errors += 1
                error = repr(exc)
            else:
                error = ""
            latency_ms = (time.monotonic() - t0) * 1000
            ref_fields = extract_fields(sample.text)
            hyp_fields = extract_fields(hyp)
            expected = bool(ref_fields)
            if expected:
                expected_samples += 1
            field_results = {}
            for key, ref_val in ref_fields.items():
                total_fields += 1
                hyp_val = hyp_fields.get(key, "")
                ok = _norm(ref_val) == _norm(hyp_val)
                matched_fields += 1 if ok else 0
                field_results[key] = {"expected": ref_val, "actual": hyp_val, "ok": ok}
            refs.append(sample.text)
            hyps.append(hyp)
            latencies.append(latency_ms)
            per_sample.append({
                "uid": sample.uid,
                "description": sample.description,
                "expected_text": sample.text,
                "actual_text": hyp,
                "expected_fields": ref_fields,
                "actual_fields": hyp_fields,
                "field_results": field_results,
                "latency_ms": round(latency_ms, 3),
                "error": error,
            })

        field_accuracy = matched_fields / total_fields if total_fields else 0.0
        report = {
            "benchmark": "structured_ocr",
            "model": model_cfg.name,
            "backend": backend_name,
            "status": "ok",
            "num_samples": len(samples),
            "expected_structured_samples": expected_samples,
            "total_fields": total_fields,
            "matched_fields": matched_fields,
            "field_accuracy": field_accuracy,
            "text_cer": corpus_cer(refs, hyps),
            "text_ned": corpus_ned(refs, hyps),
            "latency_ms_stats": summarize_latencies(latencies),
            "error_count": errors,
            "backend_providers": getattr(recognizer, "_rapidocr_providers", None),
            "per_sample": per_sample,
        }
        reasons = []
        if errors == len(samples):
            reasons.append("all samples errored")
        if field_accuracy < 0.85:
            reasons.append(f"field_accuracy {field_accuracy:.3f} < 0.85")
        if report["text_cer"] > 0.10:
            reasons.append(f"text_cer {report['text_cer']:.3f} > 0.10")
        report["verdict"] = "FAIL" if reasons else "PASS"
        report["verdict_reasons"] = reasons

    out = Path(args.out) if args.out else ROOT / "output" / "reports" / f"structured_ocr_{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({k: report.get(k) for k in ("verdict", "backend", "field_accuracy", "text_cer", "latency_ms_stats", "reason")}, ensure_ascii=False))
    return 0 if report.get("verdict") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
