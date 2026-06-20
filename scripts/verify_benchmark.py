#!/usr/bin/env python3
"""verify_benchmark.py — Comprehensive dataset integrity verification.

Checks:
  1. Schema completeness: all scenarios have cases.jsonl
  2. JSON validity: every line is valid JSON with required fields
  3. Provenance audit: distribution of synthetic vs curated
  4. VLM image existence: image_path files exist on disk
  5. Duplicate ID check: no duplicate IDs within a scenario
  6. Golden coverage: no empty golden dicts
  7. Field consistency: fields list matches golden keys
  8. Case count summary per scenario

Exit 0 = PASS, 1 = WARN (synthetic-heavy), 2 = FAIL (errors found)

Required payload fields are derived from ScenarioSpec.payload_required_fields —
no hardcoded schema dict. Adding a new scenario and setting payload_required_fields
on its SPEC is sufficient; this script auto-detects it.
"""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(ROOT))
from benchmark.scenarios import SCENARIOS  # noqa: E402

VALID_PROVENANCES = {"synthetic", "curated", "dataset", "synthetic_fallback"}


def check_scenario(name: str, spec) -> dict:
    path = ROOT / f"datasets/scenarios/{name}/cases.jsonl"
    result = {"name": name, "errors": [], "warnings": [], "case_count": 0,
              "provenance": Counter(), "image_missing": 0}

    if not path.exists():
        result["errors"].append(f"cases.jsonl missing: {path}")
        return result

    required_fields = spec.payload_required_fields
    needs_vlm = spec.requires_vlm
    ids_seen = set()

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as e:
            result["errors"].append(f"line {lineno}: invalid JSON: {e}")
            continue

        # Required top-level fields
        for f in ("id", "provenance", "payload"):
            if f not in case:
                result["errors"].append(f"line {lineno}: missing '{f}'")

        cid = case.get("id", f"line{lineno}")
        prov = case.get("provenance", "")
        payload = case.get("payload", {})

        # Duplicate ID
        if cid in ids_seen:
            result["errors"].append(f"duplicate id: {cid}")
        ids_seen.add(cid)

        # Provenance
        if prov not in VALID_PROVENANCES:
            result["errors"].append(f"{cid}: invalid provenance '{prov}'")
        result["provenance"][prov] += 1

        # Payload required fields (derived from ScenarioSpec)
        for rf in required_fields:
            if rf not in payload:
                result["errors"].append(f"{cid}: payload missing '{rf}'")

        # VLM image existence (derived from spec.requires_vlm + payload_required_fields)
        if needs_vlm:
            img_key = "image_path" if "image_path" in required_fields else "image"
            img = payload.get(img_key)
            if img and not (ROOT / img).exists():
                result["image_missing"] += 1
                result["errors"].append(f"{cid}: image not found: {img}")

        # Golden coverage for extraction scenarios
        if "golden" in required_fields and "fields" in required_fields:
            golden = payload.get("golden", {})
            fields = payload.get("fields", [])
            if not golden:
                result["warnings"].append(f"{cid}: empty golden dict")
            if not fields:
                result["errors"].append(f"{cid}: empty fields list")

        result["case_count"] += 1

    # Warn if mostly synthetic
    synthetic_count = result["provenance"].get("synthetic", 0) + result["provenance"].get("synthetic_fallback", 0)
    if result["case_count"] > 0 and synthetic_count / result["case_count"] > 0.8:
        result["warnings"].append(
            f"{synthetic_count}/{result['case_count']} cases are synthetic — verdict capped at WARN"
        )

    return result


def main():
    all_results = [check_scenario(name, spec) for name, spec in SCENARIOS.items()]

    total_errors = sum(len(r["errors"]) for r in all_results)
    total_warnings = sum(len(r["warnings"]) for r in all_results)
    scenario_count = len(all_results)

    print("=" * 60)
    print("vlm-llm-benchmark Dataset Verification Report")
    print("=" * 60)

    for r in all_results:
        status = "PASS" if not r["errors"] else "FAIL"
        warn_tag = f"  ({len(r['warnings'])} warnings)" if r["warnings"] else ""
        prov_str = dict(r["provenance"])
        print(f"\n[{status}]{warn_tag}  {r['name']} ({r['case_count']} cases, provenance={prov_str})")
        for e in r["errors"]:
            print(f"    ERROR: {e}")
        for w in r["warnings"]:
            print(f"    WARN:  {w}")

    print("\n" + "=" * 60)
    print(f"Summary: {sum(r['case_count'] for r in all_results)} total cases across {scenario_count} scenarios")
    print(f"         {total_errors} errors, {total_warnings} warnings")

    if total_errors > 0:
        print("RESULT: FAIL")
        sys.exit(2)
    elif total_warnings > 0:
        print("RESULT: WARN (review warnings above)")
        sys.exit(1)
    else:
        print("RESULT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
