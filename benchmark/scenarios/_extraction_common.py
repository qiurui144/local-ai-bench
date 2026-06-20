"""Shared field-extraction L1 scoring logic for S5 (structured_extraction) and S7 (vlm_document_extraction)."""
from __future__ import annotations

import re
from typing import Any


def _normalize(val: Any) -> str:
    """Strip whitespace, fullwidth→ASCII digits, remove currency symbols, thousands commas."""
    if val is None:
        return ""
    s = str(val).strip()
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    s = re.sub(r"[¥￥$€£]", "", s)
    s = s.replace(",", "").replace("，", "")
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


def field_accuracy_score(payload: dict, parsed: dict | None) -> dict:
    """Fraction of non-null golden fields correctly extracted (normalized match)."""
    fields = payload.get("fields", [])
    golden = payload.get("golden", {})
    extracted = parsed if isinstance(parsed, dict) else {}
    expected = [f for f in fields if golden.get(f) is not None]
    if not expected:
        return {"field_accuracy": 1.0, "n_fields": 0, "n_correct": 0}
    n_correct = sum(
        1 for f in expected
        if _normalize(extracted.get(f)) == _normalize(golden.get(f))
    )
    return {
        "field_accuracy": n_correct / len(expected),
        "n_fields": len(expected),
        "n_correct": n_correct,
    }
