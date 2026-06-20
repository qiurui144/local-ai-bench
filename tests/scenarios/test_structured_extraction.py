"""Unit tests for S5: structured_extraction scenario."""
import pytest
from benchmark.scenarios.base import ScenarioCase
from benchmark.scenarios.structured_extraction import SPEC, _normalize


# ─────────────────────────────────────────────────────────────
# _normalize: value normalization
# ─────────────────────────────────────────────────────────────

def test_normalize_strips_currency():
    assert _normalize("¥12500.00") == "12500.00"
    assert _normalize("$1,000") == "1000"

def test_normalize_fullwidth_digits():
    assert _normalize("２０２４") == "2024"

def test_normalize_removes_thousands_comma():
    assert _normalize("1,234,567") == "1234567"

def test_normalize_lowercases():
    assert _normalize("HTTPS") == "https"

def test_normalize_none():
    assert _normalize(None) == ""

def test_normalize_strips_whitespace():
    assert _normalize("  hello  ") == "hello"


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

def _case(fields, golden, text="sample document", doc_type="invoice"):
    return ScenarioCase(id="t1", provenance="curated", payload={
        "document_type": doc_type,
        "text": text,
        "fields": fields,
        "golden": golden,
    })


# ─────────────────────────────────────────────────────────────
# l1_score
# ─────────────────────────────────────────────────────────────

def test_l1_perfect_match():
    case = _case(["invoice_number", "amount"],
                 {"invoice_number": "INV-001", "amount": "5000.00"})
    parsed = {"invoice_number": "INV-001", "amount": "5000.00"}
    s = SPEC.l1_score(case, parsed)
    assert s["field_accuracy"] == 1.0
    assert s["n_correct"] == 2


def test_l1_normalized_match():
    case = _case(["amount"], {"amount": "12500.00"})
    parsed = {"amount": "¥12,500.00"}
    s = SPEC.l1_score(case, parsed)
    assert s["field_accuracy"] == 1.0


def test_l1_partial_match():
    case = _case(["a", "b", "c"], {"a": "x", "b": "y", "c": "z"})
    parsed = {"a": "x", "b": "wrong", "c": "z"}
    s = SPEC.l1_score(case, parsed)
    assert s["field_accuracy"] == pytest.approx(2/3)


def test_l1_unparseable():
    case = _case(["a", "b"], {"a": "1", "b": "2"})
    s = SPEC.l1_score(case, None)
    assert s["field_accuracy"] == 0.0
    assert s["n_correct"] == 0


def test_l1_null_golden_field_skipped():
    # null golden values are not counted as expected
    case = _case(["a", "b"], {"a": "x", "b": None})
    parsed = {"a": "x", "b": "anything"}
    s = SPEC.l1_score(case, parsed)
    assert s["n_fields"] == 1  # only "a" counts
    assert s["field_accuracy"] == 1.0


def test_l1_extra_fields_in_output_ignored():
    case = _case(["name"], {"name": "Test"})
    parsed = {"name": "Test", "extra": "garbage", "another": "field"}
    s = SPEC.l1_score(case, parsed)
    assert s["field_accuracy"] == 1.0


# ─────────────────────────────────────────────────────────────
# aggregate
# ─────────────────────────────────────────────────────────────

def test_aggregate_mean():
    agg = SPEC.aggregate_l1([
        {"field_accuracy": 1.0, "n_fields": 3, "n_correct": 3},
        {"field_accuracy": 0.5, "n_fields": 2, "n_correct": 1},
    ])
    assert agg["field_accuracy"] == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────
# spec shape + prompt
# ─────────────────────────────────────────────────────────────

def test_spec_shape():
    assert SPEC.name == "structured_extraction"
    assert not SPEC.requires_vlm
    assert "field_accuracy_min" in SPEC.default_thresholds


def test_build_prompt_contains_text_and_fields():
    case = _case(["invoice_number", "amount"],
                 {"invoice_number": "INV-001", "amount": "5000"},
                 text="Invoice INV-001 total ¥5000")
    prompt, image = SPEC.build_prompt(case)
    assert image is None
    assert "Invoice INV-001" in prompt
    assert "invoice_number" in prompt
    assert "amount" in prompt
    assert "JSON" in prompt
