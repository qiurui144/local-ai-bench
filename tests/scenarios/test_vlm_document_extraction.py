"""Unit tests for S7: vlm_document_extraction scenario."""
import pytest
from benchmark.scenarios.base import ScenarioCase
from benchmark.scenarios.vlm_document_extraction import SPEC


# ─── helpers ────────────────────────────────────────────────────────────────

def _case(doc_type, image_path, fields, golden):
    return ScenarioCase(id="t1", provenance="curated", payload={
        "document_type": doc_type,
        "image_path": image_path,
        "fields": fields,
        "golden": golden,
    })


# ─── build_prompt ────────────────────────────────────────────────────────────

def test_build_prompt_returns_image_path():
    case = _case("bank_statement", "images/bs/c01.png",
                 ["transaction_date"], {"transaction_date": "2024-01-10"})
    prompt, image = SPEC.build_prompt(case)
    assert image == "images/bs/c01.png"


def test_build_prompt_contains_doc_type_zh():
    case = _case("bank_statement", "x.png", ["transaction_date"], {"transaction_date": "x"})
    prompt, _ = SPEC.build_prompt(case)
    assert "银行流水单" in prompt


def test_build_prompt_vat_invoice_zh():
    case = _case("vat_invoice", "x.png", ["invoice_number"], {"invoice_number": "x"})
    prompt, _ = SPEC.build_prompt(case)
    assert "增值税发票" in prompt


def test_build_prompt_receipt_zh():
    case = _case("receipt", "x.png", ["amount"], {"amount": "x"})
    prompt, _ = SPEC.build_prompt(case)
    assert "收据" in prompt


def test_build_prompt_bank_transfer_zh():
    case = _case("bank_transfer", "x.png", ["amount"], {"amount": "x"})
    prompt, _ = SPEC.build_prompt(case)
    assert "汇款" in prompt


def test_build_prompt_lists_fields():
    case = _case("receipt", "x.png",
                 ["merchant", "amount", "date"],
                 {"merchant": "x", "amount": "x", "date": "x"})
    prompt, _ = SPEC.build_prompt(case)
    assert "merchant" in prompt
    assert "amount" in prompt
    assert "date" in prompt


def test_build_prompt_contains_observe_image():
    """Runner test routes by '观察图片' — must be present in every prompt."""
    case = _case("receipt", "x.png", ["amount"], {"amount": "x"})
    prompt, _ = SPEC.build_prompt(case)
    assert "观察图片" in prompt


# ─── l1_score ────────────────────────────────────────────────────────────────

def test_l1_perfect_match():
    case = _case("vat_invoice", "x.png",
                 ["invoice_number", "amount"],
                 {"invoice_number": "31200024501234", "amount": "10000.00"})
    s = SPEC.l1_score(case, {"invoice_number": "31200024501234", "amount": "10000.00"})
    assert s["field_accuracy"] == 1.0
    assert s["n_correct"] == 2


def test_l1_normalized_currency():
    case = _case("receipt", "x.png", ["amount"], {"amount": "68.00"})
    s = SPEC.l1_score(case, {"amount": "¥68.00"})
    assert s["field_accuracy"] == 1.0


def test_l1_normalized_fullwidth_digits():
    case = _case("bank_statement", "x.png",
                 ["transaction_date"], {"transaction_date": "2024-01-10"})
    s = SPEC.l1_score(case, {"transaction_date": "２０２４-０１-１０"})
    assert s["field_accuracy"] == 1.0


def test_l1_partial_match():
    case = _case("bank_transfer", "x.png",
                 ["receiver_name", "amount", "bank_name"],
                 {"receiver_name": "李四", "amount": "50000.00", "bank_name": "中国工商银行"})
    s = SPEC.l1_score(case, {"receiver_name": "李四", "amount": "99.00", "bank_name": "中国工商银行"})
    assert s["field_accuracy"] == pytest.approx(2 / 3)


def test_l1_unparseable():
    case = _case("receipt", "x.png", ["amount", "merchant"], {"amount": "68", "merchant": "星巴克"})
    s = SPEC.l1_score(case, None)
    assert s["field_accuracy"] == 0.0
    assert s["n_correct"] == 0


def test_l1_null_golden_skipped():
    case = _case("vat_invoice", "x.png",
                 ["invoice_number", "tax_rate"],
                 {"invoice_number": "12345", "tax_rate": None})
    s = SPEC.l1_score(case, {"invoice_number": "12345", "tax_rate": "13%"})
    assert s["n_fields"] == 1
    assert s["field_accuracy"] == 1.0


def test_l1_extra_output_fields_ignored():
    case = _case("bank_statement", "x.png",
                 ["counterparty"], {"counterparty": "张三"})
    s = SPEC.l1_score(case, {"counterparty": "张三", "extra": "noise"})
    assert s["field_accuracy"] == 1.0


def test_l1_raw_content_ignored():
    case = _case("receipt", "x.png", ["amount"], {"amount": "68.00"})
    s = SPEC.l1_score(case, {"amount": "68.00"}, raw_content="任意文本")
    assert s["field_accuracy"] == 1.0


# ─── aggregate ───────────────────────────────────────────────────────────────

def test_aggregate_mean():
    agg = SPEC.aggregate_l1([
        {"field_accuracy": 1.0, "n_fields": 3, "n_correct": 3},
        {"field_accuracy": 0.5, "n_fields": 2, "n_correct": 1},
    ])
    assert agg["field_accuracy"] == pytest.approx(0.75)


def test_aggregate_empty():
    agg = SPEC.aggregate_l1([])
    assert agg["field_accuracy"] == 0.0


def test_aggregate_skips_zero_field_cases():
    """Zero-field cases (all golden=None) must not inflate the aggregate."""
    zero_field = {"field_accuracy": 1.0, "n_fields": 0, "n_correct": 0}
    real_case = {"field_accuracy": 0.0, "n_fields": 2, "n_correct": 0}
    agg = SPEC.aggregate_l1([zero_field, real_case])
    assert agg["field_accuracy"] == pytest.approx(0.0)  # only real_case counts


# ─── spec shape ──────────────────────────────────────────────────────────────

def test_spec_requires_vlm():
    assert SPEC.requires_vlm is True


def test_spec_name():
    assert SPEC.name == "vlm_document_extraction"


def test_spec_cases_path():
    assert SPEC.cases_path == "datasets/scenarios/vlm_document_extraction/cases.jsonl"


def test_spec_thresholds():
    assert "field_accuracy_min" in SPEC.default_thresholds
    assert SPEC.default_thresholds["field_accuracy_min"] == pytest.approx(0.75)
