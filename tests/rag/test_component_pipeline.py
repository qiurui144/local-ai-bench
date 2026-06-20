"""Tests for benchmark.rag.component_pipeline."""
from __future__ import annotations

import pytest

from benchmark.rag.component_pipeline import FailureKind


def test_failure_kinds_enum_has_expected_members():
    members = {k.value for k in FailureKind}
    for required in (
        "retrieval_miss",
        "retrieval_noise",
        "chunk_too_small",
        "chunk_too_large",
        "hallucination",
        "over_refusal",
        "under_refusal",
        "citation_missing",
        "citation_wrong",
        "citation_fabricated",
        "prompt_injection",
        "latency_blowup",
        "schema_violation",
    ):
        assert required in members


def test_failure_kind_string_roundtrip():
    k = FailureKind.HALLUCINATION
    assert k.value == "hallucination"
    assert FailureKind("hallucination") is FailureKind.HALLUCINATION


def test_failure_kind_distinct_values():
    values = [k.value for k in FailureKind]
    assert len(values) == len(set(values))


def test_failure_kind_is_string_enum():
    assert isinstance(FailureKind.RETRIEVAL_MISS.value, str)


def test_failure_kind_unknown_raises():
    with pytest.raises(ValueError):
        FailureKind("no_such_kind")
