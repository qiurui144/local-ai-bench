"""Tests for benchmark.rag.groundedness."""
from __future__ import annotations


from benchmark.rag.groundedness import (
    Claim,
    ClaimJudgment,
    abstention_confusion_matrix,
    citation_precision_recall,
    decompose_claims,
    faithfulness_strict,
    grounded_rate,
    groundedness_report,
    ragas_faithfulness,
)


def test_decompose_claims_splits_on_punctuation():
    text = "Cats are mammals. Dogs are too! Birds fly?"
    claims = decompose_claims(text)
    assert len(claims) == 3


def test_decompose_claims_extracts_citations():
    text = "Earth is round [doc1]. The Sun is a star [doc2,doc3]."
    claims = decompose_claims(text)
    assert claims[0].cited_doc_ids == ["doc1"]
    assert set(claims[1].cited_doc_ids) == {"doc2", "doc3"}


def test_grounded_rate_all_supported():
    judgments = [
        ClaimJudgment(claim_id="c0", supported=True, supporting_doc_ids=["d"]),
        ClaimJudgment(claim_id="c1", supported=True, supporting_doc_ids=["d"]),
    ]
    assert grounded_rate(judgments) == 1.0


def test_grounded_rate_mixed():
    judgments = [
        ClaimJudgment(claim_id="c0", supported=True, supporting_doc_ids=[]),
        ClaimJudgment(claim_id="c1", supported=False, supporting_doc_ids=[]),
    ]
    assert grounded_rate(judgments) == 0.5


def test_faithfulness_strict_zero_if_any_unsupported():
    judgments = [
        ClaimJudgment(claim_id="c0", supported=True, supporting_doc_ids=["d"]),
        ClaimJudgment(claim_id="c1", supported=False, supporting_doc_ids=[]),
    ]
    assert faithfulness_strict(judgments) == 0.0


def test_ragas_faithfulness_alias():
    judgments = [ClaimJudgment(claim_id="c0", supported=True, supporting_doc_ids=["d"])]
    assert ragas_faithfulness(judgments) == 1.0


def test_citation_precision_recall_perfect():
    p, r = citation_precision_recall(["a", "b"], ["a", "b"])
    assert p == 1.0 and r == 1.0


def test_citation_precision_recall_partial():
    p, r = citation_precision_recall(["a"], ["a", "b"])
    assert p == 1.0
    assert r == 0.5


def test_abstention_confusion_basic():
    items = [(True, True), (True, True), (False, False), (False, True)]
    m = abstention_confusion_matrix(items)
    assert m.tp_answered == 2
    assert m.fp_answered == 1
    assert m.tn_refused == 1
    assert m.fn_refused == 0
    assert m.answer_recall == 1.0
    assert 0 < m.answer_precision < 1


def test_groundedness_report_end_to_end():
    claims = [
        Claim(claim_id="c0", text="x", cited_doc_ids=["d1"]),
        Claim(claim_id="c1", text="y", cited_doc_ids=["d2"]),
    ]
    judgments = [
        ClaimJudgment(claim_id="c0", supported=True, supporting_doc_ids=["d1"]),
        ClaimJudgment(claim_id="c1", supported=False, supporting_doc_ids=[]),
    ]
    rpt = groundedness_report(claims, judgments)
    assert rpt.n_claims == 2
    assert rpt.n_supported == 1
    assert rpt.grounded_rate == 0.5
    assert rpt.faithfulness_strict == 0.0
