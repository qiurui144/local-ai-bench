"""Groundedness & citation assessment (PDF Chapter 6).

Groundedness asks: is every claim in the generated answer supported by
the retrieved evidence? This is the anti-hallucination metric.

We assess at the claim level rather than answer level:
- decompose the answer into atomic claims
- for each claim, check whether a supplied citation supports it
- compute claim-level grounded rate + citation precision/recall

Beyond the PDF we add a RAGAS-style "faithfulness" alias: identical
shape but uses the strict mode where ANY unsupported claim drops the
entire answer's faithfulness to 0. We surface both because each captures
a different operational concern (gradient vs binary safety).

Provided
--------
- decompose_claims: lightweight sentence-level splitter (caller can plug in
  LLM-based decomposition).
- grounded_rate: fraction of claims marked supported by their citation.
- attribution_precision / attribution_recall.
- citation_precision_recall: doc-level citation correctness.
- abstention_confusion_matrix.
- ragas_faithfulness alias.

References
----------
- Es, S. et al. (2023). RAGAs: Automated Evaluation of Retrieval
  Augmented Generation. EACL Demos.
- Bohnet, B. et al. (2022). Attributed Question Answering. (claim-level
  attribution definition)
- Rashkin, H. et al. (2021). Measuring Attribution in Natural Language
  Generation Models. ACL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Claim:
    """An atomic claim decomposed from an answer."""

    claim_id: str
    text: str
    cited_doc_ids: List[str]  # which retrieved docs the answer attributed


@dataclass(frozen=True)
class ClaimJudgment:
    """Per-claim grounding verdict."""

    claim_id: str
    supported: bool
    supporting_doc_ids: List[str]  # subset of cited that the judge confirmed
    notes: str = ""


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


_SENT_SPLIT_RE = re.compile(r"(?<=[\.!\?。!?])\s+")


def decompose_claims(answer: str, citation_pattern: Optional[re.Pattern] = None) -> List[Claim]:
    """Lightweight sentence-level decomposition with citation extraction.

    Default: split on `. ! ? 。!?` and treat each sentence as a claim.
    Inline citations of the form `[doc_id]` are extracted into `cited_doc_ids`.

    Production-grade decomposition should use an LLM call; this function
    provides a structural fallback so the rest of the pipeline can run
    end-to-end without an extra model dependency.
    """
    pattern = citation_pattern or re.compile(r"\[([^\]]+)\]")
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(answer) if s.strip()]
    out: List[Claim] = []
    for i, sent in enumerate(sentences):
        cites = pattern.findall(sent)
        # Split comma/semicolon-separated citation lists.
        flat_cites: List[str] = []
        for c in cites:
            for piece in re.split(r"[,;]", c):
                piece = piece.strip()
                if piece:
                    flat_cites.append(piece)
        cleaned_text = pattern.sub("", sent).strip()
        out.append(Claim(claim_id=f"c{i}", text=cleaned_text, cited_doc_ids=flat_cites))
    return out


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundednessReport:
    n_claims: int
    n_supported: int
    grounded_rate: float
    faithfulness_strict: float  # 1 only if every claim supported, else 0
    attribution_precision: float
    attribution_recall: float
    citation_precision: float
    citation_recall: float


def grounded_rate(judgments: Sequence[ClaimJudgment]) -> float:
    if not judgments:
        return 1.0  # vacuous: no claims to fail
    return sum(1 for j in judgments if j.supported) / len(judgments)


def faithfulness_strict(judgments: Sequence[ClaimJudgment]) -> float:
    """RAGAS-style strict faithfulness: ANY unsupported claim drops to 0."""
    if not judgments:
        return 1.0
    return 1.0 if all(j.supported for j in judgments) else 0.0


def attribution_precision_recall(
    claims: Sequence[Claim],
    judgments: Sequence[ClaimJudgment],
) -> Tuple[float, float]:
    """For each claim, did the model cite docs and were those cites correct?

    Precision: of the cited docs across all claims, what fraction were
    judge-confirmed as supporting?
    Recall: of the docs the judge confirmed supporting, what fraction
    did the model actually cite?
    """
    judgment_by_id = {j.claim_id: j for j in judgments}
    total_cited = 0
    correct_cited = 0
    total_actual_support = 0
    cited_actual_support = 0
    for c in claims:
        j = judgment_by_id.get(c.claim_id)
        if not j:
            continue
        cited = set(c.cited_doc_ids)
        supporting = set(j.supporting_doc_ids)
        total_cited += len(cited)
        correct_cited += len(cited & supporting)
        total_actual_support += len(supporting)
        cited_actual_support += len(cited & supporting)
    prec = correct_cited / total_cited if total_cited else 0.0
    rec = cited_actual_support / total_actual_support if total_actual_support else 0.0
    return float(prec), float(rec)


# ---------------------------------------------------------------------------
# Document-level citation accuracy (per-answer, all-claims roll-up)
# ---------------------------------------------------------------------------


def citation_precision_recall(
    cited_doc_ids: Sequence[str],
    ground_truth_supporting_doc_ids: Sequence[str],
) -> Tuple[float, float]:
    """Precision/recall on per-answer citation set.

    Use when you have a single 'sources' list per answer (rather than
    inline per-claim citations) and a ground-truth list of docs that
    actually contain the answer.
    """
    cited = set(cited_doc_ids)
    gt = set(ground_truth_supporting_doc_ids)
    if not cited and not gt:
        return 1.0, 1.0  # trivially correct: no citations needed, none given
    if not cited:
        return 0.0, 0.0
    if not gt:
        return 0.0, 1.0
    overlap = len(cited & gt)
    prec = overlap / len(cited)
    rec = overlap / len(gt)
    return float(prec), float(rec)


# ---------------------------------------------------------------------------
# Abstention confusion matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstentionMatrix:
    """2x2 confusion: rows = expected (answer / refuse), cols = actual."""

    tp_answered: int  # expected answer, system answered
    fp_answered: int  # expected refuse, system answered (under_refusal)
    fn_refused: int   # expected answer, system refused (over_refusal)
    tn_refused: int   # expected refuse, system refused

    @property
    def total(self) -> int:
        return self.tp_answered + self.fp_answered + self.fn_refused + self.tn_refused

    @property
    def answer_recall(self) -> float:
        denom = self.tp_answered + self.fn_refused
        return self.tp_answered / denom if denom else 0.0

    @property
    def answer_precision(self) -> float:
        denom = self.tp_answered + self.fp_answered
        return self.tp_answered / denom if denom else 0.0


def abstention_confusion_matrix(
    items: Sequence[Tuple[bool, bool]],
) -> AbstentionMatrix:
    """Items are (expected_to_answer, system_answered) pairs."""
    tp = fp = fn = tn = 0
    for expected, actual in items:
        if expected and actual:
            tp += 1
        elif not expected and actual:
            fp += 1
        elif expected and not actual:
            fn += 1
        else:
            tn += 1
    return AbstentionMatrix(tp_answered=tp, fp_answered=fp, fn_refused=fn, tn_refused=tn)


# ---------------------------------------------------------------------------
# End-to-end report builder
# ---------------------------------------------------------------------------


def groundedness_report(
    claims: Sequence[Claim],
    judgments: Sequence[ClaimJudgment],
    per_answer_cited: Optional[Sequence[str]] = None,
    per_answer_truth: Optional[Sequence[str]] = None,
) -> GroundednessReport:
    g = grounded_rate(judgments)
    f = faithfulness_strict(judgments)
    ap, ar = attribution_precision_recall(claims, judgments)
    if per_answer_cited is not None and per_answer_truth is not None:
        cp, cr = citation_precision_recall(per_answer_cited, per_answer_truth)
    else:
        # Roll up from claim-level cites.
        all_cited: List[str] = []
        all_truth: List[str] = []
        judgment_by_id = {j.claim_id: j for j in judgments}
        for c in claims:
            all_cited.extend(c.cited_doc_ids)
            j = judgment_by_id.get(c.claim_id)
            if j:
                all_truth.extend(j.supporting_doc_ids)
        cp, cr = citation_precision_recall(all_cited, all_truth)
    return GroundednessReport(
        n_claims=len(judgments),
        n_supported=sum(1 for j in judgments if j.supported),
        grounded_rate=g,
        faithfulness_strict=f,
        attribution_precision=ap,
        attribution_recall=ar,
        citation_precision=cp,
        citation_recall=cr,
    )


# Public alias matching the RAGAS naming convention.
ragas_faithfulness = faithfulness_strict
