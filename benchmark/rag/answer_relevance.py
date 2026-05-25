"""Answer relevance assessment (PDF Chapter 5).

The relevance of a generated answer is multidimensional:
- Does it satisfy the user's intent? (semantic correctness)
- Does it answer the question, or detour? (focus / on-topic)
- Does it cover all requested facts? (completeness)
- Does it (correctly) refuse when the corpus lacks the answer? (over/under refusal)

This module hosts the metric callable surface; the LLM-as-judge prompts
that produce the per-claim labels live in judge_prompts.py.

Beyond the PDF, we add classical text-similarity baselines because:
- They are free (no LLM call needed) and useful as a sanity-check signal.
- They are needed for sanity-checking judge calibration: if BLEU/ROUGE
  rank A above B but the judge ranks B above A, that's a flag.

Provided
--------
- intent_satisfaction_rate: fraction of items judged "intent satisfied"
- partial_credit_score: claim-level coverage in [0, 1]
- over_refusal_rate / under_refusal_rate
- rouge_l, bleu_4, chrf: baseline string similarity metrics
- semantic_similarity_via_embeddings: cosine on embeddings (model-agnostic)

References
----------
- Papineni, K. et al. (2002). BLEU. ACL.
- Lin, C.-Y. (2004). ROUGE. ACL workshop.
- Popovic, M. (2015). chrF. WMT.
- Zhang, T. et al. (2020). BERTScore. ICLR.
- Sellam, T. et al. (2020). BLEURT. ACL.
- Rei, R. et al. (2020). COMET. EMNLP.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class JudgedAnswer:
    """An LLM-judge or human-judge rating of a single answer."""

    item_id: str
    intent_satisfied: bool
    claim_coverage: float  # in [0, 1]
    refused: bool
    expected_refusal: bool  # ground truth: corpus lacks answer
    notes: str = ""


# ---------------------------------------------------------------------------
# Aggregate metrics over judged answers
# ---------------------------------------------------------------------------


def intent_satisfaction_rate(judged: Sequence[JudgedAnswer]) -> float:
    if not judged:
        return 0.0
    return sum(1 for j in judged if j.intent_satisfied) / len(judged)


def partial_credit_score(judged: Sequence[JudgedAnswer]) -> float:
    if not judged:
        return 0.0
    return sum(j.claim_coverage for j in judged) / len(judged)


def over_refusal_rate(judged: Sequence[JudgedAnswer]) -> float:
    """Fraction of items refused when corpus actually has the answer.

    Over-refusal = answered=False AND expected_refusal=False
    """
    candidates = [j for j in judged if not j.expected_refusal]
    if not candidates:
        return 0.0
    return sum(1 for j in candidates if j.refused) / len(candidates)


def under_refusal_rate(judged: Sequence[JudgedAnswer]) -> float:
    """Fraction of items answered when they should have been refused.

    Under-refusal = answered=True AND expected_refusal=True
    """
    candidates = [j for j in judged if j.expected_refusal]
    if not candidates:
        return 0.0
    return sum(1 for j in candidates if not j.refused) / len(candidates)


# ---------------------------------------------------------------------------
# Text similarity baselines (free; useful as judge sanity checks)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation-aware tokenization.

    Production code should use a real tokenizer; this is for sanity-check
    baselines, not the headline metric.
    """
    text = text.lower()
    # Split on non-alphanumeric; keep meaningful Chinese chars as single tokens.
    tokens: List[str] = []
    for chunk in re.split(r"[\s\.\,\!\?\;\:\(\)\[\]\{\}\"'\-]+", text):
        if not chunk:
            continue
        # Latin sequences are kept whole; CJK chars split per character.
        if re.match(r"^[\x00-\x7f]+$", chunk):
            tokens.append(chunk)
        else:
            for ch in chunk:
                if ch.strip():
                    tokens.append(ch)
    return tokens


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length via O(n*m) DP."""
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def rouge_l(reference: str, candidate: str) -> Dict[str, float]:
    """ROUGE-L F1 on LCS. Returns precision / recall / f1."""
    ref_tok = _tokenize(reference)
    cand_tok = _tokenize(candidate)
    if not ref_tok or not cand_tok:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(ref_tok, cand_tok)
    p = lcs / len(cand_tok)
    r = lcs / len(ref_tok)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def bleu_4(reference: str, candidate: str, smooth: float = 1.0) -> float:
    """Sentence-level BLEU-4 with add-one smoothing.

    Add-one smoothing avoids the corpus-BLEU edge case of zero for any
    missing n-gram order; standard for single-sentence comparisons.
    """
    ref_tok = _tokenize(reference)
    cand_tok = _tokenize(candidate)
    if not cand_tok:
        return 0.0

    def _ngrams(tokens: Sequence[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    weights = [0.25, 0.25, 0.25, 0.25]
    log_precisions = 0.0
    for n in range(1, 5):
        cand_ng = _ngrams(cand_tok, n)
        ref_ng = _ngrams(ref_tok, n)
        if not cand_ng:
            log_precisions += weights[n - 1] * math.log(smooth / 1)
            continue
        overlap = sum(min(cand_ng[g], ref_ng[g]) for g in cand_ng)
        denom = sum(cand_ng.values())
        precision = (overlap + smooth) / (denom + smooth) if denom else smooth
        log_precisions += weights[n - 1] * math.log(max(precision, 1e-12))
    # Brevity penalty
    ref_len = len(ref_tok)
    cand_len = len(cand_tok)
    if cand_len > ref_len:
        bp = 1.0
    elif cand_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / cand_len)
    return bp * math.exp(log_precisions)


def chrf(reference: str, candidate: str, n: int = 6, beta: float = 2.0) -> float:
    """Character-n-gram F-score (Popovic 2015).

    Robust to morphology and works on CJK without segmentation; useful
    as a backstop for any language where token-based BLEU breaks down.
    """
    def char_ngrams(text: str, max_n: int) -> Counter:
        c = Counter()
        for ng in range(1, max_n + 1):
            for i in range(len(text) - ng + 1):
                c[text[i : i + ng]] += 1
        return c

    ref_ng = char_ngrams(reference, n)
    cand_ng = char_ngrams(candidate, n)
    if not ref_ng or not cand_ng:
        return 0.0
    overlap = sum(min(ref_ng[g], cand_ng[g]) for g in cand_ng)
    p = overlap / sum(cand_ng.values()) if cand_ng else 0.0
    r = overlap / sum(ref_ng.values()) if ref_ng else 0.0
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


# ---------------------------------------------------------------------------
# Embedding-based semantic similarity (BLEURT/COMET shape; model is caller's)
# ---------------------------------------------------------------------------


def semantic_similarity_via_embeddings(
    reference_embedding: Sequence[float],
    candidate_embedding: Sequence[float],
) -> float:
    """Cosine similarity between pre-computed embeddings. Range: [-1, 1].

    The caller chooses the embedding model (bge-m3, OpenAI, Cohere, etc).
    This decouples the metric from any specific embedder, so a benchmark
    can compare apples-to-apples across embedding providers.
    """
    a = np.asarray(reference_embedding, dtype=float)
    b = np.asarray(candidate_embedding, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Headline relevance summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerRelevanceReport:
    n_items: int
    intent_satisfaction: float
    claim_coverage: float
    over_refusal: float
    under_refusal: float
    extras: Dict[str, float]


def answer_relevance_report(
    judged: Sequence[JudgedAnswer],
    extras: Optional[Dict[str, float]] = None,
) -> AnswerRelevanceReport:
    return AnswerRelevanceReport(
        n_items=len(judged),
        intent_satisfaction=intent_satisfaction_rate(judged),
        claim_coverage=partial_credit_score(judged),
        over_refusal=over_refusal_rate(judged),
        under_refusal=under_refusal_rate(judged),
        extras=dict(extras or {}),
    )
