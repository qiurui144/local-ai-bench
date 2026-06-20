"""OCR quality metrics — CER / NED / WER (character and word level).

Mirrors the ASR metrics module; reuses the same edit-distance kernel.
NED = 1 - edit_distance(ref, hyp) / max(len(ref), len(hyp)) for partial-match scoring.
"""
from __future__ import annotations

import unicodedata


def _normalize(text: str) -> str:
    """NFKC-normalize and strip surrounding whitespace."""
    return unicodedata.normalize("NFKC", text).strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate."""
    ref, hyp = _normalize(ref), _normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def ned(ref: str, hyp: str) -> float:
    """Normalized Edit Distance (0=perfect, 1=worst). Used for partial/noisy GT."""
    ref, hyp = _normalize(ref), _normalize(hyp)
    denom = max(len(ref), len(hyp))
    if denom == 0:
        return 0.0
    return edit_distance(ref, hyp) / denom


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate (space-tokenised)."""
    ref_w = _normalize(ref).split()
    hyp_w = _normalize(hyp).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return edit_distance(ref_w, hyp_w) / len(ref_w)


def corpus_cer(refs: list[str], hyps: list[str]) -> float:
    total_chars = sum(len(_normalize(r)) for r in refs)
    total_edits = sum(edit_distance(_normalize(r), _normalize(h)) for r, h in zip(refs, hyps))
    return total_edits / total_chars if total_chars else 0.0


def corpus_ned(refs: list[str], hyps: list[str]) -> float:
    scores = [ned(r, h) for r, h in zip(refs, hyps)]
    return sum(scores) / len(scores) if scores else 0.0
