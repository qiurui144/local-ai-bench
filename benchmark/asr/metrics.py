"""ASR metrics: WER (word error rate) + CER (char error rate) + RTF.

Pure Python, CPU-only, deterministic — fully unit-testable without any audio /
ONNX model. Methodology mirrors the K23 SenseVoice eval
(``2026-06-02_yolo_asr_ui_eval.md`` §C): edit-distance error rate over reference
vs hypothesis, and RTF = processing_time / audio_duration.

For Chinese, **CER** is the primary metric (word segmentation is ambiguous, so
character-level error rate is the robust comparison); WER is still reported for
space-delimited / mixed text.
"""

from __future__ import annotations

import re
from typing import Sequence


def _levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Edit distance (insert/delete/substitute) between two token sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _normalize(text: str) -> str:
    """Lowercase + strip punctuation/whitespace runs for fair comparison."""
    text = text.strip().lower()
    text = re.sub(r"[，。！？、；：“”‘’（）【】,\.!\?;:\"'()\[\]]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _char_tokens(text: str) -> list[str]:
    """Characters with whitespace removed (CJK-friendly char tokenization)."""
    return [c for c in _normalize(text).replace(" ", "")]


def _word_tokens(text: str) -> list[str]:
    return [t for t in _normalize(text).split(" ") if t]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate in [0, inf); 0 = perfect. Empty ref → 0/1 by hyp."""
    ref = _char_tokens(reference)
    hyp = _char_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate in [0, inf); 0 = perfect."""
    ref = _word_tokens(reference)
    hyp = _word_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def corpus_cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Aggregate CER over a corpus (total edits / total ref chars)."""
    edits = total = 0
    for r, h in zip(refs, hyps):
        rt, ht = _char_tokens(r), _char_tokens(h)
        edits += _levenshtein(rt, ht)
        total += len(rt)
    return edits / total if total else 0.0


def corpus_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Aggregate WER over a corpus (total edits / total ref words)."""
    edits = total = 0
    for r, h in zip(refs, hyps):
        rt, ht = _word_tokens(r), _word_tokens(h)
        edits += _levenshtein(rt, ht)
        total += len(rt)
    return edits / total if total else 0.0


def rtf(processing_time_s: float, audio_duration_s: float) -> float:
    """Real-time factor = processing / audio. < 1.0 means real-time capable."""
    if audio_duration_s <= 0:
        return 0.0
    return processing_time_s / audio_duration_s


def validate_transcript(hyp: str) -> dict:
    """Numerical/sanity validation: flag empty output (the ASR 'fast-but-wrong')."""
    empty = not (hyp and hyp.strip())
    return {"ok": not empty, "empty_output": empty}
