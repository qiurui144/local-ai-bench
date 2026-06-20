"""Translation accuracy evaluation.

Calls the served (vLLM / OpenAI-compatible) model to translate a batch, then
scores the hypotheses against references with corpus-level metrics:

- **SacreBLEU** + **chrF** — pure-Python, CPU-only, reproducible (the
  ``sacrebleu`` package; with a pure-Python fallback so scoring still works
  if the package is missing).
- **COMET** — neural, **GPU-recommended** (``unbabel-comet``). Gracefully
  skipped (marked ``"COMET requires GPU/DGX"``) when the package or a GPU is
  unavailable instead of crashing the run.

Every metric output is validated (non-empty hyps, ``0 <= BLEU <= 100``, chrF in
range, finite / non-NaN-Inf) so a silently broken model surfaces as a FAIL
rather than a plausible-looking number.

Output JSON shape mirrors ``benchmark/accuracy.py``:
  per_pair  : per-language-pair metric block + level breakdown
  aggregate : corpus BLEU / chrF / COMET / term-match-rate
  verdict   : PASS / WARN / FAIL against expectations thresholds
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

from common import ModelConfig, infer_sync

from . import prompts
from .datasets import TranslationPair

logger = logging.getLogger(__name__)

# Sentinel returned by COMET helpers when the neural metric cannot run.
COMET_UNAVAILABLE = "COMET requires GPU/DGX"


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
def translate_batch(
    model_cfg: ModelConfig,
    srcs: Sequence[str],
    src_lang: str,
    tgt_lang: str,
    *,
    level: str = "l1",
    glossary: Optional[dict] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
    """Translate ``srcs`` via the served model; return one hypothesis per src.

    ``level`` selects the prompt template (``l1`` / ``l3``; L2 is a passage and
    handled by :func:`translate_passage`). On a failed call the corresponding
    hypothesis is an empty string (validation downstream flags empties).
    """
    glossary = glossary or {}
    hyps: list[str] = []
    for src in srcs:
        if level == "l3" and glossary:
            prompt = prompts.l3_terminology(src, src_lang, tgt_lang, glossary)
        else:
            prompt = prompts.l1_single_sentence(src, src_lang, tgt_lang)
        res = infer_sync(
            model_cfg,
            prompt=prompt,
            image_path=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        hyps.append(res.content.strip() if res.ok else "")
    return hyps


def translate_passage(
    model_cfg: ModelConfig,
    sentences: Sequence[str],
    src_lang: str,
    tgt_lang: str,
    *,
    max_tokens: int = 768,
    temperature: float = 0.0,
) -> list[str]:
    """L2: translate a 3-5 sentence passage, return one hypothesis per line.

    The model is asked to emit numbered lines; we strip the leading ``N.``
    numbering. If the model returns fewer lines than inputs, missing slots are
    filled with empty strings so length matches the references.
    """
    prompt = prompts.l2_context_consistency(sentences, src_lang, tgt_lang)
    res = infer_sync(
        model_cfg,
        prompt=prompt,
        image_path=None,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not res.ok:
        return ["" for _ in sentences]
    lines = [ln.strip() for ln in res.content.splitlines() if ln.strip()]
    cleaned = [_strip_numbering(ln) for ln in lines]
    if len(cleaned) < len(sentences):
        cleaned += [""] * (len(sentences) - len(cleaned))
    return cleaned[: len(sentences)]


def _strip_numbering(line: str) -> str:
    """Remove a leading ``12.`` / ``12)`` / ``12、`` enumeration marker."""
    import re

    return re.sub(r"^\s*\d+\s*[.)、]\s*", "", line)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _is_zh(lang: str) -> bool:
    return lang.startswith("zh")


def compute_sacrebleu(
    hyps: Sequence[str], refs: Sequence[str], tgt_lang: str = "en"
) -> float:
    """Corpus SacreBLEU in [0, 100].

    Uses the ``sacrebleu`` package when present (Chinese uses the ``zh``
    tokenizer; otherwise the default ``13a``). Falls back to a pure-Python
    corpus BLEU-4 with add-one smoothing if the package is unavailable, so
    CPU-only environments still get a (close) score.
    """
    if not hyps or not refs:
        return 0.0
    try:
        import sacrebleu

        tokenize = "zh" if _is_zh(tgt_lang) else "13a"
        score = sacrebleu.corpus_bleu(list(hyps), [list(refs)], tokenize=tokenize).score
    except Exception:
        score = _fallback_corpus_bleu(hyps, refs, tgt_lang)
    return float(score)


def compute_chrf(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Corpus chrF (character n-gram F-score) in [0, 100].

    ``sacrebleu`` package when present, else a pure-Python chrF over character
    n-grams (n<=6, beta=2). chrF is tokenization-free, so it is the robust
    backstop metric for Chinese where word-BLEU is brittle.
    """
    if not hyps or not refs:
        return 0.0
    try:
        import sacrebleu

        score = sacrebleu.corpus_chrf(list(hyps), [list(refs)]).score
    except Exception:
        score = _fallback_corpus_chrf(hyps, refs)
    return float(score)


def compute_comet(
    hyps: Sequence[str],
    srcs: Sequence[str],
    refs: Sequence[str],
    *,
    model_name: str = "Unbabel/wmt22-comet-da",
) -> dict:
    """Neural COMET score (GPU-recommended).

    Returns ``{"available": True, "score": float, "model": ...}`` on success,
    or ``{"available": False, "reason": COMET_UNAVAILABLE}`` when
    ``unbabel-comet`` / a CUDA GPU / the checkpoint is unavailable. Never
    raises — a missing neural metric must not break the CPU benchmark run.
    """
    if not (hyps and srcs and refs):
        return {"available": False, "reason": "empty input"}
    try:
        import torch  # noqa: F401  (presence check)
        from comet import download_model, load_from_checkpoint

        try:
            has_gpu = bool(torch.cuda.is_available())
        except Exception:
            has_gpu = False
        if not has_gpu:
            return {"available": False, "reason": COMET_UNAVAILABLE}

        ckpt = download_model(model_name)
        model = load_from_checkpoint(ckpt)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        out = model.predict(data, gpus=1, progress_bar=False)
        score = float(out["system_score"])
        return {"available": True, "score": score, "model": model_name}
    except Exception as e:  # ImportError, no checkpoint, OOM, ...
        logger.info("COMET skipped: %s", e)
        return {"available": False, "reason": COMET_UNAVAILABLE}


def term_match_rate(hyps: Sequence[str], glossaries: Sequence[dict]) -> dict:
    """L3 exact-match terminology rate.

    For each (hypothesis, glossary) pair, count how many required target terms
    appear verbatim in the hypothesis. Returns overall rate + counts.
    """
    matched = 0
    total = 0
    for hyp, glossary in zip(hyps, glossaries):
        for _src_term, tgt_term in (glossary or {}).items():
            total += 1
            if tgt_term and tgt_term.lower() in hyp.lower():
                matched += 1
    return {
        "matched_terms": matched,
        "total_terms": total,
        "term_match_rate": (matched / total) if total else 0.0,
    }


# ---- pure-Python fallbacks (only used when sacrebleu is absent) ------------
def _tokenize(text: str, tgt_lang: str) -> list[str]:
    if _is_zh(tgt_lang):
        # character-level for Chinese (matches sacrebleu ``zh`` spirit)
        return [c for c in text if not c.isspace()]
    return text.split()


def _ngram_counts(tokens: list[str], n: int) -> dict:
    counts: dict = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _fallback_corpus_bleu(
    hyps: Sequence[str], refs: Sequence[str], tgt_lang: str, max_n: int = 4
) -> float:
    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        h_tok = _tokenize(hyp, tgt_lang)
        r_tok = _tokenize(ref, tgt_lang)
        hyp_len += len(h_tok)
        ref_len += len(r_tok)
        for n in range(1, max_n + 1):
            h_ng = _ngram_counts(h_tok, n)
            r_ng = _ngram_counts(r_tok, n)
            for ng, c in h_ng.items():
                clipped[n - 1] += min(c, r_ng.get(ng, 0))
            totals[n - 1] += max(sum(h_ng.values()), 0)
    precisions = []
    for n in range(max_n):
        # add-one smoothing avoids zero collapse on short inputs
        precisions.append((clipped[n] + 1.0) / (totals[n] + 1.0))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100.0 * bp * geo_mean


def _char_ngrams(text: str, n: int) -> dict:
    s = text.replace(" ", "")
    counts: dict = {}
    for i in range(len(s) - n + 1):
        ng = s[i : i + n]
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _fallback_corpus_chrf(
    hyps: Sequence[str], refs: Sequence[str], max_n: int = 6, beta: float = 2.0
) -> float:
    f_scores = []
    for n in range(1, max_n + 1):
        match = h_tot = r_tot = 0
        for hyp, ref in zip(hyps, refs):
            h_ng = _char_ngrams(hyp, n)
            r_ng = _char_ngrams(ref, n)
            for ng, c in h_ng.items():
                match += min(c, r_ng.get(ng, 0))
            h_tot += sum(h_ng.values())
            r_tot += sum(r_ng.values())
        prec = match / h_tot if h_tot else 0.0
        rec = match / r_tot if r_tot else 0.0
        if prec + rec == 0:
            f_scores.append(0.0)
        else:
            b2 = beta * beta
            f_scores.append((1 + b2) * prec * rec / (b2 * prec + rec))
    return 100.0 * (sum(f_scores) / max_n)


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------
def validate_metrics(hyps: Sequence[str], bleu: float, chrf: float) -> list[str]:
    """Return a list of validation problems (empty list = all checks pass)."""
    problems: list[str] = []
    if not hyps or all(not h for h in hyps):
        problems.append("all hypotheses empty")
    for name, val, lo, hi in [("BLEU", bleu, 0.0, 100.0), ("chrF", chrf, 0.0, 100.0)]:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            problems.append(f"{name} is NaN/Inf")
        elif not (lo <= val <= hi):
            problems.append(f"{name}={val} out of [{lo},{hi}]")
    return problems


# ---------------------------------------------------------------------------
# Orchestrator (run_xxx idiom)
# ---------------------------------------------------------------------------
def run_translation(
    model_cfg: ModelConfig,
    pairs: Sequence[TranslationPair],
    *,
    level: str = "l1",
    thresholds: Optional[dict] = None,
    run_comet: bool = True,
) -> dict:
    """Translate ``pairs`` and score; return a benchmark result dict.

    Mirrors ``benchmark/accuracy.run_accuracy``: produces ``aggregate`` metrics,
    a PASS/WARN/FAIL ``verdict`` against ``thresholds`` (defaults below), and
    per-pair provenance.
    """
    thresholds = thresholds or {"bleu_min": 20.0, "chrf_min": 40.0, "term_match_rate_min": 0.80}
    if not pairs:
        return {"benchmark": "translation", "model": model_cfg.name, "skipped": True,
                "reason": "no pairs"}

    src_lang = pairs[0].src_lang
    tgt_lang = pairs[0].tgt_lang
    srcs = [p.src for p in pairs]
    refs = [p.ref for p in pairs]
    glossaries = [p.glossary for p in pairs]

    hyps = translate_batch(
        model_cfg, srcs, src_lang, tgt_lang,
        level=level,
        glossary=glossaries[0] if (level == "l3" and glossaries) else None,
    )

    bleu = compute_sacrebleu(hyps, refs, tgt_lang)
    chrf = compute_chrf(hyps, refs)
    comet = compute_comet(hyps, srcs, refs) if run_comet else {"available": False, "reason": "disabled"}
    terms = term_match_rate(hyps, glossaries) if level == "l3" else None

    validation = validate_metrics(hyps, bleu, chrf)
    nonempty = sum(1 for h in hyps if h)

    aggregate = {
        "level": level,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "num_pairs": len(pairs),
        "nonempty_hyps": nonempty,
        "empty_rate": 1 - nonempty / len(pairs),
        "bleu": bleu,
        "chrf": chrf,
        "comet": comet,
        "terminology": terms,
        "data_source": pairs[0].source,
    }

    reasons: list[str] = []
    if validation:
        reasons.append("FAIL: numerical validation: " + "; ".join(validation))
    if bleu < thresholds.get("bleu_min", 0):
        reasons.append(f"FAIL: BLEU {bleu:.1f} < {thresholds['bleu_min']}")
    if chrf < thresholds.get("chrf_min", 0):
        reasons.append(f"FAIL: chrF {chrf:.1f} < {thresholds['chrf_min']}")
    if level == "l3" and terms and terms["term_match_rate"] < thresholds.get("term_match_rate_min", 0):
        reasons.append(
            f"FAIL: term-match {terms['term_match_rate']*100:.0f}% < "
            f"{thresholds['term_match_rate_min']*100:.0f}%"
        )

    verdict = "FAIL" if any(r.startswith("FAIL") for r in reasons) else (
        "WARN" if reasons else "PASS"
    )

    return {
        "benchmark": "translation",
        "model": model_cfg.name,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "aggregate": aggregate,
        "per_pair": [
            {"src": p.src[:60], "ref": p.ref[:60], "hyp": h[:60], "source": p.source}
            for p, h in zip(pairs, hyps)
        ],
    }
