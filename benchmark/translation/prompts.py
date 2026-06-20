"""Translation prompt templates (L1 / L2 / L3 task levels).

Three difficulty levels, each exercising a different capability:

- **L1 single-sentence** — straight zh<->en sentence translation. The
  baseline; measures raw adequacy + fluency.
- **L2 multi-sentence context consistency** — 3-5 sentences translated as a
  block so pronoun reference and tense stay consistent across sentence
  boundaries (a common failure mode of sentence-at-a-time MT).
- **L3 terminology** — domain text where specific technical terms
  (``prompt`` / ``embedding`` / ``向量化`` ...) must be rendered with the
  caller-supplied canonical translation. Scored by exact-match term rate.

Each builder returns a single user-message string. We deliberately ask the
model to emit *only* the translation (no preamble / no quotes) so the raw
completion can be fed straight into SacreBLEU / chrF without post-stripping.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Human-readable language names for the instruction line. Falls back to the
# raw code so unusual pairs still produce a sensible prompt.
LANG_NAMES = {
    "zh": "中文 (Chinese)",
    "en": "英文 (English)",
    "ja": "日文 (Japanese)",
    "ko": "韩文 (Korean)",
    "fr": "法文 (French)",
    "de": "德文 (German)",
    "es": "西班牙文 (Spanish)",
}


def _lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


def l1_single_sentence(src: str, src_lang: str, tgt_lang: str) -> str:
    """L1: translate one sentence; output the translation only."""
    return (
        f"Translate the following {_lang_name(src_lang)} text into "
        f"{_lang_name(tgt_lang)}. Output ONLY the translation, no quotes, "
        f"no explanation.\n\n{src}"
    )


def l2_context_consistency(
    sentences: Sequence[str], src_lang: str, tgt_lang: str
) -> str:
    """L2: translate a short passage as a coherent block.

    ``sentences`` is a list of 3-5 source sentences. The model is asked to
    keep pronouns, tense, and named entities consistent across the passage and
    to return one translated sentence per line, preserving order.
    """
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    return (
        f"Translate the following {_lang_name(src_lang)} passage into "
        f"{_lang_name(tgt_lang)}. Keep pronoun reference, tense, and named "
        f"entities CONSISTENT across all sentences. Return one translated "
        f"sentence per line in the SAME order, numbered the same way. Output "
        f"ONLY the numbered translations.\n\n{numbered}"
    )


def l3_terminology(
    src: str,
    src_lang: str,
    tgt_lang: str,
    glossary: Mapping[str, str],
) -> str:
    """L3: translate while preserving a required terminology mapping.

    ``glossary`` maps a source term to its required target rendering, e.g.
    ``{"向量化": "vectorization", "提示词": "prompt"}``. The exact target
    strings must appear verbatim in the output (scored by term-match rate).
    """
    glossary_lines = "\n".join(f"  - {k} -> {v}" for k, v in glossary.items())
    return (
        f"Translate the following {_lang_name(src_lang)} technical text into "
        f"{_lang_name(tgt_lang)}. You MUST render these terms exactly as "
        f"specified (do not paraphrase them):\n{glossary_lines}\n\n"
        f"Output ONLY the translation.\n\n{src}"
    )
