"""Translation dataset loaders.

Two sources, both returning a uniform list of ``TranslationPair`` records so
``accuracy.run_translation`` / ``performance`` can iterate them identically:

- :func:`load_flores` — Flores-200 zh<->en parallel sentences via the HF
  ``facebook/flores`` dataset (``devtest`` split). When ``datasets`` or the
  network is unavailable it falls back to a small **built-in** parallel set so
  unit tests and offline smoke runs still work. The built-in fallback is
  **synthetic / hand-authored** and is flagged as such (``source="builtin"``).

- :func:`load_custom_jsonl` — a product-domain parallel corpus from a JSONL
  file, one object per line: ``{"src": ..., "tgt": ..., "domain": ...}``.
  Optional ``glossary`` (``{src_term: tgt_term}``) enables L3 term scoring.

Flores-200 language codes used here: ``zho_Hans`` (Simplified Chinese) and
``eng_Latn`` (English). The public ``facebook/flores`` config name for a pair
is the two codes joined by ``-`` (e.g. ``zho_Hans-eng_Latn``); each row then
exposes ``sentence_<code>`` columns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

FLORES_CODE = {"zh": "zho_Hans", "en": "eng_Latn"}


def _offline() -> bool:
    """Skip the HF download entirely (fail fast to the built-in fallback).

    Honours ``TRANSLATION_OFFLINE`` / ``HF_HUB_OFFLINE`` / ``HF_DATASETS_OFFLINE``
    so air-gapped GPU hosts and CI don't pay HF's multi-second retry loop.
    """
    return any(
        os.environ.get(k, "").lower() in ("1", "true", "yes")
        for k in ("TRANSLATION_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    )


@dataclass
class TranslationPair:
    """One source/reference parallel record."""

    src: str
    ref: str
    src_lang: str
    tgt_lang: str
    domain: str = "general"
    source: str = "flores"            # provenance: flores | builtin | custom
    glossary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in offline fallback (SYNTHETIC — hand-authored, not from Flores).
# Used only when HF datasets / network is unavailable. Kept tiny on purpose.
# ---------------------------------------------------------------------------
_BUILTIN_ZH = [
    "今天天气很好，我们去公园散步吧。",
    "这家公司去年的营业收入增长了百分之十五。",
    "请把会议时间改到明天下午三点。",
    "他正在学习如何使用新的机器学习框架。",
    "数据库连接失败，请检查网络配置。",
]
_BUILTIN_EN = [
    "The weather is nice today; let's go for a walk in the park.",
    "The company's revenue grew by fifteen percent last year.",
    "Please move the meeting to three o'clock tomorrow afternoon.",
    "He is learning how to use the new machine learning framework.",
    "The database connection failed; please check the network configuration.",
]


def _builtin_pairs(src_lang: str, tgt_lang: str) -> list[TranslationPair]:
    if (src_lang, tgt_lang) == ("zh", "en"):
        srcs, refs = _BUILTIN_ZH, _BUILTIN_EN
    elif (src_lang, tgt_lang) == ("en", "zh"):
        srcs, refs = _BUILTIN_EN, _BUILTIN_ZH
    else:
        raise ValueError(f"builtin fallback only supports zh<->en, got {src_lang}->{tgt_lang}")
    return [
        TranslationPair(src=s, ref=r, src_lang=src_lang, tgt_lang=tgt_lang, source="builtin")
        for s, r in zip(srcs, refs)
    ]


def load_flores(
    src_lang: str = "zh",
    tgt_lang: str = "en",
    split: str = "devtest",
    num_samples: Optional[int] = 100,
) -> list[TranslationPair]:
    """Load a Flores-200 parallel subset; fall back to built-in pairs offline.

    Parameters
    ----------
    src_lang, tgt_lang : ``"zh"`` or ``"en"``.
    split : Flores split (``dev`` or ``devtest``; ``devtest`` is the standard
        eval split).
    num_samples : cap the number of pairs (``None`` = all). Defaults to 100 to
        keep a benchmark run cheap.
    """
    if src_lang not in FLORES_CODE or tgt_lang not in FLORES_CODE:
        raise ValueError(f"load_flores supports zh/en only, got {src_lang}->{tgt_lang}")

    if _offline():
        pairs = _builtin_pairs(src_lang, tgt_lang)
    else:
        try:
            pairs = _load_flores_hf(src_lang, tgt_lang, split)
        except Exception:
            pairs = _builtin_pairs(src_lang, tgt_lang)

    if num_samples is not None:
        pairs = pairs[:num_samples]
    return pairs


def _load_flores_hf(src_lang: str, tgt_lang: str, split: str) -> list[TranslationPair]:
    from datasets import load_dataset  # imported lazily; absence -> fallback

    src_code, tgt_code = FLORES_CODE[src_lang], FLORES_CODE[tgt_lang]
    config = f"{src_code}-{tgt_code}"
    ds = load_dataset("facebook/flores", config, split=split, trust_remote_code=True)
    src_col, tgt_col = f"sentence_{src_code}", f"sentence_{tgt_code}"
    return [
        TranslationPair(
            src=row[src_col],
            ref=row[tgt_col],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            source="flores",
        )
        for row in ds
    ]


def load_custom_jsonl(
    path: Path | str,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    num_samples: Optional[int] = None,
) -> list[TranslationPair]:
    """Load a product-domain parallel corpus from JSONL.

    Each line is a JSON object with at least ``src`` and ``tgt``; optional
    ``domain`` (free-form tag) and ``glossary`` (``{src_term: tgt_term}`` for
    L3 terminology scoring).
    """
    path = Path(path)
    pairs: list[TranslationPair] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(
                TranslationPair(
                    src=obj["src"],
                    ref=obj["tgt"],
                    src_lang=obj.get("src_lang", src_lang),
                    tgt_lang=obj.get("tgt_lang", tgt_lang),
                    domain=obj.get("domain", "general"),
                    source="custom",
                    glossary=obj.get("glossary", {}) or {},
                )
            )
    if num_samples is not None:
        pairs = pairs[:num_samples]
    return pairs
