"""Translation dataset loaders.

Two sources, both returning a uniform list of ``TranslationPair`` records so
``accuracy.run_translation`` / ``performance`` can iterate them identically:

- :func:`load_flores` — Flores-200 zh<->en parallel sentences via the HF
  ``haoranxu/FLORES-200`` parquet mirror (devtest split, non-gated, no
  ``trust_remote_code``; dataset/revision overridable via ``FLORES_DATASET``
  / ``FLORES_REVISION``). When ``datasets`` or the network is unavailable it
  falls back to a small **built-in** parallel set so unit tests and offline
  smoke runs still work. The built-in fallback is **synthetic / hand-authored**,
  flagged as ``source="builtin"``, logged loudly, and capped at WARN by the
  harness — it never masquerades as a Flores-200 score.

- :func:`load_custom_jsonl` — a product-domain parallel corpus from a JSONL
  file, one object per line: ``{"src": ..., "tgt": ..., "domain": ...}``.
  Optional ``glossary`` (``{src_term: tgt_term}``) enables L3 term scoring.

``haoranxu/FLORES-200`` (the ALMA paper's eval mirror) keys pairs by short
codes: config ``"zh-en"`` / ``"en-zh"``, each row ``{"zh-en": {"zh": ..,
"en": ..}}``, single ``test`` split == the canonical Flores-200 devtest
(1012 sentences). The canonical ``zho_Hans``/``eng_Latn`` codes are kept in
``FLORES_CODE`` for reference only.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FLORES_CODE = {"zh": "zho_Hans", "en": "eng_Latn"}

# 数据源选型(2026-06-10 实测,datasets==4.5.0,无 token):
# - facebook/flores: gated,未授权一律 403
# - Muennighoff/flores200: script-only,datasets>=3 已无法加载(parquet 分支无数据)
# - haoranxu/FLORES-200(ALMA 论文 eval 镜像): 非 gated 纯 parquet,可直接加载 ✓
# 纯 parquet = 不携带 loading script,无 trust_remote_code 供应链代码执行面;
# revision pin 到 commit SHA 保数据完整性。均可用 env 覆盖。
_FLORES_DEFAULT_DATASET = "haoranxu/FLORES-200"
_FLORES_DEFAULT_REVISION = "8ecaf1bb2034f167c2520c419c6f6e28f0098c3f"


def _flores_dataset() -> str:
    return os.environ.get("FLORES_DATASET") or _FLORES_DEFAULT_DATASET


def _flores_revision() -> str:
    # `or` 而非 get(k, default):env 置空串时也回落默认值
    return os.environ.get("FLORES_REVISION") or _FLORES_DEFAULT_REVISION


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
        logger.warning(
            "offline mode: using %d builtin synthetic pairs — results are NOT Flores-200 scores",
            len(pairs),
        )
    else:
        try:
            pairs = _load_flores_hf(src_lang, tgt_lang, split)
        except Exception as e:
            pairs = _builtin_pairs(src_lang, tgt_lang)
            logger.warning(
                "Flores-200 load failed (%s: %s); falling back to %d builtin synthetic "
                "pairs — results are NOT Flores-200 scores",
                type(e).__name__, e, len(pairs),
            )

    if num_samples is not None:
        pairs = pairs[:num_samples]
    return pairs


def _load_flores_hf(src_lang: str, tgt_lang: str, split: str) -> list[TranslationPair]:
    from datasets import load_dataset  # imported lazily; absence -> fallback

    # haoranxu/FLORES-200 结构:config = "zh-en" / "en-zh",仅 "test" split
    # (即 Flores-200 标准 devtest,1012 句),行格式 {"zh-en": {"zh": ..., "en": ...}}。
    if split not in ("devtest", "test"):
        raise ValueError(
            f"{_flores_dataset()} only serves the devtest split, got {split!r}"
        )
    config = f"{src_lang}-{tgt_lang}"
    ds = load_dataset(_flores_dataset(), config, split="test",
                      revision=_flores_revision())
    return [
        TranslationPair(
            src=row[config][src_lang],
            ref=row[config][tgt_lang],
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
