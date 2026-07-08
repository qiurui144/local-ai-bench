"""确定性长上下文语料:CAIL2018 案情段落拼接 + 指定深度埋点。

数据 pin 与 scripts/derive_cail_cases.py 同源(同 repo/revision/split);
zh token 估算 1 token≈1.6 字符(spec §11,报告标注 approx,运行时用
usage.prompt_tokens 回读校正)。同输入必同输出 — 不引入 random。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ID = "china-ai-law-challenge/cail2018"
REVISION = "775098da3ba75f033781f8061900b62503e9bea0"
SPLIT = "exercise_contest_valid"
CHARS_PER_TOKEN = 1.6
LOCAL_FALLBACK = Path(__file__).parents[2] / "datasets" / "conditioned" / "local_corpus_zh.txt"


@dataclass(frozen=True)
class BuiltContext:
    text: str
    target_tokens: int
    est_tokens: int
    insertions: dict        # fact_id -> realized depth fraction


def load_needles(path: Path) -> Optional[list[dict]]:
    """读 needles.jsonl;缺失返回 None(调用方 BLOCKED,绝不静默空跑)。"""
    path = Path(path)
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for r in rows:
        missing = {"id", "role", "depth", "fact", "question", "answer"} - set(r)
        if missing:
            raise ValueError(f"needle {r.get('id')!r} missing fields: {sorted(missing)}")
    return rows


def _load_local_paragraphs(min_chars: int, limit: int) -> list[str]:
    if not LOCAL_FALLBACK.exists():
        return []
    text = LOCAL_FALLBACK.read_text(encoding="utf-8")
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_chars]
    return paras[:limit]


def load_cail_paragraphs(min_chars: int = 100, limit: int = 2000) -> list[str]:
    """CAIL2018 fact 段落,行序确定性扫描。

    RISC-V edge targets often do not have HF ``datasets``/``pyarrow`` wheels.
    In that case use the checked-in legal-domain fallback corpus so context
    support still produces measured data instead of a dependency-only BLOCKED.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(REPO_ID, split=SPLIT, revision=REVISION)
        paras = [str(row["fact"]).strip() for row in ds
                 if row.get("fact") and len(str(row["fact"]).strip()) >= min_chars]
    except Exception:
        paras = _load_local_paragraphs(min_chars, limit)
    if not paras:
        raise RuntimeError("CAIL2018 facts empty after filtering")
    return paras[:limit]


def build_context(target_tokens: int, facts: list[tuple], paragraphs: list[str]) -> BuiltContext:
    """facts: (fact_id, depth_fraction, fact_sentence)。段落循环拼到 char 预算,
    末段截尾对齐预算;事实句插在最接近 depth 的段落边界(绝不截事实句)。"""
    budget = int(target_tokens * CHARS_PER_TOKEN)
    chunks: list[str] = []
    used = 0
    i = 0
    while used < budget:
        p = paragraphs[i % len(paragraphs)]
        chunks.append(p)
        used += len(p) + 1                      # +1 = join 换行
        i += 1
    overshoot = used - budget
    if overshoot > 0 and len(chunks[-1]) > overshoot:
        chunks[-1] = chunks[-1][:-overshoot]
    insertions: dict = {}
    for fid, depth, sentence in sorted(facts, key=lambda f: float(f[1])):
        pos = min(len(chunks), max(0, round(float(depth) * len(chunks))))
        chunks.insert(pos, sentence)
        insertions[fid] = round(pos / len(chunks), 3)
    text = "\n".join(chunks)
    return BuiltContext(text=text, target_tokens=target_tokens,
                        est_tokens=round(len(text) / CHARS_PER_TOKEN),
                        insertions=insertions)
