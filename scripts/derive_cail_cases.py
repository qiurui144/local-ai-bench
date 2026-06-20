#!/usr/bin/env python3
"""Derive case_logic dataset-track cases from real CAIL2018 fact narratives.

Method (reproducible; provenance="dataset", never fabricated)
=============================================================
1. Source dataset — probed on HF API (2026-06-10): searched "cail" sorted by
   downloads; top hit ``china-ai-law-challenge/cail2018`` is gated=False with
   native parquet shards (no loading script, loads on datasets==4.5.0 without
   trust_remote_code). Pinned revision:
   ``775098da3ba75f033781f8061900b62503e9bea0``.
   We use the small single-shard split ``exercise_contest_valid``; the fact
   narrative lives in the ``fact`` column.
2. Candidate filter (deterministic scan in row order, no random module):
   - 100 <= len(fact) <= 400 chars;
   - fact splits on 。/； into 3-8 sentences (merged contiguously into at
     most 5 segments);
   - fact contains "某" — CAIL2018 court narratives are published already
     anonymized (张某某/王某 style); requiring "某" guarantees we only pick
     pre-anonymized narratives (PII guard).
3. Controlled perturbation injection — exactly ONE per perturbed case, so
   golden_findings are ground truth by construction:
   - time_conflict  (4 cases): pick the first and last segments carrying a
     full event date (YYYY年M月, year >= 2000, dates non-decreasing across
     the narrative); rewrite the later segment's year to (first_year - 2),
     making the consequence pre-date the earlier event.
   - fact_mismatch  (3 cases): pick the first non-final segment carrying a
     plain "N元" amount; append a restatement clause to the final segment
     quoting a deterministically altered amount (first digit + 1).
   - causal_break   (3 cases): swap the last two segments, so the
     consequence precedes its cause in narrative order.
   Buckets are filled greedily in scan order with priority
   time_conflict > fact_mismatch > causal_break > none (rarest first);
   5 cases stay unperturbed (consistent, golden_findings=[]).
4. Labels: unperturbed -> "consistent"; causal_break cases (pure ordering
   issue, no factual contradiction) -> "minor_issues"; time_conflict /
   fact_mismatch -> "contradictory".
5. Traceability: payload["source"] = {dataset, split, revision, index,
   perturbation}; id = "cail2018_<row index>". Idempotent: rows whose id is
   already present in cases.jsonl are skipped on re-run.

Run:  HF_HOME=/tmp/hf-... HF_HUB_DISABLE_XET=1 python scripts/derive_cail_cases.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ID = "china-ai-law-challenge/cail2018"
REVISION = "775098da3ba75f033781f8061900b62503e9bea0"
SPLIT = "exercise_contest_valid"
CASES_PATH = Path(__file__).resolve().parents[1] / (
    "datasets/scenarios/case_logic/cases.jsonl"
)

QUOTA = {"time_conflict": 4, "fact_mismatch": 3, "causal_break": 3, "none": 5}
DATE_RE = re.compile(r"(\d{4})年\d{1,2}月")
AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)元")


def split_segments(fact: str) -> list[str] | None:
    """Split into 3-5 contiguous segments; None if the fact doesn't qualify."""
    sentences = [s.strip() for s in re.split(r"[。；]", fact) if s.strip()]
    if not 3 <= len(sentences) <= 8:
        return None
    if len(sentences) <= 5:
        return sentences
    n, k = len(sentences), 5
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    segs, pos = [], 0
    for size in sizes:
        segs.append("。".join(sentences[pos:pos + size]))
        pos += size
    return segs


def time_anchor(segs: list[str]) -> tuple[int, int, int] | None:
    """(first dated seg, last dated seg, first year) if chronology usable."""
    dated = [(i, int(m.group(1))) for i, s in enumerate(segs)
             if (m := DATE_RE.search(s))]
    if len(dated) < 2:
        return None
    years = [y for _, y in dated]
    if years[0] < 2000 or any(a > b for a, b in zip(years, years[1:])):
        return None
    return dated[0][0], dated[-1][0], years[0]


def amount_anchor(segs: list[str]) -> tuple[int, str] | None:
    """(segment index, amount literal) for the first non-final 'N元'."""
    for i, s in enumerate(segs[:-1]):
        if m := AMOUNT_RE.search(s):
            return i, m.group(1)
    return None


def inject(kind: str, segs: list[str]) -> tuple[list[str], list[int]]:
    """Apply one controlled perturbation; return (segments, golden indices)."""
    segs = list(segs)
    if kind == "time_conflict":
        i, j, first_year = time_anchor(segs)  # capability pre-checked
        m = DATE_RE.search(segs[j])
        segs[j] = segs[j][:m.start(1)] + str(first_year - 2) + segs[j][m.end(1):]
        return segs, [i, j]
    if kind == "fact_mismatch":
        i, amount = amount_anchor(segs)
        head = amount.lstrip("0") or "1"
        altered = str((int(head[0]) % 9) + 1) + head[1:]
        j = len(segs) - 1
        segs[j] += f"，经核实，上述涉案金额实为{altered}元"
        return segs, [i, j]
    if kind == "causal_break":
        segs[-2], segs[-1] = segs[-1], segs[-2]
        return segs, [len(segs) - 2, len(segs) - 1]
    raise ValueError(kind)


def derive() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(REPO_ID, split=SPLIT, revision=REVISION)
    quota = dict(QUOTA)
    picked: list[dict] = []
    for index, row in enumerate(ds):
        if sum(quota.values()) == 0:
            break
        fact = row["fact"].strip()
        if not 100 <= len(fact) <= 400 or "某" not in fact:
            continue
        segs = split_segments(fact)
        if segs is None:
            continue
        if quota["time_conflict"] and time_anchor(segs):
            kind = "time_conflict"
        elif quota["fact_mismatch"] and amount_anchor(segs):
            kind = "fact_mismatch"
        elif quota["causal_break"]:
            kind = "causal_break"
        elif quota["none"]:
            kind = "none"
        else:
            continue
        quota[kind] -= 1
        if kind == "none":
            findings, label = [], "consistent"
        else:
            segs, golden_idx = inject(kind, segs)
            findings = [{"kind": kind, "segments": golden_idx}]
            # causal_break = 纯叙述顺序问题、无事实矛盾 → 一律 minor_issues;
            # 含事实/时间矛盾的扰动才算 contradictory(标签由扰动类型决定,
            # 不依赖扫描顺序)。
            label = "minor_issues" if kind == "causal_break" else "contradictory"
        picked.append({
            "id": f"cail2018_{index}",
            "provenance": "dataset",
            "payload": {
                "segments": segs,
                "golden_findings": findings,
                "consistency_label": label,
                "source": {"dataset": REPO_ID, "split": SPLIT,
                           "revision": REVISION, "index": index,
                           "perturbation": kind},
            },
        })
    if sum(quota.values()):
        raise RuntimeError(f"quota not filled: {quota}")
    return picked


def main() -> None:
    existing = set()
    if CASES_PATH.exists():
        with open(CASES_PATH, encoding="utf-8") as f:
            existing = {json.loads(ln)["id"] for ln in f if ln.strip()}
    new = [c for c in derive() if c["id"] not in existing]
    if not new:
        print("all derived ids already present; nothing to do")
        return
    with open(CASES_PATH, "a", encoding="utf-8") as f:
        for c in new:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"appended {len(new)} cases -> {CASES_PATH}")


if __name__ == "__main__":
    main()
