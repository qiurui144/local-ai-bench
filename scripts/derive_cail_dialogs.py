#!/usr/bin/env python3
"""Derive S1 wechat_intent dispute dialogs inspired by real CAIL2018 narratives.

Method (deterministic, idempotent)
==================================
1. Source narratives: ``china-ai-law-challenge/cail2018``, split
   ``exercise_contest_valid``, pinned revision
   ``775098da3ba75f033781f8061900b62503e9bea0`` — same pin, loading idiom and
   "某"-anonymization guard as scripts/derive_cail_cases.py. Candidates were
   scanned deterministically in row order with the filter: index disjoint from
   the 15 already consumed by datasets/scenarios/case_logic/cases.jsonl
   (1,4,5,6,7,10,15,17,20,21,23,24,27,28,29); fact contains "某" (CAIL court
   narratives are published pre-anonymized — requiring "某" is the PII guard);
   fact mentions a loan/debt/threat/fraud dispute (any of 借/欠/还款/恐吓/诈骗);
   80 <= len(fact) <= 500. The first 10 qualifying rows usable as two-party
   dispute dialogs were kept: indices 84, 96, 170, 291, 363, 430, 539, 543,
   618, 790.
2. Each dialog is HAND-WRITTEN — a literal mapping in DERIVED below, not
   template-generated: a 2-4 message WeChat-style exchange that the
   narrative's parties could plausibly have had, expressing exactly one
   intent from benchmark/scenarios/wechat_intent.py INTENT_LABELS. Party
   names stay anonymized (张某/李某 style, as in the source narratives).
3. HONESTY NOTE: output provenance stays "synthetic". A dialog derived from
   (inspired by) a court narrative is authored text, NOT real chat data, and
   must never be claimed otherwise. Traceability to the inspiring narrative
   is kept in payload.source = {"inspired_by", "revision", "index"}.
4. Validation before writing: every intent is a valid INTENT_LABELS member;
   every expected_entities item is a verbatim substring of the dialog text;
   every pinned source row still contains "某" plus a dispute keyword.
5. Idempotent: ids already present in dialogs.json / cases.jsonl are skipped
   on re-run (and the dataset is not even downloaded if nothing is new).

After running, re-render fixtures:  python scripts/render_wechat_case.py

Run:  HF_HOME=/tmp/hf-... HF_HUB_DISABLE_XET=1 \
          python scripts/derive_cail_dialogs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIALOGS_PATH = ROOT / "datasets/scenarios/wechat_intent/dialogs.json"
CASES_PATH = ROOT / "datasets/scenarios/wechat_intent/cases.jsonl"

REPO_ID = "china-ai-law-challenge/cail2018"
REVISION = "775098da3ba75f033781f8061900b62503e9bea0"
SPLIT = "exercise_contest_valid"
KEYWORDS = ("借", "欠", "还款", "恐吓", "诈骗")

# Hand-written dialogs (see module docstring, items 2-3). One entry per case:
# (id, source row index, intent, difficulty, expected_entities, messages).
DERIVED: list[dict] = [
    {"id": "c16", "index": 84, "intent": "debt_acknowledgment",
     "difficulty": "normal", "entities": ["郭某", "5024"],
     "messages": [
         ("left", "郭某，吃夜宵那晚你借我手机打电话，到现在人和手机都不见了"),
         ("right", "对不住，手机被我处理了，我按5024元原价赔你，一分不少"),
         ("left", "行，赔清了这事就算了"),
     ]},
    {"id": "c17", "index": 96, "intent": "loan_agreement",
     "difficulty": "normal", "entities": ["潘某", "越野车"],
     "messages": [
         ("left", "潘某，下午我要去工地办事，你的越野车借我用两个小时"),
         ("right", "可以，钥匙在我办公桌上，用完帮我停回原位"),
     ]},
    {"id": "c18", "index": 170, "intent": "promise",
     "difficulty": "normal", "entities": ["6900", "26000"],
     "messages": [
         ("left", "我支付宝里6900元被你转走，借呗还多出26000的借款，怎么回事"),
         ("right", "对不起，是我一时糊涂，这些钱我一定想办法全部赔给你"),
         ("left", "我等你消息，别让我再催"),
     ]},
    {"id": "c19", "index": 291, "intent": "denial",
     "difficulty": "hard", "entities": ["7110"],
     "messages": [
         ("left", "我银行卡少了7110元，那几天手机正好借给你用过"),
         ("right", "跟我没关系，我从来没动过你的卡，别冤枉人"),
     ]},
    {"id": "c20", "index": 363, "intent": "negotiation",
     "difficulty": "normal", "entities": ["房某"],
     "messages": [
         ("left", "房某，工程欠款拖了快一年，今天必须给个说法"),
         ("right", "公司资金确实紧张，先结一半，剩下的年前付清行不行"),
         ("left", "最多宽限到下个月，到期一分都不能少"),
         ("right", "行，就这么定"),
     ]},
    {"id": "c21", "index": 430, "intent": "threat",
     "difficulty": "normal", "entities": ["张某"],
     "messages": [
         ("left", "张某，你欠我的钱明天我亲自上门来拿"),
         ("right", "你敢上我家门闹，别怪我不客气，打到你进医院"),
     ]},
    {"id": "c22", "index": 539, "intent": "loan_agreement",
     "difficulty": "normal", "entities": ["5000"],
     "messages": [
         ("left", "弟，最近手头实在太紧，能不能周转我5000元"),
         ("right", "可以，5000我今天就汇给你，算你借我的，以后有了再还"),
         ("left", "好，这笔账我记着，一定还你"),
     ]},
    {"id": "c23", "index": 543, "intent": "debt_acknowledgment",
     "difficulty": "normal", "entities": ["林某甲", "84000"],
     "messages": [
         ("left", "林某甲，我住院这些天的医药费加起来一共84000元"),
         ("right", "人是我打伤的，这84000元我认下，全部由我来赔"),
     ]},
    {"id": "c24", "index": 618, "intent": "denial",
     "difficulty": "hard", "entities": ["陈某", "苹果手机"],
     "messages": [
         ("left", "陈某，在包厢你说借我手机打个电话，怎么拿着我的苹果手机就走了"),
         ("right", "谁拿你手机了？我根本没碰过你的苹果手机"),
         ("left", "当晚就你一个人借过手机，别装了"),
         ("right", "没有就是没有，你别赖我"),
     ]},
    {"id": "c25", "index": 790, "intent": "threat",
     "difficulty": "hard", "entities": ["文某", "80元"],
     "messages": [
         ("left", "文某，那80元欠款拖了这么久，今天必须还我"),
         ("right", "就80块你也天天催，再逼我，小心我带刀来找你"),
     ]},
]


def _intent_labels() -> tuple[str, ...]:
    """Canonical labels; cross-checked against the scenario module if present."""
    labels = ("loan_agreement", "debt_acknowledgment", "threat", "promise",
              "denial", "negotiation", "irrelevant", "harassment")
    try:
        sys.path.insert(0, str(ROOT))
        from benchmark.scenarios.wechat_intent import INTENT_LABELS
    except ImportError:
        return labels
    if tuple(INTENT_LABELS) != labels:
        raise RuntimeError("INTENT_LABELS drifted; update derive_cail_dialogs.py")
    return tuple(INTENT_LABELS)


def validate(entries: list[dict]) -> None:
    labels = _intent_labels()
    for e in entries:
        text = "".join(t for _, t in e["messages"])
        if e["intent"] not in labels:
            raise ValueError(f"{e['id']}: bad intent {e['intent']!r}")
        if not 2 <= len(e["messages"]) <= 4:
            raise ValueError(f"{e['id']}: need 2-4 messages")
        if e["difficulty"] not in ("normal", "hard"):
            raise ValueError(f"{e['id']}: bad difficulty {e['difficulty']!r}")
        for ent in e["entities"]:
            if ent not in text:
                raise ValueError(f"{e['id']}: entity {ent!r} not verbatim in dialog")


def verify_sources(entries: list[dict]) -> None:
    """Re-check the pinned narrative rows: anonymized + dispute keyword."""
    from datasets import load_dataset

    ds = load_dataset(REPO_ID, split=SPLIT, revision=REVISION)
    for e in entries:
        fact = ds[e["index"]]["fact"]
        if "某" not in fact:
            raise RuntimeError(f"{e['id']}: source row {e['index']} not anonymized")
        if not any(k in fact for k in KEYWORDS):
            raise RuntimeError(f"{e['id']}: source row {e['index']} lacks keyword")


def _fmt_dialog(e: dict) -> str:
    """Serialize one dialogs.json entry in the file's existing layout."""
    msgs = []
    for side, text in e["messages"]:
        pad = "  " if side == "left" else " "
        msgs.append(f'    {{"side": "{side}",{pad}"text": '
                    f'{json.dumps(text, ensure_ascii=False)}}}')
    return f'  {{"id": "{e["id"]}", "messages": [\n' + ",\n".join(msgs) + "]}"


def main() -> None:
    dialogs_ids = {d["id"] for d in
                   json.loads(DIALOGS_PATH.read_text(encoding="utf-8"))}
    with open(CASES_PATH, encoding="utf-8") as f:
        case_ids = {json.loads(ln)["id"] for ln in f if ln.strip()}
    new = [e for e in DERIVED
           if e["id"] not in dialogs_ids and e["id"] not in case_ids]
    skipped = [e["id"] for e in DERIVED if e not in new]
    if skipped:
        print("skip existing:", ", ".join(skipped))
    if not new:
        print("nothing to do")
        return

    validate(new)
    verify_sources(new)

    # Textual append keeps the existing c1-c15 lines byte-identical.
    body = DIALOGS_PATH.read_text(encoding="utf-8").rstrip()
    if not body.endswith("]"):
        raise RuntimeError("dialogs.json: unexpected trailing content")
    body = body[:-1].rstrip()
    body += ",\n" + ",\n".join(_fmt_dialog(e) for e in new) + "\n]\n"
    DIALOGS_PATH.write_text(body, encoding="utf-8")

    with open(CASES_PATH, "a", encoding="utf-8") as f:
        for e in new:
            row = {
                "id": e["id"], "provenance": "synthetic",
                "difficulty": e["difficulty"],
                "payload": {
                    "image": f"fixtures/scenarios/wechat_intent/{e['id']}.png",
                    "expected_intent": e["intent"],
                    "expected_entities": e["entities"],
                    "source": {"inspired_by": REPO_ID, "revision": REVISION,
                               "index": e["index"]},
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"added {len(new)} dialogs: {', '.join(e['id'] for e in new)}")
    print("next: python scripts/render_wechat_case.py")


if __name__ == "__main__":
    main()
