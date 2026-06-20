"""S1: 微信聊天截图 → 内容抽取 + 聊天意图判定(VLM)。

case payload: {"image": <仓根相对路径>, "expected_intent": <八类标签之一>,
               "expected_entities": [str], "context_hint": str(可选)}
"""
from __future__ import annotations

from pathlib import Path

from .base import ScenarioCase, ScenarioSpec

INTENT_LABELS = (
    "loan_agreement", "debt_acknowledgment", "threat", "promise",
    "denial", "negotiation", "irrelevant", "harassment",
)

_PROMPT = """你是法律证据分析助手。仔细阅读这张微信聊天截图,严格按 JSON 输出(只输出 JSON):

{
  "intent": "%s 之一",
  "entities": ["人名/金额/日期/关键物品等实体"],
  "summary": "一句话概括聊天内容",
  "intent_evidence": "判定意图的关键原文片段"
}""" % " | ".join(INTENT_LABELS)


def _build_prompt(case: ScenarioCase):
    return _PROMPT, Path(case.payload["image"])


def _l1_score(case: ScenarioCase, parsed: dict | None, raw_content: str | None = None) -> dict:
    if not isinstance(parsed, dict):
        return {"intent_hit": 0, "entity_recall": 0.0}
    expected_entities = [e.strip() for e in case.payload.get("expected_entities", [])]
    got = " ".join(str(e) for e in (parsed.get("entities") or []))
    hit = sum(1 for e in expected_entities if e and e in got)
    return {
        "intent_hit": int(parsed.get("intent") == case.payload["expected_intent"]),
        "entity_recall": (hit / len(expected_entities)) if expected_entities else 1.0,
    }


def _aggregate(per_case: list[dict]) -> dict:
    n = len(per_case) or 1
    return {
        "intent_accuracy": sum(c["intent_hit"] for c in per_case) / n,
        "entity_recall": sum(c["entity_recall"] for c in per_case) / n,
    }


JUDGE_RUBRIC = """你是评测裁判。给定微信截图场景的期望意图/实体与被测模型输出,按 1-5 分评分:
5=意图正确且实体完整、概括准确;4=意图正确、细节小瑕疵;3=意图正确但实体/概括有明显遗漏;
2=意图错误但内容抽取基本可用;1=意图错误且抽取混乱。
严格输出 JSON: {"score": <1-5 整数>, "rationale": "一句话理由"}"""

SPEC = ScenarioSpec(
    name="wechat_intent",
    cases_path="datasets/scenarios/wechat_intent/cases.jsonl",
    build_prompt=_build_prompt,
    l1_score=_l1_score,
    aggregate_l1=_aggregate,
    judge_rubric=JUDGE_RUBRIC,
    requires_vlm=True,
    default_thresholds={"intent_accuracy_min": 0.70, "entity_recall_min": 0.60},
    payload_required_fields=["image", "expected_intent"],
)
