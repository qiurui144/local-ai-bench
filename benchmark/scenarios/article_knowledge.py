"""S3: 自媒体文章知识性评估(LLM,纯文本)。

case payload: {"text", "source_url", "golden_claims": [{"claim","label"}],
               "knowledge_grade": "A|B|C|D"}
跑分只用 text 快照,source_url 仅作 provenance,绝不在跑分时拉取。
"""
from __future__ import annotations

from .base import ScenarioCase, ScenarioSpec

GRADES = ("A", "B", "C", "D")
CLAIM_LABELS = ("accurate", "inaccurate", "unverifiable")

_PROMPT_TMPL = """你是内容可信度分析助手。阅读文章,对列出的每条声明判定准确性,并给文章知识质量评级。
严格输出 JSON(只输出 JSON):

{
  "claims": [{"claim": "原样复述该声明", "label": "accurate | inaccurate | unverifiable"}],
  "grade": "A(严谨可信) | B(基本可靠) | C(夸大误导倾向) | D(明显错误信息)",
  "rationale": "一句话评级理由"
}

待判定声明:
%s

文章全文:
%s"""


def _build_prompt(case: ScenarioCase):
    claims = "\n".join(f"- {c['claim']}" for c in case.payload["golden_claims"])
    return _PROMPT_TMPL % (claims, case.payload["text"]), None


def _grade_score(got, expected) -> float:
    if got not in GRADES:
        return 0.0
    dist = abs(GRADES.index(got) - GRADES.index(expected))
    return {0: 1.0, 1: 0.5}.get(dist, 0.0)


def _l1_score(case: ScenarioCase, parsed: dict | None, raw_content: str | None = None) -> dict:
    if not isinstance(parsed, dict):
        return {"claim_accuracy": 0.0, "grade_score": 0.0}
    golden = case.payload["golden_claims"]
    got_by_text = {str(c.get("claim", "")).strip(): c.get("label")
                   for c in (parsed.get("claims") or []) if isinstance(c, dict)}
    hit = sum(1 for g in golden if got_by_text.get(g["claim"].strip()) == g["label"])
    return {
        "claim_accuracy": hit / len(golden) if golden else 1.0,
        "grade_score": _grade_score(parsed.get("grade"), case.payload["knowledge_grade"]),
    }


def _aggregate(per_case: list[dict]) -> dict:
    n = len(per_case) or 1
    return {
        "claim_accuracy": sum(c["claim_accuracy"] for c in per_case) / n,
        "grade_score": sum(c["grade_score"] for c in per_case) / n,
    }


JUDGE_RUBRIC = """你是评测裁判。给定文章、标准声明判定与被测模型输出,按 1-5 分评分:
5=声明判定全对且评级一致;4=声明基本对、评级相邻;3=声明半数对;
2=评级方向对但声明判定差;1=判定与评级均不可用。
严格输出 JSON: {"score": <1-5 整数>, "rationale": "一句话理由"}"""

SPEC = ScenarioSpec(
    name="article_knowledge",
    cases_path="datasets/scenarios/article_knowledge/cases.jsonl",
    build_prompt=_build_prompt,
    l1_score=_l1_score,
    aggregate_l1=_aggregate,
    judge_rubric=JUDGE_RUBRIC,
    requires_vlm=False,
    default_thresholds={"claim_accuracy_min": 0.60, "grade_score_min": 0.50},
    payload_required_fields=["text", "golden_claims"],
)
