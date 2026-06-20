"""S2: 案件叙述前后逻辑一致性 + 矛盾点检出(LLM,纯文本)。

case payload: {"segments": [str], "golden_findings": [{"kind", "segments"}],
               "consistency_label": "consistent|minor_issues|contradictory"}
"""
from __future__ import annotations

from .base import ScenarioCase, ScenarioSpec

FINDING_KINDS = ("time_conflict", "causal_break", "fact_mismatch")

_PROMPT_HEAD = """你是案件材料审查助手。下面是同一案件的多段陈述(编号 [0]..[N])。
找出段落之间的逻辑问题,严格输出 JSON(只输出 JSON):

{
  "consistency": "consistent | minor_issues | contradictory",
  "findings": [{"kind": "time_conflict | causal_break | fact_mismatch",
                "segments": [涉及的段落编号], "explain": "一句话"}]
}

陈述段落:
"""


def _build_prompt(case: ScenarioCase):
    body = "\n".join(f"[{i}] {s}" for i, s in enumerate(case.payload["segments"]))
    return _PROMPT_HEAD + body, None


def _norm(findings) -> set:
    out = set()
    for f in findings or []:
        try:
            out.add((str(f.get("kind")), frozenset(int(x) for x in f.get("segments", []))))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _l1_score(case: ScenarioCase, parsed: dict | None, raw_content: str | None = None) -> dict:
    if not isinstance(parsed, dict):
        return {"label_hit": 0, "finding_f1": 0.0}
    golden = _norm(case.payload.get("golden_findings"))
    got = _norm(parsed.get("findings"))
    tp = len(golden & got)
    p = tp / len(got) if got else (1.0 if not golden else 0.0)
    r = tp / len(golden) if golden else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "label_hit": int(parsed.get("consistency") == case.payload["consistency_label"]),
        "finding_f1": f1,
    }


def _aggregate(per_case: list[dict]) -> dict:
    n = len(per_case) or 1
    return {
        "label_accuracy": sum(c["label_hit"] for c in per_case) / n,
        "finding_f1": sum(c["finding_f1"] for c in per_case) / n,
    }


JUDGE_RUBRIC = """你是评测裁判。给定案件陈述段落、标准矛盾点与被测模型输出,按 1-5 分评分:
5=矛盾点全中且解释准确;4=主要矛盾点命中、解释可用;3=命中部分矛盾点;
2=一致性判断对但矛盾点基本没找到;1=一致性判断错误。
严格输出 JSON: {"score": <1-5 整数>, "rationale": "一句话理由"}"""

SPEC = ScenarioSpec(
    name="case_logic",
    cases_path="datasets/scenarios/case_logic/cases.jsonl",
    build_prompt=_build_prompt,
    l1_score=_l1_score,
    aggregate_l1=_aggregate,
    judge_rubric=JUDGE_RUBRIC,
    requires_vlm=False,
    default_thresholds={"label_accuracy_min": 0.60, "finding_f1_min": 0.50},
    payload_required_fields=["segments", "golden_findings", "consistency_label"],
)
