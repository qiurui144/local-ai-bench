"""S4: 指令遵循 (instruction_following) — IFEval 风格的格式约束程序化验证。

业界广泛采用（IFEval / FollowBench），测试模型是否严格遵守显式格式要求。
L1 评分完全程序化，无需 judge，且稳定可复现。

case payload:
  prompt        : str              含显式格式指令的任务描述
  instructions  : list[{type, value}]  程序化验证的约束列表

支持的 type:
  json_valid          value=null   输出可被 json.loads() 解析
  json_has_keys       value=[k..]  顶层 dict 或数组首项含所有指定键
  starts_with         value=str    strip 后精确以 value 开头
  ends_with           value=str    rstrip 后精确以 value 结尾
  must_include        value=str    大小写不敏感子串存在
  must_exclude        value=str    大小写不敏感子串不存在
  bullet_items_min    value=int    至少 N 行以 "- "/"• "/"* " 开头
  numbered_items_min  value=int    至少 N 行匹配 "^\\d+[.。)\\]] "
  char_count_max      value=int    len(strip 后输出) <= value
  char_count_min      value=int    len(strip 后输出) >= value

L1 = per_instruction_compliance_rate（满足指令数 / 总指令数）
聚合 = mean(compliance_rate) over all cases
"""
from __future__ import annotations

import json
import re

from .base import ScenarioCase, ScenarioSpec


def _check(output: str, inst: dict) -> bool:
    t = inst["type"]
    v = inst.get("value")
    s = output.strip()

    if t == "json_valid":
        try:
            json.loads(s)
            return True
        except Exception:
            return False

    if t == "json_has_keys":
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            return isinstance(obj, dict) and all(k in obj for k in v)
        except Exception:
            return False

    if t == "starts_with":
        return s.startswith(str(v))

    if t == "ends_with":
        return s.rstrip().endswith(str(v))

    if t == "must_include":
        return str(v).lower() in output.lower()

    if t == "must_exclude":
        return str(v).lower() not in output.lower()

    if t == "bullet_items_min":
        count = sum(1 for ln in output.split("\n")
                    if re.match(r"^[-•*]\s", ln.strip()))
        return count >= int(v)

    if t == "numbered_items_min":
        count = sum(1 for ln in output.split("\n")
                    if re.match(r"^\d+[.。)\]]\s", ln.strip()))
        return count >= int(v)

    if t == "char_count_max":
        return len(s) <= int(v)

    if t == "char_count_min":
        return len(s) >= int(v)

    return False


_PREAMBLE = "请严格遵守以下格式要求完成任务（不符合格式要求将被判定不合格）：\n\n"


def _build_prompt(case: ScenarioCase):
    return _PREAMBLE + case.payload["prompt"], None


def _l1_score(case: ScenarioCase, parsed: dict | None, raw_content: str | None = None) -> dict:
    text = raw_content or ""
    instructions = case.payload.get("instructions", [])
    if not instructions:
        return {"compliance_rate": 1.0, "n_instructions": 0, "n_satisfied": 0}
    satisfied = sum(1 for inst in instructions if _check(text, inst))
    return {
        "compliance_rate": satisfied / len(instructions),
        "n_instructions": len(instructions),
        "n_satisfied": satisfied,
    }


def _aggregate(per_case: list[dict]) -> dict:
    n = len(per_case) or 1
    return {
        "compliance_rate": sum(c.get("compliance_rate", 0) for c in per_case) / n,
    }


JUDGE_RUBRIC = """你是评测裁判。评估被测模型是否完整遵守了显式格式指令并给出有用回答。
5=格式完全符合且内容高质量；4=格式基本符合、内容合理；3=格式部分遵守；
2=格式大多未遵守但内容尚可；1=格式完全忽略且内容差。
严格输出 JSON: {"score": <1-5 整数>, "rationale": "一句话理由"}"""

SPEC = ScenarioSpec(
    name="instruction_following",
    cases_path="datasets/scenarios/instruction_following/cases.jsonl",
    build_prompt=_build_prompt,
    l1_score=_l1_score,
    aggregate_l1=_aggregate,
    judge_rubric=JUDGE_RUBRIC,
    requires_vlm=False,
    default_thresholds={"compliance_rate_min": 0.80},
    payload_required_fields=["prompt", "instructions"],
)
