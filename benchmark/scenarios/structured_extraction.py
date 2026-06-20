"""S5: 结构化抽取 (structured_extraction) — 从业务文档文本中提取关键字段为 JSON。

企业 AI 最高频落地场景之一：OCR 后处理、合同解析、票据录入、招聘信息结构化。
L1 按字段精确匹配率评分，归一化处理金额/日期格式差异。

case payload:
  document_type : str              文档类型（invoice/contract/meeting/job_posting/receipt）
  text          : str              原始文档文本
  fields        : [str]            需要抽取的字段名称列表
  golden        : {field: value}   标准答案（null 表示该字段在文档中不存在）

L1 = field_accuracy（正确字段数 / 非 null 标准字段数）
聚合 = mean(field_accuracy) over all cases
"""
from __future__ import annotations

from ._extraction_common import _normalize, field_accuracy_score  # noqa: F401  (_normalize re-exported for tests)
from .base import ScenarioCase, ScenarioSpec


_PROMPT_TMPL = """你是文档信息抽取助手。从以下{doc_type}文本中提取指定字段，以 JSON 对象格式输出（只输出 JSON，不加任何解释）。

需要提取的字段（按原文提取，字段不存在时值为 null）:
{fields_list}

文档原文:
---
{text}
---

输出格式（字段名与上方完全一致）:
{example}"""


def _build_prompt(case: ScenarioCase):
    doc_type = case.payload.get("document_type", "文档")
    fields = case.payload["fields"]
    text = case.payload["text"]
    fields_list = "\n".join(f"- {f}" for f in fields)
    sample = fields[:4]
    example = "{" + ", ".join(f'"{f}": "..."' for f in sample)
    if len(fields) > 4:
        example += ", ..."
    example += "}"
    prompt = _PROMPT_TMPL.format(
        doc_type=doc_type, fields_list=fields_list, text=text, example=example
    )
    return prompt, None


def _l1_score(case: ScenarioCase, parsed: dict | None, raw_content: str | None = None) -> dict:
    return field_accuracy_score(case.payload, parsed)


def _aggregate(per_case: list[dict]) -> dict:
    n = len(per_case) or 1
    return {
        "field_accuracy": sum(c.get("field_accuracy", 0) for c in per_case) / n,
    }


JUDGE_RUBRIC = """你是评测裁判。评估被测模型从文档中提取字段的质量。
5=所有字段准确提取；4=大部分字段正确、小瑕疵；3=约一半字段正确；
2=少数字段正确；1=基本全错或格式完全不符。
严格输出 JSON: {"score": <1-5 整数>, "rationale": "一句话理由"}"""

SPEC = ScenarioSpec(
    name="structured_extraction",
    cases_path="datasets/scenarios/structured_extraction/cases.jsonl",
    build_prompt=_build_prompt,
    l1_score=_l1_score,
    aggregate_l1=_aggregate,
    judge_rubric=JUDGE_RUBRIC,
    requires_vlm=False,
    default_thresholds={"field_accuracy_min": 0.75},
    payload_required_fields=["document_type", "text", "fields", "golden"],
)
