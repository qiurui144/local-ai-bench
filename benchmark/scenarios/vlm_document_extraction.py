"""S7: VLM 图片文档抽取 — 从业务单据图片中提取结构化字段。

业界典型多模态落地场景：VLM 直接处理银行流水、增值税发票、收据、汇款凭证图片，
输出结构化 JSON 字段。区别于 S5（文本已 OCR），S7 考察 VLM 的视觉识别+抽取一体化能力。

case payload:
  document_type : str   "bank_statement" | "vat_invoice" | "receipt" | "bank_transfer"
  image_path    : str   相对仓根路径，传给 infer_sync image_path 参数
  fields        : [str] 需要抽取的字段名列表
  golden        : dict  标准答案（null 表示字段不存在于此单据）

L1 = field_accuracy（归一化精确匹配）；复用 _extraction_common.field_accuracy_score。
"""
from __future__ import annotations

from ._extraction_common import field_accuracy_score  # noqa: F401
from .base import ScenarioCase, ScenarioSpec

_DOC_TYPE_ZH = {
    "bank_statement": "银行流水单",
    "vat_invoice": "增值税发票",
    "receipt": "收据",
    "bank_transfer": "银行汇款凭证",
}

_PROMPT_TMPL = (
    "你是文档信息抽取助手。请仔细观察图片中的{doc_type_zh}，"
    "提取以下字段，以 JSON 对象格式输出（只输出 JSON，不加任何解释）。\n\n"
    "需要提取的字段（字段不存在时值为 null）:\n{fields}\n\n"
    "输出格式:\n{{{example}}}"
)


def _build_prompt(case: ScenarioCase):
    p = case.payload
    doc_type_zh = _DOC_TYPE_ZH.get(p["document_type"], p["document_type"])
    fields_text = "\n".join(f"- {f}" for f in p["fields"])
    sample = p["fields"][:3]
    example = ", ".join(f'"{f}": "..."' for f in sample)
    if len(p["fields"]) > 3:
        example += ", ..."
    prompt = _PROMPT_TMPL.format(
        doc_type_zh=doc_type_zh, fields=fields_text, example=example
    )
    return prompt, p["image_path"]


def _l1_score(case: ScenarioCase, parsed: dict | None, raw_content: str | None = None) -> dict:
    return field_accuracy_score(case.payload, parsed)


def _aggregate(scores: list[dict]) -> dict:
    scorable = [s for s in scores if s.get("n_fields", 0) > 0]
    if not scorable:
        return {"field_accuracy": 0.0}
    return {"field_accuracy": sum(s["field_accuracy"] for s in scorable) / len(scorable)}


JUDGE_RUBRIC = (
    "你是评测裁判。评估被测模型从单据图片中提取字段的质量。\n"
    "5=所有字段准确提取；4=大部分字段正确、小瑕疵；3=约一半字段正确；\n"
    "2=少数字段正确；1=基本全错或格式完全不符。\n"
    '只输出 JSON: {"score": <1-5>, "rationale": "<一句话>"}'
)

SPEC = ScenarioSpec(
    name="vlm_document_extraction",
    cases_path="datasets/scenarios/vlm_document_extraction/cases.jsonl",
    build_prompt=_build_prompt,
    l1_score=_l1_score,
    aggregate_l1=_aggregate,
    judge_rubric=JUDGE_RUBRIC,
    requires_vlm=True,
    default_thresholds={"field_accuracy_min": 0.75},
    payload_required_fields=["document_type", "image_path", "fields", "golden"],
)
