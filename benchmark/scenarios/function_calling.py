"""S6: 函数调用 (function_calling) — 给定工具定义 + 对话，生成正确的 tool call JSON。

AI Agent / 企业 AI 集成核心能力。测试模型能否从自然语言推断正确函数名称和参数。
与 Berkeley Function-Calling Leaderboard（BFCL）同类，聚焦中文业务场景。

case payload:
  tools    : [{name, description, parameters: {field: {type, description, required}}}]
  messages : [{role, content}]              当前对话历史（末行通常是 user 请求）
  expected : {name: str, arguments: {}}     标准调用

模型输出格式:
  {"name": "<function_name>", "arguments": {"key": "value", ...}}

L1:
  name_match      : 1/0 — 函数名完全匹配
  arg_recall      : TP / |expected_args|  必填参数覆盖率（值大小写归一化）
  arg_precision   : TP / |output_args|    输出参数准确率
  arg_f1          : 调和均值

聚合 = mean over all cases
"""
from __future__ import annotations

from .base import ScenarioCase, ScenarioSpec


def _fmt_tools(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        lines.append(f"函数名: {t['name']}")
        lines.append(f"描述: {t.get('description', '')}")
        params = t.get("parameters", {})
        if params:
            lines.append("参数:")
            for pname, pmeta in params.items():
                req = "（必填）" if pmeta.get("required") else "（可选）"
                lines.append(
                    f"  - {pname} ({pmeta.get('type', 'string')}){req}: "
                    f"{pmeta.get('description', '')}"
                )
        lines.append("")
    return "\n".join(lines).strip()


def _fmt_messages(messages: list[dict]) -> str:
    role_map = {"user": "用户", "assistant": "助手", "system": "系统"}
    return "\n".join(
        f"{role_map.get(m['role'], m['role'])}: {m['content']}"
        for m in messages
    )


_PROMPT_TMPL = """你是一个 AI 助手，可以调用以下工具函数。
根据最新用户请求，判断应调用哪个函数及使用哪些参数。
只输出一个 JSON 对象，格式为 {{"name": "函数名", "arguments": {{参数键值对}}}}，不输出任何其他内容。
如不需要调用函数，输出 {{"name": null, "arguments": {{}}}}.

可用函数:
{tools}

对话历史:
{messages}

请输出调用的函数 JSON:"""


def _build_prompt(case: ScenarioCase):
    tools_text = _fmt_tools(case.payload["tools"])
    messages_text = _fmt_messages(case.payload["messages"])
    return _PROMPT_TMPL.format(tools=tools_text, messages=messages_text), None


def _norm_val(v) -> str:
    return str(v).strip().lower().replace(" ", "").replace("，", ",")


def _l1_score(case: ScenarioCase, parsed: dict | None, raw_content: str | None = None) -> dict:
    expected = case.payload["expected"]
    exp_name = expected.get("name")
    exp_args = expected.get("arguments", {})

    if not isinstance(parsed, dict):
        return {"name_match": 0, "arg_recall": 0.0, "arg_precision": 0.0, "arg_f1": 0.0}

    got_name = parsed.get("name")
    got_args = parsed.get("arguments") or {}

    name_match = int(got_name == exp_name)

    if not exp_args:
        arg_recall = arg_precision = arg_f1 = 1.0
    else:
        tp = sum(
            1 for k, v in exp_args.items()
            if k in got_args and _norm_val(got_args[k]) == _norm_val(v)
        )
        arg_recall = tp / len(exp_args)
        arg_precision = tp / len(got_args) if got_args else 0.0
        arg_f1 = (
            2 * arg_precision * arg_recall / (arg_precision + arg_recall)
            if (arg_precision + arg_recall) else 0.0
        )

    return {
        "name_match": name_match,
        "arg_recall": arg_recall,
        "arg_precision": arg_precision,
        "arg_f1": arg_f1,
    }


def _aggregate(per_case: list[dict]) -> dict:
    n = len(per_case) or 1
    return {
        "name_accuracy": sum(c.get("name_match", 0) for c in per_case) / n,
        "arg_recall": sum(c.get("arg_recall", 0) for c in per_case) / n,
        "arg_f1": sum(c.get("arg_f1", 0) for c in per_case) / n,
    }


JUDGE_RUBRIC = """你是评测裁判。评估被测模型的函数调用是否正确。
5=函数名和所有参数完全正确；4=函数名正确、参数有小偏差；3=函数名正确但参数明显有误；
2=函数名错误但大意理解正确；1=完全错误或格式不符。
严格输出 JSON: {"score": <1-5 整数>, "rationale": "一句话理由"}"""

SPEC = ScenarioSpec(
    name="function_calling",
    cases_path="datasets/scenarios/function_calling/cases.jsonl",
    build_prompt=_build_prompt,
    l1_score=_l1_score,
    aggregate_l1=_aggregate,
    judge_rubric=JUDGE_RUBRIC,
    requires_vlm=False,
    default_thresholds={"name_accuracy_min": 0.85, "arg_f1_min": 0.75},
    payload_required_fields=["tools", "messages", "expected"],
)
