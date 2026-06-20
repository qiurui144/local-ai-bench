"""Real-scenario benchmark dimension.

S1 wechat_intent            — VLM: 微信截图意图抽取
S2 case_logic               — LLM: 案件陈述逻辑一致性
S3 article_knowledge        — LLM: 自媒体文章知识性评估
S4 instruction_following    — LLM: IFEval 风格指令遵循（程序化 L1，无需 judge）
S5 structured_extraction    — LLM: 业务文档结构化字段抽取
S6 function_calling         — LLM: 工具函数调用（BFCL 同类）
S7 vlm_document_extraction  — VLM: 扫描件/票据结构化字段抽取
S8 adversarial_stability    — LLM: 对抗稳健性（提示注入 / 锚点攻击 / 上下文混淆 / 角色混淆 / 边界输入）
"""
from .adversarial_stability import SPEC as _ADV_STAB
from .article_knowledge import SPEC as _ARTICLE
from .base import ScenarioCase, ScenarioSpec, load_cases  # noqa: F401
from .case_logic import SPEC as _CASE_LOGIC
from .function_calling import SPEC as _FUNC_CALL
from .instruction_following import SPEC as _INSTR_FOLLOW
from .structured_extraction import SPEC as _STRUCT_EXTRACT
from .vlm_document_extraction import SPEC as _VLM_DOC
from .wechat_intent import SPEC as _WECHAT

SCENARIOS: dict[str, ScenarioSpec] = {
    _WECHAT.name: _WECHAT,
    _CASE_LOGIC.name: _CASE_LOGIC,
    _ARTICLE.name: _ARTICLE,
    _INSTR_FOLLOW.name: _INSTR_FOLLOW,
    _STRUCT_EXTRACT.name: _STRUCT_EXTRACT,
    _FUNC_CALL.name: _FUNC_CALL,
    _VLM_DOC.name: _VLM_DOC,
    _ADV_STAB.name: _ADV_STAB,
}
