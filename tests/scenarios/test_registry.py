"""注册表完整性:八个场景全部注册且与各模块 SPEC 同一对象。"""
from benchmark.scenarios import SCENARIOS
from benchmark.scenarios.adversarial_stability import SPEC as S8
from benchmark.scenarios.article_knowledge import SPEC as S3
from benchmark.scenarios.case_logic import SPEC as S2
from benchmark.scenarios.function_calling import SPEC as S6
from benchmark.scenarios.instruction_following import SPEC as S4
from benchmark.scenarios.structured_extraction import SPEC as S5
from benchmark.scenarios.vlm_document_extraction import SPEC as S7
from benchmark.scenarios.wechat_intent import SPEC as S1


def test_all_eight_registered():
    """Original S1-S8 identity checks — these scenarios are committed to the project."""
    assert SCENARIOS["wechat_intent"] is S1
    assert SCENARIOS["case_logic"] is S2
    assert SCENARIOS["article_knowledge"] is S3
    assert SCENARIOS["instruction_following"] is S4
    assert SCENARIOS["structured_extraction"] is S5
    assert SCENARIOS["function_calling"] is S6
    assert SCENARIOS["vlm_document_extraction"] is S7
    assert SCENARIOS["adversarial_stability"] is S8


def test_all_scenarios_have_required_attributes():
    """Verify every registered scenario has required ScenarioSpec attributes.

    This test auto-scales — no need to update a count when adding new scenarios.
    Adding S9+ only requires updating the SCENARIOS registry and this test
    will automatically validate the new entry.
    """
    assert len(SCENARIOS) >= 8, f"Expected ≥8 scenarios, got {len(SCENARIOS)}"
    for name, spec in SCENARIOS.items():
        assert spec.name == name, f"{name}: spec.name mismatch"
        assert spec.cases_path, f"{name}: cases_path is empty"
        assert callable(spec.build_prompt), f"{name}: build_prompt not callable"
        assert callable(spec.l1_score), f"{name}: l1_score not callable"
        assert callable(spec.aggregate_l1), f"{name}: aggregate_l1 not callable"
        assert isinstance(spec.judge_rubric, str) and spec.judge_rubric, f"{name}: judge_rubric empty"
        assert isinstance(spec.payload_required_fields, list), \
            f"{name}: payload_required_fields not a list"
