from benchmark.scenarios.base import ScenarioCase
from benchmark.scenarios.wechat_intent import SPEC


def _case(intent="loan_agreement", entities=("张三", "5000元")):
    return ScenarioCase(id="c1", provenance="synthetic", payload={
        "image": "fixtures/scenarios/wechat_intent/c1.png",
        "expected_intent": intent,
        "expected_entities": list(entities),
    })


def test_spec_shape():
    assert SPEC.name == "wechat_intent"
    assert SPEC.requires_vlm
    assert SPEC.cases_path == "datasets/scenarios/wechat_intent/cases.jsonl"


def test_build_prompt_returns_image():
    prompt, image = SPEC.build_prompt(_case())
    assert "意图" in prompt and "JSON" in prompt
    assert str(image).endswith("c1.png")


def test_l1_score_exact_hit():
    parsed = {"intent": "loan_agreement", "entities": ["张三", "5000元", "明天"]}
    s = SPEC.l1_score(_case(), parsed)
    assert s["intent_hit"] == 1
    assert s["entity_recall"] == 1.0


def test_l1_score_miss_and_partial():
    parsed = {"intent": "denial", "entities": ["张三"]}
    s = SPEC.l1_score(_case(), parsed)
    assert s["intent_hit"] == 0
    assert s["entity_recall"] == 0.5


def test_l1_score_unparseable_output():
    s = SPEC.l1_score(_case(), None)
    assert s["intent_hit"] == 0 and s["entity_recall"] == 0.0


def test_aggregate():
    agg = SPEC.aggregate_l1([
        {"intent_hit": 1, "entity_recall": 1.0},
        {"intent_hit": 0, "entity_recall": 0.5},
    ])
    assert agg["intent_accuracy"] == 0.5
    assert agg["entity_recall"] == 0.75
