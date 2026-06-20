from benchmark.scenarios.base import ScenarioCase
from benchmark.scenarios.article_knowledge import SPEC, _grade_score


def _case():
    return ScenarioCase(id="c1", provenance="curated", payload={
        "text": "维生素C可以治愈感冒,这是常识。",
        "source_url": "https://example.com/a",
        "golden_claims": [
            {"claim": "维生素C可以治愈感冒", "label": "inaccurate"},
        ],
        "knowledge_grade": "D",
    })


def test_spec_shape():
    assert SPEC.name == "article_knowledge"
    assert not SPEC.requires_vlm
    assert SPEC.cases_path == "datasets/scenarios/article_knowledge/cases.jsonl"


def test_build_prompt_embeds_claims():
    prompt, image = SPEC.build_prompt(_case())
    assert image is None
    assert "维生素C可以治愈感冒" in prompt and "JSON" in prompt


def test_grade_score_exact_adjacent_far():
    assert _grade_score("B", "B") == 1.0
    assert _grade_score("B", "C") == 0.5
    assert _grade_score("A", "D") == 0.0
    assert _grade_score(None, "B") == 0.0


def test_l1_score():
    parsed = {"claims": [{"claim": "维生素C可以治愈感冒", "label": "inaccurate"}],
              "grade": "C"}
    s = SPEC.l1_score(_case(), parsed)
    assert s["claim_accuracy"] == 1.0
    assert s["grade_score"] == 0.5


def test_l1_claim_order_insensitive_match_by_text():
    parsed = {"claims": [{"claim": "无关", "label": "accurate"},
                         {"claim": "维生素C可以治愈感冒", "label": "accurate"}],
              "grade": "D"}
    s = SPEC.l1_score(_case(), parsed)
    assert s["claim_accuracy"] == 0.0      # 标签判错
    assert s["grade_score"] == 1.0


def test_aggregate():
    agg = SPEC.aggregate_l1([{"claim_accuracy": 1.0, "grade_score": 0.5},
                             {"claim_accuracy": 0.0, "grade_score": 1.0}])
    assert agg["claim_accuracy"] == 0.5
    assert agg["grade_score"] == 0.75
