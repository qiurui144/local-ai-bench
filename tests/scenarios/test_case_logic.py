from benchmark.scenarios.base import ScenarioCase
from benchmark.scenarios.case_logic import SPEC


def _case():
    return ScenarioCase(id="c1", provenance="dataset", payload={
        "segments": ["1月1日借款5万", "1月3日还清", "至今未还分文"],
        "golden_findings": [{"kind": "fact_mismatch", "segments": [1, 2]}],
        "consistency_label": "contradictory",
    })


def test_spec_shape():
    assert SPEC.name == "case_logic"
    assert not SPEC.requires_vlm
    assert SPEC.cases_path == "datasets/scenarios/case_logic/cases.jsonl"


def test_build_prompt_embeds_segments():
    prompt, image = SPEC.build_prompt(_case())
    assert image is None
    assert "[0]" in prompt and "1月1日借款5万" in prompt


def test_l1_perfect():
    parsed = {"consistency": "contradictory",
              "findings": [{"kind": "fact_mismatch", "segments": [2, 1]}]}  # 顺序无关
    s = SPEC.l1_score(_case(), parsed)
    assert s["label_hit"] == 1
    assert s["finding_f1"] == 1.0


def test_l1_partial():
    parsed = {"consistency": "minor_issues",
              "findings": [{"kind": "fact_mismatch", "segments": [1, 2]},
                           {"kind": "time_conflict", "segments": [0, 1]}]}  # 1 真 1 假
    s = SPEC.l1_score(_case(), parsed)
    assert s["label_hit"] == 0
    assert abs(s["finding_f1"] - (2 * 1.0 * 0.5) / 1.5) < 1e-9  # P=0.5 R=1.0


def test_l1_unparseable():
    s = SPEC.l1_score(_case(), None)
    assert s == {"label_hit": 0, "finding_f1": 0.0}


def test_aggregate():
    agg = SPEC.aggregate_l1([{"label_hit": 1, "finding_f1": 1.0},
                             {"label_hit": 0, "finding_f1": 0.5}])
    assert agg["label_accuracy"] == 0.5
    assert agg["finding_f1"] == 0.75
