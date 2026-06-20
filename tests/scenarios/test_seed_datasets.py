"""种子数据完整性:三场景各 ≥5 条可加载、provenance 合法、引用的截图存在。"""
from pathlib import Path

from benchmark.scenarios import SCENARIOS
from benchmark.scenarios.base import load_cases

ROOT = Path(__file__).resolve().parents[2]


def test_all_seed_datasets_load_and_validate():
    for name, spec in SCENARIOS.items():
        cases = load_cases(ROOT / spec.cases_path)
        assert cases is not None, f"{name}: cases.jsonl missing"
        assert len(cases) >= 5, f"{name}: need >=5 seed cases"


def test_wechat_images_exist():
    for c in load_cases(ROOT / SCENARIOS["wechat_intent"].cases_path):
        img = ROOT / c.payload["image"]
        assert img.exists(), f"missing screenshot {img}"
        assert img.stat().st_size > 1000          # 真渲染过,不是空文件


def test_seed_cases_l1_self_consistency():
    """golden 自洽:把 golden 自身喂给 l1_score 应得满分(防 schema 笔误)。"""
    s2 = SCENARIOS["case_logic"]
    for c in load_cases(ROOT / s2.cases_path):
        perfect = {"consistency": c.payload["consistency_label"],
                   "findings": c.payload["golden_findings"]}
        assert s2.l1_score(c, perfect)["finding_f1"] == 1.0

    s3 = SCENARIOS["article_knowledge"]
    for c in load_cases(ROOT / s3.cases_path):
        perfect = {"claims": c.payload["golden_claims"],
                   "grade": c.payload["knowledge_grade"]}
        s = s3.l1_score(c, perfect)
        assert s["claim_accuracy"] == 1.0 and s["grade_score"] == 1.0
