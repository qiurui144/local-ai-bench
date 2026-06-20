"""策展轨 ingest:校验 + 追加一条 case 到对应场景 cases.jsonl。

用法: python scripts/curate_scenario_case.py <scenario> <case.json>
case.json 是单条 case(与 cases.jsonl 行同 schema);本脚本强制:
- provenance ∈ {curated, dataset}(synthetic 走渲染器,不走本入口)
- 必填 payload 字段齐全(按场景 schema)
- article_knowledge 必须含 source_url + collected_at(采集日期,YYYY-MM-DD)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.scenarios import SCENARIOS  # noqa: E402

REQUIRED = {
    "wechat_intent": {"image", "expected_intent", "expected_entities"},
    "case_logic": {"segments", "golden_findings", "consistency_label"},
    "article_knowledge": {"text", "source_url", "collected_at",
                          "golden_claims", "knowledge_grade"},
}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in REQUIRED:
        print(f"usage: curate_scenario_case.py <{'|'.join(REQUIRED)}> <case.json>")
        return 2
    scenario, case_file = sys.argv[1], Path(sys.argv[2])
    obj = json.loads(case_file.read_text(encoding="utf-8"))
    if obj.get("provenance") not in ("curated", "dataset"):
        print("provenance must be curated|dataset for this entry point")
        return 2
    missing = REQUIRED[scenario] - set(obj.get("payload", {}))
    if missing:
        print(f"payload missing fields: {sorted(missing)}")
        return 2
    if scenario not in SCENARIOS:
        print(f"scenario {scenario} not registered")
        return 2
    target = Path(__file__).resolve().parents[1] / SCENARIOS[scenario].cases_path
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"appended {obj['id']} -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
