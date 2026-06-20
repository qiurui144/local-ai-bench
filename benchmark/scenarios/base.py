"""真实场景维度的共享数据模型。

provenance 是强制字段(synthetic | curated | dataset):合成/策展/公开数据集
三轨必须显式声明,报告按 provenance 分布展示 — 合成数据不得冒充真实数据。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ALLOWED_PROVENANCE = ("synthetic", "curated", "dataset")


@dataclass
class ScenarioCase:
    """一条评测 case。payload 为场景自定义字段(截图路径/文本段落/golden)。"""

    id: str
    provenance: str
    payload: dict
    difficulty: str = "normal"


@dataclass
class ScenarioSpec:
    """一个场景 = 数据集路径 + prompt 构造 + L1 评分器 + judge rubric。

    - build_prompt(case) -> (prompt_text, image_path | None)
    - l1_score(case, parsed_json | None) -> dict
    - aggregate_l1(per_case: list[dict]) -> dict
    - judge_rubric: L2 judge 的 system rubric 文本
    - requires_vlm: 仅 VLM 模型可跑的场景置 True
    """

    name: str
    cases_path: str                     # 相对仓根
    build_prompt: Callable
    l1_score: Callable
    aggregate_l1: Callable
    judge_rubric: str
    requires_vlm: bool = False
    default_thresholds: dict = field(default_factory=dict)
    payload_required_fields: list = field(default_factory=list)  # for verify_benchmark.py


def load_cases(path: Path, num_samples: Optional[int] = None) -> Optional[list[ScenarioCase]]:
    """读 cases.jsonl;文件缺失返回 None(调用方据此 BLOCKED,绝不静默空跑)。"""
    path = Path(path)
    if not path.exists():
        return None
    cases: list[ScenarioCase] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prov = obj.get("provenance")
            if prov not in ALLOWED_PROVENANCE:
                raise ValueError(
                    f"case {obj.get('id')!r}: provenance must be one of "
                    f"{ALLOWED_PROVENANCE}, got {prov!r}"
                )
            cases.append(ScenarioCase(
                id=str(obj["id"]),
                provenance=prov,
                payload=obj.get("payload", {}),
                difficulty=obj.get("difficulty", "normal"),
            ))
    if num_samples is not None:
        cases = cases[:num_samples]
    return cases
