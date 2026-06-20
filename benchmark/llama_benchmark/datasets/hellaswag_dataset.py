"""HellaSwag 数据集加载器。

数据格式：每条样本包含 activity_label、ctx（上下文）、endings（4 个候选句子结尾）和 label（正确结尾索引 0-3）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

# 供应链固定(2026-06-10 实测,datasets==4.5.0):Rowan/hellaswag 非 gated 纯
# parquet,无 trust_remote_code 可直接加载;revision pin 到 main commit SHA。
HELLASWAG_DEFAULT_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"


def hellaswag_revision() -> str:
    # `or` 而非 get(k, default):env 置空串时也回落默认值
    return os.environ.get("HELLASWAG_REVISION") or HELLASWAG_DEFAULT_REVISION

HELLASWAG_BUILTIN_SAMPLES = [
    {
        "activity_label": "Cooking",
        "ctx": "She poured the batter into the pan and placed it in the oven.",
        "endings": [
            "The cake slowly rose and turned golden brown.",
            "She then went outside to mow the lawn.",
            "The car engine started with a roar.",
            "He signed the important document.",
        ],
        "label": 0,
    },
    {
        "activity_label": "Sports",
        "ctx": "The basketball player dribbled down the court and",
        "endings": [
            "jumped to make a slam dunk.",
            "decided to plant a vegetable garden.",
            "started knitting a warm sweater.",
            "read the morning newspaper.",
        ],
        "label": 0,
    },
    {
        "activity_label": "Gardening",
        "ctx": "She carefully planted the seeds in the soil and watered them.",
        "endings": [
            "Over the next few weeks, tiny green sprouts appeared.",
            "The airplane landed safely at the airport.",
            "He submitted his tax return forms.",
            "The stock market closed higher.",
        ],
        "label": 0,
    },
    {
        "activity_label": "Technology",
        "ctx": "He noticed the laptop battery was critically low so he",
        "endings": [
            "plugged in the charging cable immediately.",
            "went for a long swim in the ocean.",
            "baked a chocolate cake for dessert.",
            "painted watercolors all afternoon.",
        ],
        "label": 0,
    },
    {
        "activity_label": "Travel",
        "ctx": "After months of planning, she finally arrived at the airport with her luggage and",
        "endings": [
            "checked in for her long-awaited vacation flight.",
            "decided to stay home and watch television.",
            "started a new software project.",
            "repaired the broken fence in the backyard.",
        ],
        "label": 0,
    },
]


class HellaSwagDataset(AbstractDataset):
    """HellaSwag 常识推理数据集。

    使用 `Rowan/hellaswag` HuggingFace 数据集。
    评测方式：length-normalized logprob，选 logprob/长度 最大的候选结尾。
    """

    def __init__(
        self,
        split: str = "validation",
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(split=split, num_samples=num_samples, **kwargs)

    def _load_hf(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(
            "Rowan/hellaswag", split=self.split, revision=hellaswag_revision()
        )
        samples = []
        for row in ds:
            samples.append(
                {
                    "activity_label": row["activity_label"],
                    "ctx": row["ctx"],
                    "endings": row["endings"],
                    "label": int(row["label"]),
                }
            )
        return samples

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return list(HELLASWAG_BUILTIN_SAMPLES)
