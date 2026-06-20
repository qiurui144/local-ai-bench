"""GSM8K 数据集加载器。

数据格式：每条样本包含 question（小学数学应用题）和 answer（含 CoT 推理步骤，`####` 后为最终数字答案）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

# 供应链固定(2026-06-10 实测,datasets==4.5.0):openai/gsm8k 非 gated 纯 parquet,
# 无 trust_remote_code 可直接加载;revision pin 到 main commit SHA 保数据完整性。
GSM8K_DEFAULT_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def gsm8k_revision() -> str:
    # `or` 而非 get(k, default):env 置空串时也回落默认值
    return os.environ.get("GSM8K_REVISION") or GSM8K_DEFAULT_REVISION

GSM8K_BUILTIN_SAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins for friends with 4. How many eggs does she sell at the farmers' market per day if she sells them for $2 each?",
        "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber.\nSo the total amount of fiber is 2+1=<<2+1=3>>3 bolts of fiber.\n#### 3",
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000.\nHe increased the value of the house by 80,000*1.5=$<<80000*1.5=120000>>120,000.\nSo the new value of the house is 80,000+120,000=$<<80000+120000=200000>>200,000.\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000.\n#### 70000",
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week.\nSo he writes 6*2=<<6*2=12>>12 pages every week.\nThat means he writes 12*52=<<12*52=624>>624 pages a year.\n#### 624",
    },
    {
        "question": "Mark has a garden with 3 rows of 4 plants each. He waters 1/3 of his plants. How many plants does he water?",
        "answer": "He has 3*4=<<3*4=12>>12 plants.\nHe waters 12*(1/3)=<<12*(1/3)=4>>4 plants.\n#### 4",
    },
]


class GSM8KDataset(AbstractDataset):
    """GSM8K 小学数学推理数据集。

    使用 `openai/gsm8k` HuggingFace 数据集（main 配置）。
    answer 字段含完整 CoT 推理，`####` 后为最终数字答案。
    """

    def __init__(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(split=split, num_samples=num_samples, **kwargs)

    def _load_hf(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(
            "openai/gsm8k", "main", split=self.split, revision=gsm8k_revision()
        )
        return [{"question": row["question"], "answer": row["answer"]} for row in ds]

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return list(GSM8K_BUILTIN_SAMPLES)
