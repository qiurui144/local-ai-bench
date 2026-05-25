"""MMLU 数据集加载器。

数据格式：每条样本包含 question、choices（A/B/C/D）、answer（正确选项字母）和 subject。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

MMLU_BUILTIN_SAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Paris", "London", "Madrid"],
        "answer": "B",
        "subject": "geography",
    },
    {
        "question": "Which element has atomic number 1?",
        "choices": ["Helium", "Oxygen", "Hydrogen", "Carbon"],
        "answer": "C",
        "subject": "chemistry",
    },
    {
        "question": "What is 2^10?",
        "choices": ["512", "1024", "2048", "256"],
        "answer": "B",
        "subject": "mathematics",
    },
    {
        "question": "Who wrote 'Pride and Prejudice'?",
        "choices": ["Charlotte Brontë", "Jane Austen", "George Eliot", "Virginia Woolf"],
        "answer": "B",
        "subject": "literature",
    },
    {
        "question": "What is the speed of light in a vacuum (approximately)?",
        "choices": ["3×10^6 m/s", "3×10^8 m/s", "3×10^10 m/s", "3×10^4 m/s"],
        "answer": "B",
        "subject": "physics",
    },
]


class MMLUDataset(AbstractDataset):
    """MMLU 多选题数据集。

    使用 `cais/mmlu` HuggingFace 数据集，支持按学科筛选。
    subject=None 时加载 'all' 拆分（全 57 学科混合）。
    """

    def __init__(
        self,
        subject: str = "all",
        split: str = "test",
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(split=split, num_samples=num_samples, **kwargs)
        self.subject = subject

    def _load_hf(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", self.subject, split=self.split, trust_remote_code=True)
        samples = []
        for row in ds:
            choices = row["choices"]
            answer_idx = row["answer"]  # 0-3 整数
            answer_letter = "ABCD"[answer_idx]
            samples.append(
                {
                    "question": row["question"],
                    "choices": choices,
                    "answer": answer_letter,
                    "subject": row.get("subject", self.subject),
                }
            )
        return samples

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return list(MMLU_BUILTIN_SAMPLES)
