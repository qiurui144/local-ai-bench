"""NLP 指标：WER / CER。"""

from __future__ import annotations

import re
from typing import List

from benchmark.llama_benchmark.core.base_metric import AbstractMetric


def _normalize_text(text: str) -> str:
    """标准化文本：转小写、去标点、合并空白。"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class WERMetric(AbstractMetric):
    """Word Error Rate（词错误率）。

    WER = (S + D + I) / N
    其中 S=替换, D=删除, I=插入, N=参考词数。
    """

    name = "wer"
    unit = ""
    higher_is_better = False

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs,
    ) -> float:
        try:
            from jiwer import wer, transforms
        except ImportError:
            raise ImportError("请安装 jiwer: pip install jiwer")

        if self.normalize:
            predictions = [_normalize_text(p) for p in predictions]
            references = [_normalize_text(r) for r in references]

        return float(wer(references, predictions))


class CERMetric(AbstractMetric):
    """Character Error Rate（字符错误率）。

    适用于中文、日文等无词边界语言。
    """

    name = "cer"
    unit = ""
    higher_is_better = False

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs,
    ) -> float:
        try:
            from jiwer import cer
        except ImportError:
            raise ImportError("请安装 jiwer: pip install jiwer")

        if self.normalize:
            predictions = [_normalize_text(p) for p in predictions]
            references = [_normalize_text(r) for r in references]

        return float(cer(references, predictions))
