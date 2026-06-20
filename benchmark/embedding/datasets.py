"""Retrieval dataset loaders for the embedding + rerank dimensions.

A retrieval set is a list of :class:`RetrievalQuery` records, each with a query
string, a list of candidate documents, and the indices of the relevant
candidate(s). Both ``benchmark.embedding`` and ``benchmark.rerank`` iterate
these identically.

Two sources:

- :func:`load_retrieval_jsonl` — a JSONL file, one object per line:
  ``{"query": ..., "candidates": [...], "relevant": [idx, ...]}``. This is the
  path for a real corpus (e.g. a C-MTEB / CMedQA subset exported to JSONL).
- :func:`load_builtin_retrieval` — a small **synthetic / hand-authored**
  Chinese retrieval set used for offline smoke runs and unit tests. It is
  flagged ``source="builtin"`` so it never masquerades as real benchmark data.

Provenance is honest: ``source`` is ``"custom"`` for the shipped JSONL and
``"builtin"`` for the synthetic fallback (mirrors the translation module's
flores/builtin/custom convention).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RetrievalQuery:
    """One retrieval query with its candidate pool and gold relevance."""

    query: str
    candidates: list[str]
    relevant: set[int]                 # indices into ``candidates``
    qid: str = ""
    domain: str = "general"
    source: str = "custom"             # provenance: custom | builtin
    meta: dict = field(default_factory=dict)


def load_retrieval_jsonl(
    path: Path | str,
    *,
    num_samples: Optional[int] = None,
) -> list[RetrievalQuery]:
    """Load a retrieval set from JSONL. Raises if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"retrieval corpus not found: {path}")
    out: list[RetrievalQuery] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            candidates = list(obj["candidates"])
            relevant = set(int(r) for r in obj.get("relevant", []))
            if not candidates or not relevant:
                # Skip malformed rows rather than poisoning metrics with a
                # query that has no scorable gold.
                continue
            out.append(RetrievalQuery(
                query=obj["query"],
                candidates=candidates,
                relevant=relevant,
                qid=str(obj.get("qid", i)),
                domain=obj.get("domain", "general"),
                source="custom",
                meta=obj.get("meta", {}),
            ))
            if num_samples is not None and len(out) >= num_samples:
                break
    return out


# ---------------------------------------------------------------------------
# Built-in synthetic Chinese retrieval set (offline / unit-test fallback).
# Hand-authored — NOT a real benchmark corpus. Kept small on purpose.
# ---------------------------------------------------------------------------
_BUILTIN: list[dict] = [
    {
        "query": "如何重置我的账户密码？",
        "candidates": [
            "在登录页点击「忘记密码」，输入注册邮箱即可收到重置链接。",
            "本店周末营业时间为上午十点到晚上九点。",
            "新用户首单可享受八折优惠。",
            "请在设置页面的安全选项中修改并重置登录密码。",
        ],
        "relevant": [0, 3],
        "domain": "support",
    },
    {
        "query": "向量数据库适合存储什么类型的数据？",
        "candidates": [
            "向量数据库用于存储和检索高维嵌入向量，常用于语义搜索。",
            "关系型数据库以行和列的二维表结构存储结构化数据。",
            "今天的天气晴朗，气温适宜出行。",
            "嵌入向量可以表示文本、图像的语义，存入向量库做近邻检索。",
        ],
        "relevant": [0, 3],
        "domain": "tech",
    },
    {
        "query": "合同违约金和定金有什么区别？",
        "candidates": [
            "违约金是约定的违约赔偿，定金具有担保性质且适用定金罚则。",
            "公司年会将于下月在市中心酒店举办。",
            "定金与违约金性质不同，定金双倍返还，违约金按约定赔偿损失。",
            "苹果是一种常见的水果，富含维生素。",
        ],
        "relevant": [0, 2],
        "domain": "legal",
    },
    {
        "query": "深度学习模型为什么需要 GPU 加速？",
        "candidates": [
            "GPU 拥有大量并行核心，能高效完成矩阵乘法等张量运算，加速训练。",
            "图书馆借书需要出示有效的读者证。",
            "神经网络训练涉及海量并行计算，GPU 比 CPU 更适合。",
            "明天股市预计小幅波动。",
        ],
        "relevant": [0, 2],
        "domain": "tech",
    },
    {
        "query": "怎样申请退款？",
        "candidates": [
            "在订单详情页选择「申请退款」并填写退款原因即可提交。",
            "我们的客服热线工作日全天开放。",
            "退款需在收货后七天内于订单页发起申请。",
            "该商品库存充足，可立即下单。",
        ],
        "relevant": [0, 2],
        "domain": "support",
    },
    {
        "query": "什么是提示词工程？",
        "candidates": [
            "提示词工程是设计和优化大模型输入提示以获得更好输出的方法。",
            "晚餐推荐清淡饮食，有助于睡眠。",
            "通过精心构造 prompt 来引导大语言模型，是提示词工程的核心。",
            "登山时应携带充足的饮用水。",
        ],
        "relevant": [0, 2],
        "domain": "tech",
    },
]


def load_builtin_retrieval(
    *,
    num_samples: Optional[int] = None,
    domain: Optional[str] = None,
) -> list[RetrievalQuery]:
    """Synthetic Chinese retrieval set for offline smoke runs + unit tests."""
    rows = _BUILTIN
    if domain:
        rows = [r for r in rows if r.get("domain") == domain]
    out: list[RetrievalQuery] = []
    for i, r in enumerate(rows):
        out.append(RetrievalQuery(
            query=r["query"],
            candidates=list(r["candidates"]),
            relevant=set(r["relevant"]),
            qid=f"builtin-{i}",
            domain=r.get("domain", "general"),
            source="builtin",
        ))
        if num_samples is not None and len(out) >= num_samples:
            break
    return out


def load_retrieval(
    path: Path | str | None = None,
    *,
    num_samples: Optional[int] = None,
) -> list[RetrievalQuery]:
    """Load ``path`` if it exists, else fall back to the built-in synthetic set."""
    if path is not None and Path(path).exists():
        return load_retrieval_jsonl(path, num_samples=num_samples)
    return load_builtin_retrieval(num_samples=num_samples)
