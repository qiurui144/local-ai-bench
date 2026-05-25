"""性能指标：延迟分位数 / 吞吐量计算。"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """计算延迟分布统计（P50 / P95 / P99 / mean / min / max）。

    Args:
        latencies_ms: 延迟样本列表（毫秒）

    Returns:
        包含各分位数的字典，单位 ms
    """
    if not latencies_ms:
        return {}
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(np.std(arr)),
    }


def compute_throughput(
    total_tokens: int,
    total_seconds: float,
) -> float:
    """计算 tokens/s 吞吐量。"""
    if total_seconds <= 0:
        return 0.0
    return total_tokens / total_seconds


def compute_tpot(
    total_completion_tokens: int,
    total_generation_ms: float,
) -> float:
    """计算 TPOT（Time Per Output Token，ms/token）。"""
    if total_completion_tokens <= 0:
        return 0.0
    return total_generation_ms / total_completion_tokens
