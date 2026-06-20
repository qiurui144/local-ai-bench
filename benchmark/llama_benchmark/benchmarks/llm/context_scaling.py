"""Context Length 扩展曲线测试：检测非线性 TTFT 增长（KV cache / attention 复杂度）。

覆盖 L7 层分析：评估 context length 对 TTFT 和 decode TPS 的影响。
超线性增长（O(n²)）表明缺少 Flash Attention 或 KV cache 溢出到系统内存。
"""

from __future__ import annotations

import time
from typing import Dict, List

from benchmark.llama_benchmark.core.result import BenchmarkStatus, MetricResult, TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


def run_context_scaling(
    backend,
    context_lengths: List[int],
    model_name: str,
    repeats_per_length: int = 3,
    output_tokens: int = 64,
) -> TaskResult:
    """
    测试不同 context length 下的 TTFT 和 decode TPS。

    检测"膝点"：相邻两档 TTFT/context_length 比值超过 2× 视为非线性增长。
    """
    start_time = time.time()
    scaling_data: Dict[int, Dict[str, float]] = {}
    all_metrics: List[MetricResult] = []

    for ctx_len in context_lengths:
        prompt = _make_context_prompt(ctx_len)
        ttft_samples: List[float] = []
        decode_tps_samples: List[float] = []

        for rep in range(repeats_per_length):
            try:
                ttft_ms, _ = backend.measure_ttft(prompt, max_tokens=output_tokens)
                ttft_samples.append(ttft_ms)

                last_stats = getattr(backend, "_last_ttft_stats", None)
                if last_stats:
                    ec = last_stats.get("eval_count", 0)
                    ed_ns = last_stats.get("eval_duration_ns", 0)
                    if ec > 0 and ed_ns > 0:
                        decode_tps_samples.append(ec / (ed_ns / 1e9))
            except Exception as e:
                logger.warning(f"[{model_name}] context_scaling ctx={ctx_len} rep={rep}: {e}")

        if not ttft_samples:
            continue

        avg_ttft = sum(ttft_samples) / len(ttft_samples)
        avg_decode_tps = (
            sum(decode_tps_samples) / len(decode_tps_samples) if decode_tps_samples else 0.0
        )

        scaling_data[ctx_len] = {
            "ttft_ms": round(avg_ttft, 2),
            "decode_tps": round(avg_decode_tps, 2),
        }

        all_metrics.extend([
            MetricResult(
                name=f"context_{ctx_len}_ttft_ms",
                value=round(avg_ttft, 2),
                unit="ms",
                higher_is_better=False,
                status=BenchmarkStatus.PASS,
            ),
            MetricResult(
                name=f"context_{ctx_len}_decode_tps",
                value=round(avg_decode_tps, 2),
                unit="tokens/s",
                higher_is_better=True,
                status=BenchmarkStatus.PASS,
            ),
        ])

    scaling_nonlinear, nonlinear_at = _detect_nonlinear_scaling(scaling_data)

    return TaskResult(
        task_name="context_scaling",
        model_name=model_name,
        metrics=all_metrics,
        num_samples=len(context_lengths) * repeats_per_length,
        duration_seconds=time.time() - start_time,
        status=BenchmarkStatus.PASS,
        metadata={
            "scaling_data": scaling_data,
            "scaling_nonlinear": scaling_nonlinear,
            "nonlinear_at_context": nonlinear_at,
            "context_lengths_tested": list(scaling_data.keys()),
        },
    )


def _make_context_prompt(target_tokens: int) -> str:
    """生成近似目标 token 数的 prompt（每轮约 12 token）。"""
    unit = "The AI system analyzes language patterns and generates contextual responses effectively "
    prompt = ""
    while len(prompt.split()) < target_tokens // 1.3:
        prompt += unit
    return prompt.strip()


def _detect_nonlinear_scaling(
    scaling_data: Dict[int, Dict[str, float]],
    threshold_ratio: float = 2.0,
) -> tuple:
    """
    检测 TTFT 随 context length 是否出现超线性放大（膝点检测）。

    算法：计算归一化 TTFT（TTFT / ctx_len），若相邻两档的比值 >threshold_ratio，
    则认为出现超线性增长（O(n²) attention 信号）。
    """
    ctx_lengths = sorted(scaling_data.keys())
    if len(ctx_lengths) < 2:
        return False, None

    normalized = [
        scaling_data[ctx]["ttft_ms"] / ctx for ctx in ctx_lengths
    ]

    for i in range(1, len(normalized)):
        if normalized[i - 1] > 0:
            ratio = normalized[i] / normalized[i - 1]
            if ratio > threshold_ratio:
                return True, ctx_lengths[i]

    return False, None
