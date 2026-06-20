"""LLM 性能 Benchmark：TTFT / TPOT / Throughput，MLPerf Inference 标准。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from benchmark.llama_benchmark.core.config import PerformanceConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    TaskResult,
)
from benchmark.llama_benchmark.metrics.performance import (
    compute_latency_stats,
    compute_throughput,
)
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)

# 标准测试 prompt 模板（按不同长度生成）
PROMPT_TEMPLATES = {
    128: "Explain the concept of machine learning in simple terms. " * 4,
    512: "Write a comprehensive analysis of the impact of artificial intelligence on modern society, "
         "covering economic, social, and ethical dimensions. " * 8,
    1024: "Provide a detailed technical explanation of transformer architecture, "
          "including self-attention mechanisms, positional encoding, and multi-head attention. " * 12,
    2048: "Write an extensive research report on climate change, including historical data, "
          "current trends, future projections, and potential mitigation strategies. " * 16,
}


def _make_prompt(target_tokens: int) -> str:
    """生成近似目标 token 数的 prompt。"""
    # 每个模板单元约 10-20 个 token，循环直到达到目标长度
    base = "The following is a detailed discussion: "
    word = "artificial intelligence and machine learning systems are transforming industries "
    prompt = base
    while len(prompt.split()) < target_tokens // 1.3:
        prompt += word
    return prompt


def run_performance(
    backend,
    config: PerformanceConfig,
    model_name: str,
) -> TaskResult:
    """执行性能测试（TTFT / TPOT / Throughput）。

    测试矩阵：prompt_lengths × output_lengths 各测试 num_test_requests 次。
    先执行 num_warmup_requests 次预热，不计入统计。
    """
    start_time = time.time()

    # 预热
    warmup_prompt = _make_prompt(128)
    logger.info(f"[{model_name}] 性能预热 {config.num_warmup_requests} 次...")
    for _ in range(config.num_warmup_requests):
        try:
            backend.generate(warmup_prompt, max_tokens=32, temperature=0.0)
        except Exception as e:
            logger.warning(f"预热失败: {e}")

    all_metrics: List[MetricResult] = []
    results_by_config: Dict[str, Dict] = {}

    for prompt_len in config.prompt_lengths:
        for output_len in config.output_lengths:
            key = f"p{prompt_len}_o{output_len}"
            prompt = _make_prompt(prompt_len)

            ttft_list: List[float] = []
            total_latency_list: List[float] = []
            ollama_stats_list: List[Dict[str, Any]] = []

            for _ in tqdm(
                range(config.num_test_requests),
                desc=f"  性能 prompt={prompt_len} output={output_len}",
                leave=False,
            ):
                try:
                    ttft_ms, total_ms = backend.measure_ttft(prompt, max_tokens=output_len)
                    ttft_list.append(ttft_ms)
                    total_latency_list.append(total_ms)
                    # 提取服务端元数据（OllamaBackend 支持，其他后端跳过）
                    last_stats = getattr(backend, "_last_ttft_stats", None)
                    if last_stats:
                        ollama_stats_list.append(last_stats)
                except Exception as e:
                    logger.warning(f"性能测试失败: {e}")

            if not ttft_list:
                continue

            ttft_stats = compute_latency_stats(ttft_list)
            total_stats = compute_latency_stats(total_latency_list)
            generation_ms = [t - f for t, f in zip(total_latency_list, ttft_list)]
            avg_tpot = (
                sum(generation_ms) / (len(generation_ms) * output_len)
                if generation_ms else 0.0
            )
            throughput = compute_throughput(
                len(ttft_list) * output_len,
                sum(total_latency_list) / 1000,
            )

            # 计算时序分解（仅 Ollama 后端有服务端元数据）
            timing_breakdown: Optional[Dict[str, float]] = None
            if ollama_stats_list:
                avg_stats = {
                    k: sum(s[k] for s in ollama_stats_list) / len(ollama_stats_list)
                    for k in ollama_stats_list[0]
                }
                prompt_eval_ms = avg_stats["prompt_eval_duration_ns"] / 1e6
                token_gen_ms = avg_stats["eval_duration_ns"] / 1e6
                model_load_ms = avg_stats["load_duration_ns"] / 1e6
                avg_ttft = sum(ttft_list) / len(ttft_list)
                network_overhead_ms = max(0.0, avg_ttft - prompt_eval_ms - model_load_ms)

                # prefill/decode 分离 TPS（判断算力 vs 带宽瓶颈）
                total_prompt_tokens = sum(s.get("prompt_eval_count", 0) for s in ollama_stats_list)
                total_eval_tokens = sum(s.get("eval_count", 0) for s in ollama_stats_list)
                total_prompt_dur_s = sum(
                    s["prompt_eval_duration_ns"] for s in ollama_stats_list
                ) / 1e9
                total_eval_dur_s = sum(
                    s["eval_duration_ns"] for s in ollama_stats_list
                ) / 1e9
                prefill_tps = (
                    round(total_prompt_tokens / total_prompt_dur_s, 2)
                    if total_prompt_dur_s > 0 else 0.0
                )
                decode_tps = (
                    round(total_eval_tokens / total_eval_dur_s, 2)
                    if total_eval_dur_s > 0 else 0.0
                )
                prefill_decode_ratio = (
                    round(prompt_eval_ms / token_gen_ms, 3)
                    if token_gen_ms > 0 else 0.0
                )

                timing_breakdown = {
                    "model_load_ms": round(model_load_ms, 2),
                    "prompt_eval_ms": round(prompt_eval_ms, 2),
                    "token_gen_ms": round(token_gen_ms, 2),
                    "network_overhead_ms": round(network_overhead_ms, 2),
                    "tokens_per_second": decode_tps,
                    # 新增：prefill/decode 分离 TPS
                    "prefill_tokens_per_second": prefill_tps,
                    "decode_tokens_per_second": decode_tps,
                    "prefill_decode_ratio": prefill_decode_ratio,
                }

            results_by_config[key] = {
                "ttft": ttft_stats,
                "total_latency": total_stats,
                "tpot_ms": avg_tpot,
                "throughput_tps": throughput,
                "timing_breakdown": timing_breakdown,
            }

            # 检查阈值
            ttft_threshold = config.thresholds.get("ttft_ms", ThresholdConfig())
            tpot_threshold = config.thresholds.get("tpot_ms", ThresholdConfig())
            throughput_threshold = config.thresholds.get("throughput_tps", ThresholdConfig())

            ttft_status = (
                BenchmarkStatus.PASS
                if ttft_threshold.check(ttft_stats.get("p99_ms", 0))
                else BenchmarkStatus.FAIL
            )
            tpot_status = (
                BenchmarkStatus.PASS
                if tpot_threshold.check(avg_tpot)
                else BenchmarkStatus.FAIL
            )
            throughput_status = (
                BenchmarkStatus.PASS
                if throughput_threshold.check(throughput)
                else BenchmarkStatus.FAIL
            )

            all_metrics.extend([
                MetricResult(
                    name=f"ttft_p50_ms_{key}",
                    value=round(ttft_stats.get("p50_ms", 0), 2),
                    unit="ms",
                    higher_is_better=False,
                    status=BenchmarkStatus.PASS,
                ),
                MetricResult(
                    name=f"ttft_p99_ms_{key}",
                    value=round(ttft_stats.get("p99_ms", 0), 2),
                    unit="ms",
                    higher_is_better=False,
                    threshold=ttft_threshold.max_value,
                    status=ttft_status,
                ),
                MetricResult(
                    name=f"tpot_ms_{key}",
                    value=round(avg_tpot, 3),
                    unit="ms/token",
                    higher_is_better=False,
                    threshold=tpot_threshold.max_value,
                    status=tpot_status,
                ),
                MetricResult(
                    name=f"throughput_tps_{key}",
                    value=round(throughput, 2),
                    unit="tokens/s",
                    higher_is_better=True,
                    threshold=throughput_threshold.min_value,
                    status=throughput_status,
                ),
            ])

    # 整体状态：任一指标 FAIL 则整体 FAIL
    overall_status = (
        BenchmarkStatus.FAIL
        if any(m.status == BenchmarkStatus.FAIL for m in all_metrics)
        else BenchmarkStatus.PASS
    )

    # 提取所有 timing_breakdown 合并到 metadata 顶层（取第一个配置的分解数据）
    first_breakdown = next(
        (v["timing_breakdown"] for v in results_by_config.values() if v.get("timing_breakdown")),
        None,
    )

    return TaskResult(
        task_name="performance",
        model_name=model_name,
        metrics=all_metrics,
        num_samples=len(config.prompt_lengths) * len(config.output_lengths) * config.num_test_requests,
        duration_seconds=time.time() - start_time,
        status=overall_status,
        metadata={
            "results_by_config": results_by_config,
            "timing_breakdown": first_breakdown,
        },
    )


def run_concurrency_stress(
    backend,
    concurrency_levels: List[int],
    model_name: str,
    prompt: Optional[str] = None,
    max_tokens: int = 128,
    requests_per_level: int = 12,
) -> TaskResult:
    """并发压力测试：在多个并发度下测量 QPS 和 P95 延迟。

    与 ResourceMonitor 联动，记录各并发度下的 GPU/CPU 峰值利用率。
    """
    from benchmark.llama_benchmark.utils.resource_monitor import ResourceMonitor

    start_time = time.time()
    if prompt is None:
        prompt = _make_prompt(128)

    all_metrics: List[MetricResult] = []
    stress_results: Dict[str, Any] = {}

    for concurrency in concurrency_levels:
        logger.info(f"[{model_name}] 并发压测 concurrency={concurrency}...")

        monitor = ResourceMonitor(interval_ms=100)
        monitor.start()
        wall_start = time.perf_counter()
        total_tokens = 0
        latencies: List[float] = []

        def _single_request(_: int) -> Tuple[int, float]:
            t0 = time.perf_counter()
            try:
                if hasattr(backend, "generate_with_stats"):
                    text, stats = backend.generate_with_stats(prompt, max_tokens=max_tokens)
                    tokens = stats.get("eval_count") or stats.get("completion_tokens", max_tokens)
                else:
                    backend.generate(prompt, max_tokens=max_tokens)
                    tokens = max_tokens
            except Exception as e:
                logger.warning(f"并发请求失败: {e}")
                tokens = 0
            return tokens, (time.perf_counter() - t0) * 1000  # ms

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_single_request, i) for i in range(requests_per_level)]
            for fut in as_completed(futures):
                tokens, lat_ms = fut.result()
                total_tokens += tokens
                if lat_ms > 0:
                    latencies.append(lat_ms)

        wall_elapsed = time.perf_counter() - wall_start
        monitor.stop()
        resource_summary = monitor.get_summary()

        qps = round(len(latencies) / wall_elapsed, 2) if wall_elapsed > 0 else 0.0
        tps = round(total_tokens / wall_elapsed, 2) if wall_elapsed > 0 else 0.0
        latencies_sorted = sorted(latencies)
        p95_idx = int(len(latencies_sorted) * 0.95)
        p95_lat = round(latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)], 2) if latencies_sorted else 0.0

        stress_results[f"c{concurrency}"] = {
            "qps": qps,
            "tps": tps,
            "p95_latency_ms": p95_lat,
            "total_requests": len(latencies),
            "resource_summary": resource_summary.to_dict(),
        }

        all_metrics.extend([
            MetricResult(
                name=f"concurrency_{concurrency}_qps",
                value=qps,
                unit="req/s",
                higher_is_better=True,
                status=BenchmarkStatus.PASS,
            ),
            MetricResult(
                name=f"concurrency_{concurrency}_tps",
                value=tps,
                unit="tokens/s",
                higher_is_better=True,
                status=BenchmarkStatus.PASS,
            ),
            MetricResult(
                name=f"concurrency_{concurrency}_p95_latency_ms",
                value=p95_lat,
                unit="ms",
                higher_is_better=False,
                status=BenchmarkStatus.PASS,
            ),
        ])
        if resource_summary.gpu_util_p95_percent is not None:
            all_metrics.append(MetricResult(
                name=f"concurrency_{concurrency}_gpu_util_p95",
                value=resource_summary.gpu_util_p95_percent,
                unit="%",
                higher_is_better=False,
                status=BenchmarkStatus.PASS,
            ))

    return TaskResult(
        task_name="concurrency_stress",
        model_name=model_name,
        metrics=all_metrics,
        num_samples=sum(c * requests_per_level for c in concurrency_levels),
        duration_seconds=time.time() - start_time,
        status=BenchmarkStatus.PASS,
        metadata={"stress_results": stress_results, "concurrency_levels": concurrency_levels},
    )


def run_sustained_load(
    backend,
    model_name: str,
    duration_s: int = 60,
    window_s: int = 10,
    max_tokens: int = 128,
    prompt: Optional[str] = None,
) -> TaskResult:
    """持续负载测试：检测 TPS 随时间的衰减（热降频信号）。

    持续发送请求 duration_s 秒，每 window_s 秒统计一个 TPS 窗口。
    输出 tps_windows（列表）和 tps_degradation_pct（首末衰减百分比）。
    """
    from benchmark.llama_benchmark.utils.resource_monitor import ResourceMonitor

    if prompt is None:
        prompt = _make_prompt(128)

    start_time = time.time()
    tps_windows: List[float] = []
    window_start = time.perf_counter()
    window_tokens = 0
    monitor = ResourceMonitor(interval_ms=100)
    monitor.start()
    monitor.record_event("sustained_load_start")

    deadline = time.perf_counter() + duration_s
    all_metrics: List[MetricResult] = []

    while time.perf_counter() < deadline:
        try:
            if hasattr(backend, "generate_with_stats"):
                _, stats = backend.generate_with_stats(prompt, max_tokens=max_tokens)
                tokens = stats.get("eval_count") or stats.get("completion_tokens", max_tokens)
            else:
                backend.generate(prompt, max_tokens=max_tokens)
                tokens = max_tokens
            window_tokens += tokens
        except Exception as e:
            logger.warning(f"[{model_name}] sustained_load 请求失败: {e}")

        now = time.perf_counter()
        if now - window_start >= window_s:
            elapsed = now - window_start
            window_tps = round(window_tokens / elapsed, 2) if elapsed > 0 else 0.0
            tps_windows.append(window_tps)
            window_tokens = 0
            window_start = now
            monitor.record_event(f"window_{len(tps_windows)}_end")

    monitor.record_event("sustained_load_end")
    monitor.stop()
    resource_summary = monitor.get_summary()

    # 计算 TPS 衰减
    tps_degradation_pct = 0.0
    if len(tps_windows) >= 2 and tps_windows[0] > 0:
        tps_degradation_pct = round(
            (tps_windows[0] - tps_windows[-1]) / tps_windows[0] * 100, 2
        )

    all_metrics.extend([
        MetricResult(
            name="sustained_tps_initial",
            value=tps_windows[0] if tps_windows else 0.0,
            unit="tokens/s",
            higher_is_better=True,
            status=BenchmarkStatus.PASS,
        ),
        MetricResult(
            name="sustained_tps_final",
            value=tps_windows[-1] if tps_windows else 0.0,
            unit="tokens/s",
            higher_is_better=True,
            status=BenchmarkStatus.PASS,
        ),
        MetricResult(
            name="tps_degradation_pct",
            value=tps_degradation_pct,
            unit="%",
            higher_is_better=False,
            status=BenchmarkStatus.FAIL if tps_degradation_pct > 20 else BenchmarkStatus.PASS,
        ),
    ])

    return TaskResult(
        task_name="sustained_load",
        model_name=model_name,
        metrics=all_metrics,
        num_samples=len(tps_windows),
        duration_seconds=time.time() - start_time,
        status=BenchmarkStatus.PASS,
        metadata={
            "sustained_load": {
                "tps_windows": tps_windows,
                "tps_degradation_pct": tps_degradation_pct,
                "duration_s": duration_s,
                "window_s": window_s,
            },
            "resource_summary": resource_summary.to_dict(),
        },
    )
