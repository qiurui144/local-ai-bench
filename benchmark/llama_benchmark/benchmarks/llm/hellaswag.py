"""HellaSwag Benchmark：常识推理，length-normalized logprob 评分。"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from tqdm import tqdm

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    SampleResult,
    TaskResult,
)
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


def build_ctx(activity_label: str, ctx_a: str, ctx_b: str) -> str:
    return f"{activity_label}: {ctx_a} {ctx_b}".strip()


def run_hellaswag(
    backend,
    config: BenchmarkTaskConfig,
    model_name: str,
) -> TaskResult:
    """执行 HellaSwag benchmark，使用 logprob 方式评分。

    HellaSwag 的 4 个选项长度不同，直接比较 logprob 会偏向短选项。
    此处使用 length-normalized logprob：将整个 context+ending 送入模型，
    用最后 len(ending_tokens) 个 token 的平均 logprob 作为分数。
    Ollama 后端：使用 generate_with_logprobs 近似（取首 token logprob）。
    """
    start_time = time.time()

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")

    dataset_path = str(config.dataset_path) if config.dataset_path else "Rowan/hellaswag"
    logger.info(f"加载 HellaSwag 数据集: {dataset_path}")

    try:
        ds = load_dataset(dataset_path, split="validation", trust_remote_code=True)
    except Exception as e:
        return TaskResult(
            task_name="hellaswag",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=0.0,
            status=BenchmarkStatus.ERROR,
            error_message=f"数据集加载失败: {e}",
        )

    if config.num_samples:
        ds = ds.select(range(min(config.num_samples, len(ds))))

    correct = 0
    total = len(ds)
    sample_results: List[SampleResult] = []

    for item in tqdm(ds, desc="HellaSwag", unit="q"):
        ctx = build_ctx(
            item.get("activity_label", ""),
            item.get("ctx_a", ""),
            item.get("ctx_b", ""),
        )
        endings: List[str] = item["endings"]
        label = int(item["label"])

        # 对每个 ending 构造完整文本，用首 token logprob 近似 length-normalized 分数
        best_idx = 0
        best_score = float("-inf")
        scores: Dict[int, float] = {}

        infer_start = time.perf_counter_ns()
        for i, ending in enumerate(endings):
            prompt = f"{ctx} {ending}"
            # 用最后一个词作为 logprob 候选（近似）
            last_word = ending.strip().split()[-1] if ending.strip() else ending
            try:
                logprobs = backend.generate_with_logprobs(prompt, [last_word])
                score = logprobs.get(last_word, float("-inf"))
                # Length normalization
                score = score / max(len(ending.split()), 1)
            except Exception:
                score = float("-inf")
            scores[i] = score
            if score > best_score:
                best_score = score
                best_idx = i
        latency_ms = (time.perf_counter_ns() - infer_start) / 1_000_000

        is_correct = best_idx == label
        if is_correct:
            correct += 1

        sample_results.append(
            SampleResult(
                sample_id=item.get("ind", str(len(sample_results))),
                input=ctx,
                expected=str(label),
                predicted=str(best_idx),
                correct=is_correct,
                latency_ms=latency_ms,
                metadata={"scores": scores},
            )
        )

    accuracy = correct / total if total > 0 else 0.0
    threshold = config.thresholds.get("accuracy", ThresholdConfig())
    status = (
        BenchmarkStatus.PASS if threshold.check(accuracy) else BenchmarkStatus.FAIL
    )

    return TaskResult(
        task_name="hellaswag",
        model_name=model_name,
        metrics=[
            MetricResult(
                name="accuracy",
                value=round(accuracy, 4),
                higher_is_better=True,
                threshold=threshold.min_value,
                status=status,
            )
        ],
        num_samples=total,
        duration_seconds=time.time() - start_time,
        status=status,
        sample_results=sample_results,
    )
