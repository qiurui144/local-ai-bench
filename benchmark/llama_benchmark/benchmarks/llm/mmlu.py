"""MMLU Benchmark：57 学科多选题，使用 logprob 评分。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from tqdm import tqdm

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    SampleResult,
    TaskResult,
)
from benchmark.llama_benchmark.utils.logging import get_logger

if TYPE_CHECKING:
    from benchmark.llama_benchmark.backends.ollama_backend import OllamaBackend
    from benchmark.llama_benchmark.backends.llama_backend import LlamaCppBackend

logger = get_logger(__name__)

CHOICES = ["A", "B", "C", "D"]

MMLU_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the following multiple choice question by selecting A, B, C, or D."
)


def build_prompt(question: str, choices: List[str], few_shot_examples: str = "") -> str:
    """构造 MMLU 提示词。"""
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(CHOICES, choices))
    prompt = f"{few_shot_examples}Question: {question}\n{options}\nAnswer:"
    return prompt


def predict_choice(
    backend,
    prompt: str,
) -> Tuple[str, Dict[str, float]]:
    """使用 logprob 方式预测选项，返回 (预测选项, {选项: logprob})。"""
    logprobs = backend.generate_with_logprobs(prompt, CHOICES, max_tokens=1)
    best = max(logprobs, key=lambda k: logprobs[k])
    return best, logprobs


def run_mmlu(
    backend,
    config: BenchmarkTaskConfig,
    model_name: str,
) -> TaskResult:
    """执行 MMLU benchmark，返回 TaskResult。"""
    start_time = time.time()

    from benchmark.llama_benchmark.datasets.mmlu_dataset import MMLUDataset

    logger.info("加载 MMLU 数据集...")
    try:
        dataset = MMLUDataset(
            num_samples=config.num_samples,
            dataset_path=config.dataset_path,
        )
        samples = dataset.load()
    except Exception as e:
        return TaskResult(
            task_name="mmlu",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=0.0,
            status=BenchmarkStatus.ERROR,
            error_message=f"数据集加载失败: {e}",
        )

    correct = 0
    total = len(samples)
    subject_correct: Dict[str, int] = {}
    subject_total: Dict[str, int] = {}
    sample_results: List[SampleResult] = []

    for item in tqdm(samples, desc="MMLU", unit="q"):
        subject = item.get("subject", "unknown")
        subject_total[subject] = subject_total.get(subject, 0) + 1

        prompt = build_prompt(item["question"], item["choices"])
        expected = item["answer"]  # 数据集加载器已归一化为 A/B/C/D 字母

        infer_start = time.perf_counter_ns()
        try:
            predicted, logprobs = predict_choice(backend, prompt)
        except Exception as e:
            logger.warning(f"MMLU 推理失败: {e}")
            predicted = "A"
            logprobs = {}
        latency_ms = (time.perf_counter_ns() - infer_start) / 1_000_000

        is_correct = predicted == expected
        if is_correct:
            correct += 1
            subject_correct[subject] = subject_correct.get(subject, 0) + 1

        sample_results.append(
            SampleResult(
                sample_id=str(item.get("question", "")[:30]),
                input=item["question"],
                expected=expected,
                predicted=predicted,
                correct=is_correct,
                latency_ms=latency_ms,
                metadata={"subject": subject, "logprobs": logprobs},
            )
        )

    accuracy = correct / total if total > 0 else 0.0
    subject_accuracy = {
        s: subject_correct.get(s, 0) / subject_total[s]
        for s in subject_total
    }

    threshold = config.thresholds.get("accuracy", ThresholdConfig())
    status = (
        BenchmarkStatus.PASS if threshold.check(accuracy) else BenchmarkStatus.FAIL
    )

    metrics = [
        MetricResult(
            name="accuracy",
            value=round(accuracy, 4),
            unit="",
            higher_is_better=True,
            threshold=threshold.min_value,
            status=status,
        ),
        MetricResult(
            name="accuracy_by_subject",
            value=None,
            details={s: round(v, 4) for s, v in subject_accuracy.items()},
        ),
    ]

    return TaskResult(
        task_name="mmlu",
        model_name=model_name,
        metrics=metrics,
        num_samples=total,
        duration_seconds=time.time() - start_time,
        status=status,
        sample_results=sample_results,
        metadata={"num_subjects": len(subject_total)},
    )
