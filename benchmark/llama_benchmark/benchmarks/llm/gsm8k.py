"""GSM8K Benchmark：8500 道小学数学题，CoT 8-shot 推理。"""

from __future__ import annotations

import re
import time
from typing import List, Optional

from tqdm import tqdm

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    SampleResult,
    TaskResult,
)
from benchmark.llama_benchmark.datasets.base_dataset import synthetic_fallback_metadata
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)

# 8-shot CoT 示例（来自原始论文）
GSM8K_FEW_SHOT = """\
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is #### 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is #### 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is #### 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is #### 8

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is #### 9

Q: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is #### 29

Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
A: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. The answer is #### 33

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. 23 - 15 = 8. The answer is #### 8

"""


def extract_answer(text: str) -> Optional[str]:
    """从模型输出中提取 #### 后的数字答案。"""
    # 匹配 "#### <数字>" 格式
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # 尝试匹配最后一个数字
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def run_gsm8k(
    backend,
    config: BenchmarkTaskConfig,
    model_name: str,
) -> TaskResult:
    """执行 GSM8K benchmark。"""
    start_time = time.time()

    from benchmark.llama_benchmark.datasets.gsm8k_dataset import GSM8KDataset

    logger.info("加载 GSM8K 数据集...")
    try:
        dataset = GSM8KDataset(num_samples=config.num_samples, dataset_path=config.dataset_path)
        samples = dataset.load()
    except Exception as e:
        return TaskResult(
            task_name="gsm8k",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=0.0,
            status=BenchmarkStatus.ERROR,
            error_message=f"数据集加载失败: {e}",
        )

    correct = 0
    total = len(samples)
    sample_results: List[SampleResult] = []

    for item in tqdm(samples, desc="GSM8K", unit="q"):
        question = item["question"]
        # 从 answer 字段提取标准答案
        ref_answer = extract_answer(item["answer"])

        prompt = GSM8K_FEW_SHOT + f"Q: {question}\nA:"

        infer_start = time.perf_counter_ns()
        try:
            predicted_text = backend.generate(
                prompt,
                max_tokens=512,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"GSM8K 推理失败: {e}")
            predicted_text = ""
        latency_ms = (time.perf_counter_ns() - infer_start) / 1_000_000

        pred_answer = extract_answer(predicted_text)
        is_correct = (pred_answer is not None and ref_answer is not None
                      and pred_answer == ref_answer)
        if is_correct:
            correct += 1

        sample_results.append(
            SampleResult(
                sample_id=question[:30],
                input=question,
                expected=ref_answer or "",
                predicted=pred_answer or "",
                correct=is_correct,
                latency_ms=latency_ms,
            )
        )

    accuracy = correct / total if total > 0 else 0.0
    threshold = config.thresholds.get("accuracy", ThresholdConfig())
    status = (
        BenchmarkStatus.PASS if threshold.check(accuracy) else BenchmarkStatus.FAIL
    )

    return TaskResult(
        task_name="gsm8k",
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
        metadata=synthetic_fallback_metadata(dataset),
    )
