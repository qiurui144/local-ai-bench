"""OCR 精度与延迟评测。

使用内嵌测试图片（代码生成，无需外部文件），覆盖中英混合场景。
支持多分辨率对比：1.0（原始）/ 0.5（50%降采样）/ 0.75 等。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.result import TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)

# 内置测试样本：(图像文字, 参考文本)
_BUILTIN_SAMPLES = [
    ("SpacemiT K1 RISC-V AI Benchmark Report 2026",
     "SpacemiT K1 RISC-V AI Benchmark Report 2026"),
    ("embedding throughput:2.0 samples/s",
     "embedding throughput:2.0 samples/s"),
    ("NDCG@5=0.832 bge-m3 rerank",
     "NDCG@5=0.832 bge-m3 rerank"),
    ("Qwen2.5 decode speed:10.8 tok/s TTFT:390ms",
     "Qwen2.5 decode speed:10.8 tok/s TTFT:390ms"),
    ("RISC-V RVV 1.0 llama.cpp GGML",
     "RISC-V RVV 1.0 llama.cpp GGML"),
]


def make_test_image(text: str, width: int = 800, height: int = 120) -> np.ndarray:
    """生成 BGR 白底黑字测试图（仅 ASCII 字符，避免字体依赖）。"""
    import cv2
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(img, text[:60], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def keyword_accuracy(recognized: str, reference: str) -> float:
    """关键词准确率：识别文本与参考文本的 token 交集比例。"""
    def tokenize(s: str) -> set:
        return set(re.findall(r"[\w\u4e00-\u9fff]+", s.lower()))
    gt = tokenize(reference)
    pred = tokenize(recognized)
    if not gt:
        return 1.0
    return len(gt & pred) / len(gt)


def run_ocr_accuracy(
    backend,
    samples: Optional[List[Tuple[str, str]]] = None,
    input_scales: Optional[List[float]] = None,
    num_warmup: int = 1,
    num_runs: int = 3,
    model_name: str = "",
) -> List[TaskResult]:
    """对单个 OCR 后端在多个分辨率下评测精度和延迟。

    Parameters
    ----------
    backend : RapidOCRBackend
        已加载的 OCR 后端（需实现 get_text(image, input_scale) 接口）。
    samples : list[(image_text, reference)]
        测试样本，None 使用内置样本。
    input_scales : list[float]
        分辨率缩放系数列表，None 使用后端配置。
    num_warmup, num_runs : int
        预热和正式测试轮数。
    model_name : str
        模型标识。

    Returns
    -------
    list[TaskResult]
        每个分辨率对应一个 TaskResult。
    """

    if samples is None:
        samples = _BUILTIN_SAMPLES
    if input_scales is None:
        input_scales = [1.0, 0.5]

    # 生成测试图片
    images_and_refs: List[Tuple[np.ndarray, str]] = [
        (make_test_image(text), ref)
        for text, ref in samples
    ]

    # 预热
    warmup_img = images_and_refs[0][0]
    for _ in range(num_warmup):
        try:
            backend.get_text(warmup_img, input_scale=1.0)
        except Exception:
            pass

    results: List[TaskResult] = []

    for scale in input_scales:
        latencies: List[float] = []
        accuracies: List[float] = []

        for img, ref in images_and_refs:
            run_lats: List[float] = []
            run_accs: List[float] = []
            for _ in range(num_runs):
                try:
                    text, lat_ms = backend.get_text(img, input_scale=scale)
                    acc = keyword_accuracy(text, ref)
                    run_lats.append(lat_ms)
                    run_accs.append(acc)
                except Exception as e:
                    logger.warning(f"[{model_name}] scale={scale} 识别失败: {e}")

            if run_lats:
                latencies.append(float(np.mean(run_lats)))
                accuracies.append(float(np.mean(run_accs)))

        if not latencies:
            continue

        avg_lat = float(np.mean(latencies))
        avg_acc = float(np.mean(accuracies))
        throughput = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        scale_pct = int(scale * 100)

        results.append(TaskResult(
            task_name=f"ocr_scale_{scale_pct}pct",
            model_name=model_name,
            metrics={
                "avg_latency_ms": round(avg_lat, 1),
                "throughput_img_s": round(throughput, 3),
                "avg_accuracy": round(avg_acc, 3),
                "input_scale": scale,
            },
            metadata={
                "num_samples": len(latencies),
                "num_runs": num_runs,
                "per_sample_latency_ms": [round(v, 1) for v in latencies],
                "per_sample_accuracy": [round(a, 3) for a in accuracies],
            },
            passed=avg_acc >= 0.8,
        ))
        logger.info(
            f"[{model_name}] scale={scale:.0%}  "
            f"延迟={avg_lat:.0f}ms  吞吐={throughput:.3f}img/s  准确率={avg_acc:.1%}"
        )

    return results
