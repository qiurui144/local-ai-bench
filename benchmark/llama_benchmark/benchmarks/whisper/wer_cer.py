"""Whisper WER/CER Benchmark：基于 LibriSpeech / Common Voice 数据集。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    SampleResult,
    TaskResult,
)
from benchmark.llama_benchmark.datasets.librispeech_dataset import (
    librispeech_revision,
)
from benchmark.llama_benchmark.metrics.nlp import CERMetric, WERMetric
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


def _load_librispeech(dataset_path: str, num_samples: Optional[int]) -> List[Tuple[str, str]]:
    """加载 LibriSpeech 数据集，返回 [(audio_path, reference_text), ...]。"""
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")

    logger.info(f"加载 LibriSpeech: {dataset_path}")
    # parquet 版(datasets>=3)config="clean" 的测试 split 名是 "test"(legacy 脚本
    # 时代的 "test.clean" 已不存在);仅默认 hub 数据集 pin revision。加载失败直接
    # 抛出,由 run_wer_cer 转为 ERROR TaskResult — 不允许静默换数据源。
    load_kwargs = (
        {"revision": librispeech_revision()}
        if dataset_path == "openslr/librispeech_asr"
        else {}
    )
    ds = load_dataset(dataset_path, "clean", split="test", **load_kwargs)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    samples = []
    tmp_dir = Path("/tmp/librispeech_wav")
    tmp_dir.mkdir(exist_ok=True)

    for i, item in enumerate(ds):
        import soundfile as sf
        audio_data = item["audio"]
        wav_path = str(tmp_dir / f"sample_{i}.wav")
        sf.write(wav_path, audio_data["array"], audio_data["sampling_rate"])
        ref_text = item.get("text", item.get("sentence", ""))
        samples.append((wav_path, ref_text))

    return samples


def run_wer_cer(
    backend,
    config: BenchmarkTaskConfig,
    model_name: str,
    dataset_name: str = "librispeech",
    language: str = "en",
    beam_size: int = 5,
) -> TaskResult:
    """执行 WER/CER benchmark。"""
    start_time = time.time()

    dataset_path = (
        str(config.dataset_path) if config.dataset_path else "openslr/librispeech_asr"
    )

    # 加载音频数据
    try:
        samples = _load_librispeech(dataset_path, config.num_samples)
    except Exception as e:
        return TaskResult(
            task_name="wer_cer",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=0.0,
            status=BenchmarkStatus.ERROR,
            error_message=f"数据集加载失败: {e}",
        )

    predictions: List[str] = []
    references: List[str] = []
    sample_results: List[SampleResult] = []

    for audio_path, ref_text in tqdm(samples, desc="Whisper WER/CER", unit="audio"):
        infer_start = time.perf_counter_ns()
        try:
            if hasattr(backend, "transcribe"):
                # FasterWhisperBackend
                pred_text, _ = backend.transcribe(audio_path, language=language, beam_size=beam_size)
            else:
                # OllamaBackend（通过 generate 接口）
                pred_text = backend.generate(f"[Transcribe audio: {audio_path}]", max_tokens=512)
        except Exception as e:
            logger.warning(f"转录失败: {e}")
            pred_text = ""
        latency_ms = (time.perf_counter_ns() - infer_start) / 1_000_000

        predictions.append(pred_text)
        references.append(ref_text)
        sample_results.append(
            SampleResult(
                sample_id=os.path.basename(audio_path),
                input=audio_path,
                expected=ref_text,
                predicted=pred_text,
                correct=False,
                latency_ms=latency_ms,
            )
        )

    wer_metric = WERMetric(normalize=True)
    cer_metric = CERMetric(normalize=True)

    wer_val = wer_metric.compute(predictions, references)
    cer_val = cer_metric.compute(predictions, references)

    wer_threshold = config.thresholds.get("wer", ThresholdConfig())
    cer_threshold = config.thresholds.get("cer", ThresholdConfig())

    wer_status = BenchmarkStatus.PASS if wer_threshold.check(wer_val) else BenchmarkStatus.FAIL
    cer_status = BenchmarkStatus.PASS if cer_threshold.check(cer_val) else BenchmarkStatus.FAIL
    overall = (
        BenchmarkStatus.FAIL
        if BenchmarkStatus.FAIL in (wer_status, cer_status)
        else BenchmarkStatus.PASS
    )

    return TaskResult(
        task_name="wer_cer",
        model_name=model_name,
        metrics=[
            MetricResult(
                name="wer",
                value=round(wer_val, 4),
                unit="",
                higher_is_better=False,
                threshold=wer_threshold.max_value,
                status=wer_status,
            ),
            MetricResult(
                name="cer",
                value=round(cer_val, 4),
                unit="",
                higher_is_better=False,
                threshold=cer_threshold.max_value,
                status=cer_status,
            ),
        ],
        num_samples=len(samples),
        duration_seconds=time.time() - start_time,
        status=overall,
        sample_results=sample_results,
    )
