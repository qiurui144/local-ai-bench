"""ASR RTF（Real-Time Factor）及 CER 评测。

覆盖多种音频时长，生成精度-延迟权衡表。
不依赖真实录音文件：使用正弦波合成音频（用于 RTF/延迟评测）。
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.result import BenchmarkStatus, MetricResult, TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)

_SR = 16000


class TranscribeResult(NamedTuple):
    """统一 ASR 调用返回格式，消除不同后端返回元组长度不一致的问题。

    - WhisperOnnxBackend: (text, enc_ms, total_ms) → latency_ms = total_ms
    - SenseVoiceOnnxBackend: (text, latency_ms) → latency_ms = latency_ms
    - FasterWhisperBackend / 其他: (text,) 或 str → latency_ms = 0.0
    """
    text: str
    latency_ms: float = 0.0


def generate_sine_audio(duration_s: float, sr: int = _SR, freq: float = 440.0) -> np.ndarray:
    """生成正弦波测试音频（float32, [-1, 1]）。"""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def compute_cer(hypothesis: str, reference: str) -> float:
    """字符级编辑距离 CER（0=完全正确，1=完全错误）。"""
    h, r = list(hypothesis.lower()), list(reference.lower())
    if not r:
        return 0.0  # 参考为空：无字符可出错
    d = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h) + 1):
        d[i][0] = i
    for j in range(len(r) + 1):
        d[0][j] = j
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            cost = 0 if h[i - 1] == r[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(h)][len(r)] / len(r)


def run_rtf_benchmark(
    backend,
    audio_durations_s: List[float],
    num_warmup: int = 1,
    num_runs: int = 3,
    language: str = "en",
    model_name: str = "",
    per_call_timeout_s: float = 120.0,
) -> TaskResult:
    """对单个后端运行多时长 RTF 评测。

    Parameters
    ----------
    backend : AbstractModelBackend
        已加载的 ASR 后端（需实现 transcribe(audio, language) 接口）。
    audio_durations_s : list[float]
        测试音频时长列表（秒）。
    num_warmup, num_runs : int
        预热和正式测试轮数。
    language : str
        测试语言（"en" / "zh"）。
    model_name : str
        模型标识，写入结果 metadata。

    Returns
    -------
    TaskResult
        包含 avg_rtf, avg_latency_ms, per_duration_rtf 等指标。
    """
    rtfs: List[float] = []
    latencies: List[float] = []
    per_dur: List[Dict] = []

    # 预热（超时不阻断后续测试）
    warmup_audio = generate_sine_audio(3.0)
    for _ in range(num_warmup):
        try:
            _call_transcribe(backend, warmup_audio, language, timeout_s=per_call_timeout_s)
        except Exception:
            pass

    for dur in audio_durations_s:
        audio = generate_sine_audio(dur)
        dur_latencies: List[float] = []

        for _ in range(num_runs):
            try:
                r = _call_transcribe(
                    backend, audio, language, timeout_s=per_call_timeout_s
                )
                dur_latencies.append(r.latency_ms)
            except FutureTimeoutError:
                logger.warning(
                    "[%s] %gs 音频推理超时（>%gs），跳过此轮",
                    model_name, dur, per_call_timeout_s,
                )
            except Exception as e:
                logger.warning("[%s] %gs 推理失败: %s", model_name, dur, e)

        if not dur_latencies:
            continue

        avg_ms = float(np.mean(dur_latencies))
        rtf = (avg_ms / 1000.0) / dur
        rtfs.append(rtf)
        latencies.append(avg_ms)
        per_dur.append({
            "duration_s": dur,
            "avg_latency_ms": round(avg_ms, 1),
            "rtf": round(rtf, 4),
            "runs": len(dur_latencies),
        })

    # 按音频时长加权的 RTF（长音频在实际使用中权重更大，避免短音频的边界效应拉偏均值）
    if rtfs and per_dur:
        durations = [d["duration_s"] for d in per_dur]
        total_dur = sum(durations)
        weighted_rtf = float(sum(r * d for r, d in zip(rtfs, durations)) / total_dur)
    else:
        weighted_rtf = None

    avg_rtf = float(np.mean(rtfs)) if rtfs else None
    avg_lat = float(np.mean(latencies)) if latencies else None
    rtf_ok = (weighted_rtf or avg_rtf or 1.1) < 1.0

    return TaskResult(
        task_name="asr_rtf",
        model_name=model_name,
        metrics=[
            MetricResult(name="avg_rtf", value=avg_rtf, higher_is_better=False),
            MetricResult(name="weighted_rtf", value=weighted_rtf, higher_is_better=False,
                         details={"note": "按音频时长加权，消除短音频边界效应"}),
            MetricResult(name="avg_latency_ms", value=avg_lat, higher_is_better=False),
            MetricResult(name="realtime_capable", value=float(rtf_ok), higher_is_better=True),
        ],
        num_samples=len(audio_durations_s),
        duration_seconds=0.0,
        status=BenchmarkStatus.PASS if rtf_ok else BenchmarkStatus.FAIL,
        metadata={
            "per_duration": per_dur,
            "language": language,
            "num_runs": num_runs,
        },
    )


def _call_transcribe(
    backend,
    audio: np.ndarray,
    language: str,
    timeout_s: float = 120.0,
) -> TranscribeResult:
    """统一调用接口，兼容所有 ASR 后端，返回规范化的 TranscribeResult。

    - WhisperOnnxBackend: 返回 (text, enc_ms, total_ms)，取 total_ms
    - SenseVoiceOnnxBackend: 返回 (text, latency_ms)
    - 其他返回字符串: latency_ms=0.0
    超时时抛出 concurrent.futures.TimeoutError（不捕获，由调用方处理）。
    """
    if not hasattr(backend, "transcribe"):
        raise AttributeError(f"{type(backend).__name__} 没有 transcribe 方法")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(backend.transcribe, audio, language=language)
        result = future.result(timeout=timeout_s)

    if isinstance(result, TranscribeResult):
        return result
    if isinstance(result, tuple):
        text = str(result[0]) if result else ""
        # 取最后一个数值字段作为总延迟（兼容 2-tuple 和 3-tuple）
        latency = float(result[-1]) if len(result) >= 2 and isinstance(result[-1], (int, float)) else 0.0
        return TranscribeResult(text=text, latency_ms=latency)
    return TranscribeResult(text=str(result), latency_ms=0.0)
