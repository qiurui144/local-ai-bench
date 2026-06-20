"""说话人分离 benchmark：DER 评测。

评测流程：
1. 加载数据集（AMI / AISHELL-4 / CallHome 等）
2. 对每个音频调用 backend.diarize()
3. 与参考 RTTM 比对，使用 compute_der() 计算 DER 及分量
4. 汇报宏平均 DER 和各分量指标
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig
from benchmark.llama_benchmark.core.result import BenchmarkStatus, MetricResult, TaskResult
from benchmark.llama_benchmark.datasets.base_dataset import SYNTHETIC_FALLBACK_WARNING
from benchmark.llama_benchmark.metrics.speaker import compute_der, compute_rtf
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)

# 类型别名
Segment = Tuple[float, float, str]


def run_diarization(
    backend: Any,
    task_cfg: BenchmarkTaskConfig,
    model_name: str,
    collar: float = 0.25,
    skip_overlap: bool = False,
    datasets: Optional[List[str]] = None,
) -> TaskResult:
    """执行说话人分离评测，返回 TaskResult。

    Args:
        backend:       实现 diarize(audio_path) 的后端实例
        task_cfg:      BenchmarkTaskConfig（enabled / num_samples / dataset_path / thresholds）
        model_name:    模型名称（用于记录）
        collar:        边界忽略窗口（秒），NIST 标准 0.25s
        skip_overlap:  是否跳过重叠语音区域
        datasets:      数据集名称列表（"ami" / "aishell4" / "callhome"）
    """
    start_time = time.monotonic()
    datasets = datasets or ["ami"]
    samples, synthetic_fallback = _load_samples(task_cfg, datasets)

    if not samples:
        return TaskResult(
            task_name="diarization",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=time.monotonic() - start_time,
            status=BenchmarkStatus.ERROR,
            error_message="未找到可用的评测样本",
            metadata=_result_metadata(
                datasets=datasets,
                collar=collar,
                skip_overlap=skip_overlap,
                num_samples=0,
                errors=[],
                synthetic_fallback=synthetic_fallback,
            ),
        )

    num_samples = task_cfg.num_samples
    if num_samples:
        samples = samples[:num_samples]

    all_ders: List[float] = []
    all_missed: List[float] = []
    all_fa: List[float] = []
    all_sc: List[float] = []
    all_rtf: List[float] = []
    errors: List[str] = []

    for audio_path, ref_segments, audio_duration in samples:
        if audio_path is None:
            # 无音频文件（单元测试用虚拟样本）
            continue
        try:
            hyp_segments, latency_ms = backend.diarize(audio_path)
            result = compute_der(ref_segments, hyp_segments, collar, skip_overlap)
            all_ders.append(result["der"])
            all_missed.append(result["missed_speech"])
            all_fa.append(result["false_alarm"])
            all_sc.append(result["speaker_confusion"])
            if audio_duration and audio_duration > 0:
                all_rtf.append(compute_rtf(latency_ms / 1000.0, audio_duration))
        except Exception as exc:
            errors.append(f"{audio_path}: {exc}")
            logger.warning(f"[{model_name}] 分离失败: {audio_path}: {exc}")

    if not all_ders:
        return TaskResult(
            task_name="diarization",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=time.monotonic() - start_time,
            status=BenchmarkStatus.ERROR,
            error_message=f"所有样本推理失败: {'; '.join(errors[:3])}",
            metadata=_result_metadata(
                datasets=datasets,
                collar=collar,
                skip_overlap=skip_overlap,
                num_samples=0,
                errors=errors,
                synthetic_fallback=synthetic_fallback,
            ),
        )

    import numpy as np

    avg_der = float(np.mean(all_ders))
    avg_missed = float(np.mean(all_missed))
    avg_fa = float(np.mean(all_fa))
    avg_sc = float(np.mean(all_sc))

    metrics: Dict[str, MetricResult] = {
        "der": MetricResult(name="der", value=round(avg_der, 4),
                            unit="", higher_is_better=False),
        "missed_speech": MetricResult(name="missed_speech", value=round(avg_missed, 4),
                                      unit="", higher_is_better=False),
        "false_alarm": MetricResult(name="false_alarm", value=round(avg_fa, 4),
                                    unit="", higher_is_better=False),
        "speaker_confusion": MetricResult(name="speaker_confusion", value=round(avg_sc, 4),
                                          unit="", higher_is_better=False),
        "num_evaluated": MetricResult(name="num_evaluated", value=len(all_ders),
                                      unit="samples", higher_is_better=True),
    }
    if all_rtf:
        metrics["rtf"] = MetricResult(
            name="rtf", value=round(float(np.mean(all_rtf)), 4),
            unit="", higher_is_better=False,
        )

    # 阈值检查
    status = BenchmarkStatus.PASS
    for metric_name, threshold_cfg in task_cfg.thresholds.items():
        if metric_name in metrics and not threshold_cfg.check(metrics[metric_name].value):
            status = BenchmarkStatus.FAIL
            metrics[metric_name].status = BenchmarkStatus.FAIL
            logger.warning(
                f"[{model_name}] {metric_name}={metrics[metric_name].value} 未达到阈值"
            )

    return TaskResult(
        task_name="diarization",
        model_name=model_name,
        metrics=list(metrics.values()),
        num_samples=len(all_ders),
        duration_seconds=time.monotonic() - start_time,
        status=status,
        metadata=_result_metadata(
            datasets=datasets,
            collar=collar,
            skip_overlap=skip_overlap,
            num_samples=len(all_ders),
            errors=errors,
            synthetic_fallback=synthetic_fallback,
        ),
    )


def _result_metadata(
    datasets: List[str],
    collar: float,
    skip_overlap: bool,
    num_samples: int,
    errors: List[str],
    synthetic_fallback: bool,
) -> Dict[str, Any]:
    """构造 TaskResult.metadata；合成回退必须机器可读地标注，不得伪装成真实分数。"""
    metadata: Dict[str, Any] = {
        "datasets": datasets,
        "collar": collar,
        "skip_overlap": skip_overlap,
        "num_samples": num_samples,
        "errors": errors[:5],
    }
    if synthetic_fallback:
        metadata["synthetic_fallback"] = True
        metadata["warning"] = SYNTHETIC_FALLBACK_WARNING
    return metadata


# ── 数据集加载 ──────────────────────────────────────────────────────────────────

def _load_samples(
    task_cfg: BenchmarkTaskConfig,
    datasets: List[str],
) -> Tuple[List[Tuple[Optional[str], List[Segment], Optional[float]]], bool]:
    """加载说话人分离评测样本。

    Returns:
        (samples, synthetic_fallback)
        samples = [(audio_path, ref_segments, audio_duration_seconds), ...]
        audio_path 为 None 时表示无音频文件（测试用）；
        synthetic_fallback = True 表示任一数据源回退到了内置合成样本，
        其得分不是真实 benchmark 分数。
    """
    # 优先使用指定本地目录
    if task_cfg.dataset_path:
        return _load_from_local(task_cfg.dataset_path), False

    samples: List[Tuple[Optional[str], List[Segment], Optional[float]]] = []
    synthetic_fallback = False
    for ds_name in datasets:
        ds_name_lower = ds_name.lower()
        try:
            # num_samples 走构造参数（与 gsm8k/mmlu 一致）；基类 load() 无参
            if "ami" in ds_name_lower:
                from benchmark.llama_benchmark.datasets.ami_dataset import AMIDataset
                ds = AMIDataset(num_samples=task_cfg.num_samples)
            elif "aishell4" in ds_name_lower or "aishell-4" in ds_name_lower:
                from benchmark.llama_benchmark.datasets.aishell4_dataset import AISHELL4Dataset
                ds = AISHELL4Dataset(num_samples=task_cfg.num_samples)
            elif "callhome" in ds_name_lower:
                from benchmark.llama_benchmark.datasets.callhome_dataset import CallhomeDataset
                ds = CallhomeDataset(num_samples=task_cfg.num_samples)
            else:
                logger.warning(f"未知数据集: {ds_name}，跳过")
                continue
            ds.load()
            samples.extend(_dataset_to_samples(ds))
            if getattr(ds, "synthetic_fallback", False):
                synthetic_fallback = True
        except Exception as e:
            logger.warning(f"加载数据集 {ds_name} 失败: {e}，使用内置虚拟样本")
            samples.extend(_builtin_samples())
            synthetic_fallback = True

    if not samples:
        samples = _builtin_samples()
        synthetic_fallback = True

    return samples, synthetic_fallback


def _dataset_to_samples(ds: Any) -> List[Tuple[Optional[str], List[Segment], Optional[float]]]:
    """将数据集对象转为 (audio_path, ref_segments, duration) 元组。"""
    result = []
    for item in ds:
        audio_path = item.get("audio_path")
        ref_segments = item.get("segments", [])
        duration = item.get("duration")
        result.append((audio_path, ref_segments, duration))
    return result


def _load_from_local(
    dataset_path: Any,
) -> List[Tuple[Optional[str], List[Segment], Optional[float]]]:
    """从本地目录加载评测数据。

    期望目录结构::

        dataset_path/
          ├── audio/        (或直接放 .wav/.flac)
          │   ├── file1.wav
          │   └── file2.wav
          └── rttm/         (或 *.rttm 文件)
              ├── file1.rttm
              └── file2.rttm
    """
    from pathlib import Path
    from benchmark.llama_benchmark.metrics.speaker import parse_rttm

    root = Path(dataset_path)
    audio_dir = root / "audio" if (root / "audio").exists() else root
    rttm_dir = root / "rttm" if (root / "rttm").exists() else root

    samples = []
    for audio_file in sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.flac")):
        file_id = audio_file.stem
        rttm_file = rttm_dir / f"{file_id}.rttm"
        if not rttm_file.exists():
            logger.warning(f"找不到参考 RTTM: {rttm_file}")
            continue
        with open(rttm_file) as f:
            rttm_data = parse_rttm(f.read())
        ref_segs = rttm_data.get(file_id, [])
        if not ref_segs:
            # 尝试第一个键
            ref_segs = next(iter(rttm_data.values()), [])
        duration = max((e for _, e, _ in ref_segs), default=None) if ref_segs else None
        samples.append((str(audio_file), ref_segs, duration))

    return samples


def _builtin_samples() -> List[Tuple[Optional[str], List[Segment], Optional[float]]]:
    """内置虚拟样本（不需要真实音频，用于单元测试）。"""
    return [
        (
            None,  # 无音频路径
            [
                (0.0, 2.5, "SPK_A"),
                (2.8, 5.0, "SPK_B"),
                (5.3, 8.0, "SPK_A"),
                (8.2, 10.0, "SPK_B"),
            ],
            10.0,
        )
    ]
