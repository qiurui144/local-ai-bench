"""说话人分析指标：DER / EER / RTF。

DER（Diarization Error Rate）是说话人分离的标准评测指标：
  DER = (Missed Speech + False Alarm + Speaker Confusion) / 参考语音总时长

EER（Equal Error Rate）是说话人确认的标准指标：
  EER = FAR == FRR 时的错误率（越低越好）

RTF（Real Time Factor）衡量速度：
  RTF = 处理时间 / 音频时长（<1 表示快于实时）
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_metric import AbstractMetric


# ── 类型别名 ──────────────────────────────────────────────────────────────────
# 说话人分离片段：(起始秒, 结束秒, 说话人ID)
Segment = Tuple[float, float, str]


# ════════════════════════════════════════════════════════════════════════════
# DER — Diarization Error Rate
# ════════════════════════════════════════════════════════════════════════════

def compute_der(
    reference: List[Segment],
    hypothesis: List[Segment],
    collar: float = 0.25,
    skip_overlap: bool = False,
    resolution: float = 0.01,
) -> Dict[str, float]:
    """计算 DER 及各分量。

    Args:
        reference:    参考标注片段列表 [(start, end, speaker), ...]
        hypothesis:   模型输出片段列表
        collar:       边界忽略窗口（秒），NIST 标准为 0.25s
        skip_overlap: 是否跳过参考标注中的重叠语音区域
        resolution:   时间轴离散化精度（秒）

    Returns:
        {
            "der": float,                  # 总 DER（0–∞，越低越好）
            "missed_speech": float,        # 漏检率
            "false_alarm": float,          # 误报率
            "speaker_confusion": float,    # 说话人混淆率
            "total_ref_duration": float,   # 参考语音总时长（秒）
            "num_ref_speakers": int,       # 参考说话人数
            "num_hyp_speakers": int,       # 假设说话人数
        }

    优先使用 pyannote.metrics（工业标准）；不可用时使用内置 numpy 实现。
    """
    try:
        return _compute_der_pyannote(reference, hypothesis, collar, skip_overlap)
    except ImportError:
        return _compute_der_numpy(reference, hypothesis, collar, skip_overlap, resolution)


def _compute_der_pyannote(
    reference: List[Segment],
    hypothesis: List[Segment],
    collar: float,
    skip_overlap: bool,
) -> Dict[str, float]:
    """使用 pyannote.metrics 计算 DER（工业标准实现）。"""
    from pyannote.core import Annotation, Segment as PySegment
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_ann = Annotation()
    for start, end, spk in reference:
        ref_ann[PySegment(start, end)] = spk

    hyp_ann = Annotation()
    for start, end, spk in hypothesis:
        hyp_ann[PySegment(start, end)] = spk

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    detail = metric(ref_ann, hyp_ann, detailed=True)

    total = detail.get("total", 1.0) or 1.0
    return {
        "der": round(float(metric[ref_ann.uri or "ref"]), 4)
        if False else round(detail.get("diarization error rate", 0.0), 4),
        "missed_speech": round(detail.get("missed detection", 0.0) / total, 4),
        "false_alarm": round(detail.get("false alarm", 0.0) / total, 4),
        "speaker_confusion": round(detail.get("confusion", 0.0) / total, 4),
        "total_ref_duration": round(total, 2),
        "num_ref_speakers": len(set(s for _, _, s in reference)),
        "num_hyp_speakers": len(set(s for _, _, s in hypothesis)),
    }


def _compute_der_numpy(
    reference: List[Segment],
    hypothesis: List[Segment],
    collar: float,
    skip_overlap: bool,
    resolution: float,
) -> Dict[str, float]:
    """纯 numpy 实现 DER，不依赖 pyannote。

    算法：
    1. 时间轴离散化（默认 10ms 分辨率）
    2. 构建参考/假设说话人活动矩阵
    3. 应用 collar 掩码（排除边界帧）
    4. 匈牙利算法求最优说话人映射
    5. 计算各误差分量
    """
    from scipy.optimize import linear_sum_assignment

    if not reference:
        return {
            "der": 0.0, "missed_speech": 0.0, "false_alarm": 0.0,
            "speaker_confusion": 0.0, "total_ref_duration": 0.0,
            "num_ref_speakers": 0, "num_hyp_speakers": 0,
        }

    max_time = max(
        max(e for _, e, _ in reference),
        max((e for _, e, _ in hypothesis), default=0.0),
    )
    n_frames = int(math.ceil(max_time / resolution)) + 2

    ref_speakers = sorted(set(s for _, _, s in reference))
    hyp_speakers = sorted(set(s for _, _, s in hypothesis))
    ref_idx = {s: i for i, s in enumerate(ref_speakers)}
    hyp_idx = {s: i for i, s in enumerate(hyp_speakers)}

    ref_mat = np.zeros((n_frames, len(ref_speakers)), dtype=np.bool_)
    hyp_mat = np.zeros((n_frames, len(hyp_speakers)), dtype=np.bool_)

    for start, end, spk in reference:
        fs, fe = int(start / resolution), int(end / resolution)
        ref_mat[fs:fe, ref_idx[spk]] = True

    for start, end, spk in hypothesis:
        fs, fe = int(start / resolution), int(end / resolution)
        hyp_mat[fs:fe, hyp_idx[spk]] = True

    # ── collar 掩码：排除参考片段边界附近的帧 ─────────────────────────────
    collar_frames = int(collar / resolution)
    keep = np.ones(n_frames, dtype=np.bool_)
    for start, end, _ in reference:
        fs, fe = int(start / resolution), int(end / resolution)
        keep[max(0, fs - collar_frames): fs + collar_frames + 1] = False
        keep[max(0, fe - collar_frames): fe + collar_frames + 1] = False

    # ── skip_overlap：排除参考中有多个说话人同时活跃的帧 ──────────────────
    if skip_overlap:
        overlap_frames = ref_mat.sum(axis=1) > 1
        keep &= ~overlap_frames

    ref_k = ref_mat[keep]   # (kept_frames, n_ref_spk)
    hyp_k = hyp_mat[keep]   # (kept_frames, n_hyp_spk)

    ref_any = ref_k.any(axis=1)
    hyp_any = hyp_k.any(axis=1)

    total_ref_frames = int(ref_any.sum())
    if total_ref_frames == 0:
        return {
            "der": 0.0, "missed_speech": 0.0, "false_alarm": 0.0,
            "speaker_confusion": 0.0,
            "total_ref_duration": round(sum(e - s for s, e, _ in reference), 2),
            "num_ref_speakers": len(ref_speakers),
            "num_hyp_speakers": len(hyp_speakers),
        }

    missed_frames = int((ref_any & ~hyp_any).sum())
    fa_frames = int((hyp_any & ~ref_any).sum())

    # ── 匈牙利映射：最大化参考说话人与假设说话人的重叠帧数 ──────────────
    sc_frames = 0
    if ref_speakers and hyp_speakers:
        # overlap[i, j] = ref_speaker_i 和 hyp_speaker_j 同时活跃的帧数
        overlap = (ref_k.T.astype(np.int32)) @ (hyp_k.astype(np.int32))
        row_ind, col_ind = linear_sum_assignment(-overlap)
        correct_mapped = int(overlap[row_ind, col_ind].sum())
        # 两者均活跃但未被正确映射的帧 = 说话人混淆
        both_active = int((ref_any & hyp_any).sum())
        sc_frames = max(0, both_active - correct_mapped)

    total_ref = total_ref_frames * resolution
    der = (missed_frames + fa_frames + sc_frames) * resolution / total_ref

    return {
        "der": round(der, 4),
        "missed_speech": round(missed_frames * resolution / total_ref, 4),
        "false_alarm": round(fa_frames * resolution / total_ref, 4),
        "speaker_confusion": round(sc_frames * resolution / total_ref, 4),
        "total_ref_duration": round(total_ref, 2),
        "num_ref_speakers": len(ref_speakers),
        "num_hyp_speakers": len(hyp_speakers),
    }


class DERMetric(AbstractMetric):
    """Diarization Error Rate（说话人分离错误率）。"""

    name = "der"
    unit = ""
    higher_is_better = False

    def __init__(self, collar: float = 0.25, skip_overlap: bool = False) -> None:
        self.collar = collar
        self.skip_overlap = skip_overlap

    def compute(
        self,
        predictions: List[List[Segment]],
        references: List[List[Segment]],
        **kwargs,
    ) -> float:
        """计算多个音频文件的宏平均 DER。"""
        if not references:
            return 0.0
        ders = []
        for hyp, ref in zip(predictions, references):
            result = compute_der(ref, hyp, self.collar, self.skip_overlap)
            ders.append(result["der"])
        return float(np.mean(ders))


# ════════════════════════════════════════════════════════════════════════════
# EER — Equal Error Rate（说话人确认）
# ════════════════════════════════════════════════════════════════════════════

def compute_eer(
    scores: List[float],
    labels: List[int],
) -> Dict[str, float]:
    """计算 EER 和 minDCF。

    Args:
        scores: 相似度分数列表（越高越相似）
        labels: 标签列表（1=同一说话人，0=不同说话人）

    Returns:
        {"eer": float, "eer_threshold": float, "min_dcf": float}
    """
    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    # 按阈值从高到低扫描
    thresholds = np.sort(np.unique(scores_arr))[::-1]
    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"eer": 0.0, "eer_threshold": 0.5, "min_dcf": 0.0}

    best_eer = 1.0
    best_threshold = 0.0
    min_dcf = 1.0

    for thr in thresholds:
        accept = scores_arr >= thr
        fa_rate = float((accept & (labels_arr == 0)).sum()) / n_neg  # FAR
        fr_rate = float((~accept & (labels_arr == 1)).sum()) / n_pos  # FRR

        # EER：FAR 和 FRR 最接近的点
        if abs(fa_rate - fr_rate) < abs(best_eer - (fa_rate + fr_rate) / 2):
            best_eer = (fa_rate + fr_rate) / 2
            best_threshold = float(thr)

        # minDCF（NIST 2008 标准，p_target=0.01, c_miss=1, c_fa=1）
        p_target = 0.01
        dcf = p_target * fr_rate + (1 - p_target) * fa_rate
        if dcf < min_dcf:
            min_dcf = dcf

    return {
        "eer": round(best_eer, 4),
        "eer_threshold": round(best_threshold, 4),
        "min_dcf": round(min_dcf, 4),
    }


class EERMetric(AbstractMetric):
    """Equal Error Rate（说话人确认等错误率）。"""

    name = "eer"
    unit = ""
    higher_is_better = False

    def compute(
        self,
        predictions: List[float],
        references: List[int],
        **kwargs,
    ) -> float:
        return compute_eer(predictions, references)["eer"]


# ════════════════════════════════════════════════════════════════════════════
# RTF — Real Time Factor（实时倍率）
# ════════════════════════════════════════════════════════════════════════════

def compute_rtf(processing_seconds: float, audio_duration_seconds: float) -> float:
    """计算 RTF = 处理时间 / 音频时长。

    RTF < 1：快于实时（如 RTF=0.1 表示 10× 快于实时）。
    RTF = 1：实时。
    RTF > 1：慢于实时（无法支持实时应用）。
    """
    if audio_duration_seconds <= 0:
        return 0.0
    return round(processing_seconds / audio_duration_seconds, 4)


# ── RTTM 格式解析工具 ────────────────────────────────────────────────────────

def parse_rttm(rttm_content: str) -> Dict[str, List[Segment]]:
    """解析 RTTM 文件内容，返回 {file_id: [(start, end, speaker), ...]}。

    RTTM 格式：
        SPEAKER file_id 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
    """
    result: Dict[str, List[Segment]] = {}
    for line in rttm_content.strip().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        file_id = parts[1]
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        end = start + duration
        result.setdefault(file_id, []).append((start, end, speaker))
    return result


def segments_to_rttm(file_id: str, segments: List[Segment]) -> str:
    """将片段列表转换为 RTTM 格式字符串。"""
    lines = []
    for start, end, speaker in sorted(segments, key=lambda x: x[0]):
        duration = end - start
        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
    return "\n".join(lines)
