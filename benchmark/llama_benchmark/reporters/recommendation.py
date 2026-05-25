"""硬件选型推荐引擎：基于 CompareReport 数据，输出 6 条规则驱动的推荐结论。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.compare_result import CompareReport, MetricValue


def generate_recommendations(report: "CompareReport") -> List[str]:
    """基于对比数据生成推荐结论列表（Markdown bullet 格式）。

    6 条规则：
    1. LLM 吞吐最高 → 推荐高并发推理
    2. TTFT 最低 + 生成效率最高 → 推荐实时对话
    3. Whisper WER 最低 → 推荐 ASR 场景
    4. RTF < 0.3 → 标注实时处理能力
    5. GPU util P95 > 85% → 标注 GPU 已饱和警告
    6. 性价比：throughput_tps / TDP → 推荐能效最优方案
    """
    recs: List[str] = []
    labels = [h.hw_label for h in report.hardware_configs]

    if len(labels) < 2:
        recs.append("需要至少两套硬件报告才能生成对比推荐。")
        return recs

    # 规则 1：LLM 吞吐最高
    winner, best_val = _find_best_hw(report, task_keyword="performance", metric_keyword="throughput_tps", higher=True)
    if winner is not None:
        recs.append(
            f"**高并发推理首选**：`{labels[winner]}` LLM 吞吐最高"
            f"（throughput_tps ≈ {best_val:.1f} tokens/s），适合批量/并发 LLM 推理场景。"
        )

    # 规则 2：TTFT 最低 → 实时对话
    winner_ttft, ttft_val = _find_best_hw(report, task_keyword="performance", metric_keyword="ttft_p50", higher=False)
    if winner_ttft is not None and ttft_val is not None:
        recs.append(
            f"**实时对话首选**：`{labels[winner_ttft]}` TTFT P50 最低"
            f"（≈ {ttft_val:.0f} ms），延迟最敏感的交互场景推荐此方案。"
        )

    # 规则 3：Whisper WER 最低
    winner_wer, wer_val = _find_best_hw(report, task_keyword="wer_cer", metric_keyword="wer", higher=False)
    if winner_wer is not None and wer_val is not None:
        recs.append(
            f"**ASR 精度首选**：`{labels[winner_wer]}` WER 最低"
            f"（{wer_val:.1%}），语音识别场景推荐此方案。"
        )

    # 规则 4：RTF < 0.3 实时能力标注
    for hw_idx, label in enumerate(labels):
        rtf_val = _get_metric_value(report, hw_idx, task_keyword="wer_cer", metric_keyword="rtf")
        if rtf_val is not None and rtf_val < 0.3:
            recs.append(
                f"**实时 ASR**：`{label}` RTF = {rtf_val:.3f} < 0.3，具备实时语音处理能力。"
            )

    # 规则 5：GPU 饱和警告
    for hw_idx, label in enumerate(labels):
        util_p95 = _get_metric_value(
            report, hw_idx, task_keyword="concurrency_stress", metric_keyword="gpu_util_p95"
        )
        if util_p95 is not None and util_p95 > 85:
            recs.append(
                f"**GPU 饱和警告**：`{label}` 并发压测期间 GPU 利用率 P95 = {util_p95:.0f}%，"
                "已接近饱和——继续增加并发不会提升吞吐，应考虑多卡或更高算力 GPU。"
            )

    # 规则 6：性价比（throughput_tps / TDP）
    perf_per_watt: List[Tuple[int, float]] = []
    for hw_idx, hw_cfg in enumerate(report.hardware_configs):
        tps_val = _get_metric_value(
            report, hw_idx, task_keyword="performance", metric_keyword="throughput_tps"
        )
        # TDP 存储在 system_info 里，通过 hw_label 无法直接获取，使用备注字段或跳过
        # 此处从 gpu_model 和全局 GPU 带宽表估算；生产中可从 report JSON 读取
        tdp = _estimate_tdp_from_label(hw_cfg.gpu_model or "")
        if tps_val is not None and tdp > 0:
            perf_per_watt.append((hw_idx, tps_val / tdp))

    if len(perf_per_watt) >= 2:
        perf_per_watt.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_score = perf_per_watt[0]
        recs.append(
            f"**能效最优**：`{labels[best_idx]}` 性价比得分最高"
            f"（throughput_tps / TDP ≈ {best_score:.3f}），适合电力受限场景。"
        )

    if not recs:
        recs.append("当前数据不足以生成推荐结论（指标缺失或仅有一套硬件报告）。")

    return recs


def _find_best_hw(
    report: "CompareReport",
    task_keyword: str,
    metric_keyword: str,
    higher: bool,
) -> Tuple[Optional[int], Optional[float]]:
    """在所有硬件中找到指定指标的最优硬件索引和值。"""
    candidates: List[Tuple[int, float]] = []

    for model_name, tasks in report.metric_table.items():
        for task_name, metrics in tasks.items():
            if task_keyword.lower() not in task_name.lower():
                continue
            for metric_name, values in metrics.items():
                if metric_keyword.lower() not in metric_name.lower():
                    continue
                for hw_idx, mv in enumerate(values):
                    if mv is not None:
                        candidates.append((hw_idx, mv.value))

    if not candidates:
        return None, None

    # 多个数值取平均
    hw_agg: Dict[int, List[float]] = {}
    for hw_idx, val in candidates:
        hw_agg.setdefault(hw_idx, []).append(val)

    hw_avg = {hw_idx: sum(vals) / len(vals) for hw_idx, vals in hw_agg.items()}
    best_idx = max(hw_avg, key=lambda i: hw_avg[i]) if higher else min(hw_avg, key=lambda i: hw_avg[i])
    return best_idx, hw_avg[best_idx]


def _get_metric_value(
    report: "CompareReport",
    hw_idx: int,
    task_keyword: str,
    metric_keyword: str,
) -> Optional[float]:
    """获取指定硬件的特定指标值（取最后匹配到的）。"""
    for model_name, tasks in report.metric_table.items():
        for task_name, metrics in tasks.items():
            if task_keyword.lower() not in task_name.lower():
                continue
            for metric_name, values in metrics.items():
                if metric_keyword.lower() not in metric_name.lower():
                    continue
                if hw_idx < len(values) and values[hw_idx] is not None:
                    return values[hw_idx].value
    return None


# 常见 GPU TDP 查表（W）
_GPU_TDP_TABLE: Dict[str, float] = {
    "H100 SXM": 700.0,
    "H100": 350.0,
    "A100 SXM": 400.0,
    "A100": 300.0,
    "A800": 400.0,
    "A40": 300.0,
    "A10G": 150.0,
    "A10": 150.0,
    "V100": 300.0,
    "T4": 70.0,
    "L40S": 350.0,
    "L4": 72.0,
    "RTX 4090": 450.0,
    "RTX 4080": 320.0,
    "RTX 3090": 350.0,
    "RTX 3080": 320.0,
    "MI300X": 750.0,
    "MI250X": 500.0,
}


def _estimate_tdp_from_label(gpu_model: str) -> float:
    if not gpu_model:
        return 0.0
    name_upper = gpu_model.upper()
    for key, tdp in _GPU_TDP_TABLE.items():
        if key.upper() in name_upper:
            return tdp
    return 0.0
