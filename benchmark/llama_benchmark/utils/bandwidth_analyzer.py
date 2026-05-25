"""内存带宽利用率分析：GPU HBM 和 CPU DDR 双路径 Roofline 模型。

覆盖 L7 层分析：判断 decode 阶段是否达到内存带宽瓶颈。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.llama_benchmark.utils.isa_detector import ISAProfile


def estimate_memory_bound_tps(
    bandwidth_gbps: float,
    model_params_billions: float,
    bits_per_weight: int = 4,
) -> float:
    """
    估算 memory-bandwidth-bound decode 的理论 TPS 上限（Roofline 模型）。

    LLM decode 阶段每生成一个 token 需要读取一遍模型权重：
      理论上限 (tok/s) = bandwidth (GB/s) / weight_size (GB/token)
      weight_size = params × bytes_per_param / (1024³)
    """
    if bandwidth_gbps <= 0 or model_params_billions <= 0:
        return 0.0
    bytes_per_param = bits_per_weight / 8.0
    weight_size_gb = model_params_billions * 1e9 * bytes_per_param / (1024 ** 3)
    if weight_size_gb <= 0:
        return 0.0
    return round(bandwidth_gbps / weight_size_gb, 2)


def get_cpu_ddr_bandwidth_gbps(isa_profile: "ISAProfile") -> Optional[float]:
    """
    从 ISAProfile 估算 CPU 内存带宽（GB/s）。

    公式：channels × speed_mts (MT/s) × 8 bytes / 1000
    每条 DDR 通道 bus width = 64 bit = 8 bytes；speed_mts 是每秒传输次数。
    """
    if isa_profile.ddr_channels and isa_profile.ddr_speed_mts:
        bw = isa_profile.ddr_channels * isa_profile.ddr_speed_mts * 8 / 1000
        return round(bw, 1)
    return None


def analyze_bandwidth_utilization(
    timing_breakdown: Dict[str, Any],
    decode_tps: float,
    system_info: Dict[str, Any],
    isa_profile: Optional["ISAProfile"] = None,
) -> Dict[str, Any]:
    """
    分析内存带宽利用率，返回理论上限和实际利用率。

    自动选择路径：
    - GPU 路径：system_info["gpu"][0]["memory_bandwidth_gbps"]
    - CPU DDR 路径：isa_profile.ddr_channels × ddr_speed_mts（优先级次于 GPU）
    """
    result: Dict[str, Any] = {
        "decode_tps_actual": decode_tps,
        "bandwidth_source": "unknown",
        "bandwidth_gbps": None,
        "theoretical_max_decode_tps": None,
        "bandwidth_utilization_pct": None,
        "model_params_billions": None,
        "bound_type": "unknown",
        "evidence": "",
    }

    # GPU 带宽优先
    gpu_bandwidth: Optional[float] = None
    gpus = system_info.get("gpu", [])
    if gpus and gpus[0].get("memory_bandwidth_gbps"):
        gpu_bandwidth = float(gpus[0]["memory_bandwidth_gbps"])

    cpu_ddr_bandwidth: Optional[float] = None
    if isa_profile is not None:
        cpu_ddr_bandwidth = get_cpu_ddr_bandwidth_gbps(isa_profile)

    if gpu_bandwidth:
        bandwidth_gbps = gpu_bandwidth
        result["bandwidth_source"] = "gpu_hbm"
    elif cpu_ddr_bandwidth:
        bandwidth_gbps = cpu_ddr_bandwidth
        result["bandwidth_source"] = "cpu_ddr"
    else:
        result["evidence"] = "无法获取内存带宽信息（无 GPU 带宽查表且无 DDR 通道数据）"
        return result

    result["bandwidth_gbps"] = bandwidth_gbps

    model_params = _estimate_model_params(timing_breakdown, system_info)
    if model_params is None or model_params <= 0:
        result["evidence"] = "无法推断模型参数量"
        return result

    result["model_params_billions"] = model_params

    theoretical_tps = estimate_memory_bound_tps(bandwidth_gbps, model_params)
    result["theoretical_max_decode_tps"] = theoretical_tps

    if theoretical_tps > 0 and decode_tps > 0:
        utilization = min(decode_tps / theoretical_tps * 100, 100.0)
        result["bandwidth_utilization_pct"] = round(utilization, 1)

        if utilization > 75:
            result["bound_type"] = "memory_bound"
            result["evidence"] = (
                f"带宽利用率 {utilization:.1f}% > 75%，"
                f"理论上限 {theoretical_tps:.1f} tok/s，实际 {decode_tps:.1f} tok/s，"
                f"内存带宽为主要瓶颈"
            )
        elif utilization < 30:
            result["bound_type"] = "compute_or_other_bound"
            result["evidence"] = (
                f"带宽利用率仅 {utilization:.1f}% < 30%，"
                f"瓶颈可能在算力或其他因素（非带宽受限）"
            )
        else:
            result["bound_type"] = "mixed"
            result["evidence"] = f"带宽利用率 {utilization:.1f}%，介于带宽和算力之间"

    return result


def _estimate_model_params(
    timing_breakdown: Dict[str, Any],
    system_info: Dict[str, Any],
) -> Optional[float]:
    """
    推断模型参数量（Billions）。

    优先顺序：timing_breakdown 显式字段 → GPU VRAM 启发式
    """
    params = timing_breakdown.get("model_params_billions")
    if params:
        return float(params)

    # GPU VRAM 启发式：Q4 量化约 0.5 GB/B params
    gpus = system_info.get("gpu", [])
    if gpus:
        vram_gb = gpus[0].get("memory_total_gb", 0)
        if vram_gb > 0:
            return round(vram_gb / 0.5 * 0.7, 1)

    return None
