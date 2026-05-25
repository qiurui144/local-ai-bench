"""全链路瓶颈自动分类器：覆盖 L1-L7 共 11 条规则。

规则集覆盖：
  L7 GPU: memory_bound / compute_bound / network_bound / io_bound / thermal
  L7 通用: attention_quadratic / prefill_heavy
  L6: branch_misprediction / memory_latency_bound
  L5: toolchain_misconfigured（RVV/AVX 未启用）
  L4/CPU: ddr_bandwidth_bound
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BottleneckReport:
    bound_type: str                             # 主要瓶颈类型
    confidence: str                             # "high" | "medium" | "low"
    evidence: List[str] = field(default_factory=list)       # 支撑证据
    recommendations: List[str] = field(default_factory=list)  # 优化建议
    secondary_issues: List[str] = field(default_factory=list)  # 次要问题

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bound_type": self.bound_type,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "secondary_issues": self.secondary_issues,
        }


def classify_bottleneck(
    task_metadata: Dict[str, Any],
    system_info: Dict[str, Any],
    perf_snapshot: Optional[Any] = None,
    toolchain_profile: Optional[Any] = None,
    isa_profile: Optional[Any] = None,
    bandwidth_analysis: Optional[Dict[str, Any]] = None,
) -> BottleneckReport:
    """
    基于多层数据自动分类瓶颈，按优先级返回主要瓶颈和次要问题。

    参数：
      task_metadata:      performance/context_scaling/sustained_load 任务的 metadata
      system_info:        system_info.py 返回的系统信息字典
      perf_snapshot:      PerfSnapshot 对象（来自 system_profiler）
      toolchain_profile:  ToolchainProfile 对象（来自 toolchain_inspector）
      isa_profile:        ISAProfile 对象（来自 isa_detector）
      bandwidth_analysis: analyze_bandwidth_utilization() 返回的字典
    """
    issues: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 10 [L5]: 工具链配置错误（优先级最高，最容易修复）
    # ──────────────────────────────────────────────────────────────────────────
    if toolchain_profile is not None and isa_profile is not None:
        if (isa_profile.has_rvv and not toolchain_profile.ggml_rvv_enabled):
            issues.append({
                "type": "toolchain_misconfigured",
                "priority": 10,
                "confidence": "high",
                "evidence": "RVV 硬件可用但 llama.cpp 未编译 RVV kernel",
                "recommendation": (
                    'CMAKE_ARGS="-DGGML_RVV=on" pip install llama-cpp-python --no-cache-dir'
                ),
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 3 [L7]: 网络瓶颈
    # ──────────────────────────────────────────────────────────────────────────
    timing_breakdown = task_metadata.get("timing_breakdown", {})
    if timing_breakdown:
        network_ms = timing_breakdown.get("network_overhead_ms", 0)
        prompt_eval_ms = timing_breakdown.get("prompt_eval_ms", 1)
        if network_ms > prompt_eval_ms and network_ms > 50:
            issues.append({
                "type": "network_bound",
                "priority": 7,
                "confidence": "high",
                "evidence": f"网络/调度开销 {network_ms:.0f}ms > prompt_eval {prompt_eval_ms:.0f}ms",
                "recommendation": "将推理服务本地化部署或减少 HTTP RTT",
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 4 [L7]: IO 瓶颈（模型反复加载）
    # ──────────────────────────────────────────────────────────────────────────
    if timing_breakdown:
        model_load_ms = timing_breakdown.get("model_load_ms", 0)
        if model_load_ms > 3000:
            issues.append({
                "type": "io_bound",
                "priority": 6,
                "confidence": "medium",
                "evidence": f"模型加载耗时 {model_load_ms:.0f}ms > 3000ms（非冷启动仍如此表明反复卸载）",
                "recommendation": "增大 GPU 显存、调大 Ollama keep_alive 参数（如 OLLAMA_KEEP_ALIVE=1h）",
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 5 [L7]: 热降频（持续负载 TPS 衰减）
    # ──────────────────────────────────────────────────────────────────────────
    sustained = task_metadata.get("sustained_load", {})
    tps_degradation = sustained.get("tps_degradation_pct", 0)
    if tps_degradation > 20:
        issues.append({
            "type": "thermal",
            "priority": 8,
            "confidence": "high",
            "evidence": f"持续负载 TPS 衰减 {tps_degradation:.1f}% > 20%，疑似热降频",
            "recommendation": "改善散热设计、降低 TDP 功耗限制或增加风冷/水冷",
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 6 [L7]: Attention 二次方增长（context scaling）
    # ──────────────────────────────────────────────────────────────────────────
    if task_metadata.get("scaling_nonlinear"):
        nonlinear_at = task_metadata.get("nonlinear_at_context", "未知")
        issues.append({
            "type": "attention_quadratic",
            "priority": 5,
            "confidence": "high",
            "evidence": f"TTFT 在 context length={nonlinear_at} 出现超线性增长（O(n²)）",
            "recommendation": "启用 Flash Attention（Ollama 默认启用），或减少 context length 上限",
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 7 [L7]: Prefill 为主要瓶颈
    # ──────────────────────────────────────────────────────────────────────────
    if timing_breakdown:
        prompt_eval_ms = timing_breakdown.get("prompt_eval_ms", 0)
        token_gen_ms = timing_breakdown.get("token_gen_ms", 1)
        if token_gen_ms > 0 and prompt_eval_ms / token_gen_ms > 3.0:
            issues.append({
                "type": "prefill_heavy",
                "priority": 4,
                "confidence": "medium",
                "evidence": f"prefill/decode 比值 {prompt_eval_ms/token_gen_ms:.1f} > 3.0，prefill 占主导",
                "recommendation": "增大 batch size、考虑 speculative decoding 或 prefix caching",
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 1/11 [L7 GPU/CPU]: 内存带宽瓶颈
    # ──────────────────────────────────────────────────────────────────────────
    if bandwidth_analysis:
        bw_util = bandwidth_analysis.get("bandwidth_utilization_pct", 0)
        bw_source = bandwidth_analysis.get("bandwidth_source", "unknown")
        if bw_util and bw_util > 75:
            rule_type = "memory_bound" if bw_source == "gpu_hbm" else "ddr_bandwidth_bound"
            rec = (
                "升级高带宽 GPU（HBM3）/ 更低精度量化 / 增大 batch size"
                if bw_source == "gpu_hbm"
                else "降低量化精度（Q4→Q2）/ 增加 DDR 通道数 / 启用 NUMA 绑核"
            )
            issues.append({
                "type": rule_type,
                "priority": 9,
                "confidence": "high",
                "evidence": bandwidth_analysis.get("evidence", f"带宽利用率 {bw_util:.1f}%"),
                "recommendation": rec,
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 2 [L7 GPU]: 算力瓶颈
    # ──────────────────────────────────────────────────────────────────────────
    resource_summary = task_metadata.get("resource_summary", {})
    gpu_util_p95 = resource_summary.get("gpu_util_p95_percent")
    bw_util = (bandwidth_analysis or {}).get("bandwidth_utilization_pct", 100)
    if gpu_util_p95 and gpu_util_p95 > 85 and (not bw_util or bw_util < 50):
        issues.append({
            "type": "compute_bound",
            "priority": 7,
            "confidence": "medium",
            "evidence": f"GPU util P95 {gpu_util_p95:.0f}% > 85% 且带宽利用率较低",
            "recommendation": "升级更高 FLOP GPU、使用算子融合（Flash Attention / PagedAttention）",
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 8 [L6]: 分支预测失败（CPU perf）
    # ──────────────────────────────────────────────────────────────────────────
    if perf_snapshot is not None and perf_snapshot.available:
        if perf_snapshot.ipc is not None and perf_snapshot.ipc < 1.0:
            issues.append({
                "type": "branch_misprediction",
                "priority": 6,
                "confidence": "medium",
                "evidence": f"IPC={perf_snapshot.ipc:.2f} < 1.0，疑似分支预测失败或访存停顿",
                "recommendation": "使用 PGO（Profile-Guided Optimization）编译 llama.cpp；检查 NUMA 绑核",
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Rule 9 [L6]: L3 缓存命中率低
    # ──────────────────────────────────────────────────────────────────────────
    if perf_snapshot is not None and perf_snapshot.available:
        if perf_snapshot.l3_cache_miss_rate is not None and perf_snapshot.l3_cache_miss_rate > 5:
            issues.append({
                "type": "memory_latency_bound",
                "priority": 5,
                "confidence": "medium",
                "evidence": f"L3 cache miss rate {perf_snapshot.l3_cache_miss_rate:.1f}% > 5%",
                "recommendation": "NUMA 绑核（numactl --cpunodebind=0）、减少并发线程数、增大 L3 缓存",
            })

    # ──────────────────────────────────────────────────────────────────────────
    # 汇总：按优先级取主要瓶颈，其余作为次要问题
    # ──────────────────────────────────────────────────────────────────────────
    if not issues:
        return BottleneckReport(
            bound_type="no_bottleneck_detected",
            confidence="low",
            evidence=["未触发任何瓶颈规则，可能数据不足或性能良好"],
            recommendations=["建议启用 profiling.enabled=true 获取更完整数据"],
        )

    issues_sorted = sorted(issues, key=lambda x: x["priority"], reverse=True)
    primary = issues_sorted[0]
    secondary = issues_sorted[1:]

    return BottleneckReport(
        bound_type=primary["type"],
        confidence=primary["confidence"],
        evidence=[primary["evidence"]],
        recommendations=[primary["recommendation"]],
        secondary_issues=[f"{i['type']}: {i['evidence']}" for i in secondary],
    )
