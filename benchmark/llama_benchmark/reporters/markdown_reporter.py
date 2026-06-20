"""Markdown 格式摘要报告生成。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.llama_benchmark.core.result import BenchmarkStatus
from benchmark.llama_benchmark.reporters.base_reporter import AbstractReporter

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.result import BenchmarkSuiteResult, ModelBenchmarkResult

STATUS_ICON = {
    BenchmarkStatus.PASS: "PASS",
    BenchmarkStatus.FAIL: "FAIL",
    BenchmarkStatus.SKIP: "SKIP",
    BenchmarkStatus.ERROR: "ERROR",
}


class MarkdownReporter(AbstractReporter):
    def generate(self, result: "BenchmarkSuiteResult") -> Path:
        result.compute_summary()
        lines = self._build_report(result)
        output_path = self._get_output_path(result.run_id, "_summary.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return output_path

    def _build_report(self, result: "BenchmarkSuiteResult") -> list:
        s = result.summary
        lines = [
            "# Benchmark Report",
            "",
            f"**Run ID**: `{result.run_id}`  ",
            f"**Suite**: {result.suite_name}  ",
            f"**Start**: {result.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Duration**: {result.duration_seconds:.1f}s",
            "",
            "## Overall Results",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Models | {s.get('total_models', 0)} |",
            f"| Total Tasks | {s.get('total_tasks', 0)} |",
            f"| Passed | {s.get('passed', 0)} |",
            f"| Failed | {s.get('failed', 0)} |",
            f"| Errored | {s.get('errored', 0)} |",
            f"| Pass Rate | {s.get('pass_rate', 0):.1%} |",
            "",
            "## Model Results",
            "",
        ]

        for mr in result.model_results:
            lines.extend(self._format_model(mr))

        # 汇总失败项
        failed_tasks = [
            (mr.model_name, t)
            for mr in result.model_results
            for t in mr.task_results
            if t.status in (BenchmarkStatus.FAIL, BenchmarkStatus.ERROR)
        ]
        if failed_tasks:
            lines += ["", "## Failed / Error Tasks", ""]
            for model_name, task in failed_tasks:
                icon = STATUS_ICON[task.status]
                msg = task.error_message or ""
                lines.append(f"- `{model_name}` / `{task.task_name}` [{icon}] {msg}")

        return lines

    def _format_model(self, mr: "ModelBenchmarkResult") -> list:
        lines = [
            f"### {mr.model_name} ({mr.model_type})",
            "",
            f"Backend: `{mr.backend}` | Duration: {mr.duration_seconds:.1f}s",
            "",
        ]

        # ISA / 工具链信息（来自 system_info）
        sys_info = getattr(mr, "system_info", None) or {}
        isa = sys_info.get("isa", {})
        toolchain = sys_info.get("toolchain", {})
        if isa:
            lines += self._format_isa_section(isa)
        if toolchain:
            lines += self._format_toolchain_section(toolchain)

        lines += [
            "| Task | Metric | Value | Threshold | Status |",
            "|------|--------|-------|-----------|--------|",
        ]

        for task in mr.task_results:
            task_icon = STATUS_ICON.get(task.status, "?")
            if not task.metrics:
                lines.append(
                    f"| {task.task_name} | — | — | — | {task_icon} |"
                )
            for metric in task.metrics:
                if metric.value is None:
                    continue
                val_str = f"{metric.value:.4f}"
                thresh_str = (
                    f"≥{metric.threshold}" if metric.higher_is_better and metric.threshold
                    else f"≤{metric.threshold}" if not metric.higher_is_better and metric.threshold
                    else "—"
                )
                m_icon = STATUS_ICON.get(metric.status, "")
                lines.append(
                    f"| {task.task_name} | {metric.name} | "
                    f"{val_str} {metric.unit} | {thresh_str} | {m_icon} |"
                )

            if not task.metadata:
                continue

            # timing_breakdown（含 prefill/decode 分离 TPS）
            if task.metadata.get("timing_breakdown"):
                lines += self._format_timing_breakdown(task.task_name, task.metadata["timing_breakdown"])

            # 并发压测
            if task.metadata.get("stress_results"):
                lines += self._format_stress_results(task.task_name, task.metadata["stress_results"])

            # Context Scaling
            if task.metadata.get("scaling_data"):
                lines += self._format_context_scaling(task.metadata)

            # 持续负载
            if task.metadata.get("sustained_load"):
                lines += self._format_sustained_load(task.metadata["sustained_load"])

            # 瓶颈分类
            if task.metadata.get("bottleneck_report"):
                lines += self._format_bottleneck_report(task.metadata["bottleneck_report"])

        lines.append("")
        return lines

    def _format_isa_section(self, isa: dict) -> list:
        arch = isa.get("arch", "unknown")
        lines = ["", "**ISA / 内核状态**", ""]
        lines += ["| 项目 | 值 |", "|------|-----|"]

        lines.append(f"| 架构 | {arch} |")
        lines.append(f"| 内核版本 | {isa.get('kernel_version', '—')} |")

        if arch == "x86_64":
            lines.append(f"| AVX2 | {'✓' if isa.get('has_avx2') else '✗'} |")
            lines.append(f"| AVX-512F | {'✓' if isa.get('has_avx512f') else '✗'} |")
            lines.append(f"| FMA | {'✓' if isa.get('has_fma') else '✗'} |")
        elif "aarch64" in arch:
            lines.append(f"| NEON | {'✓' if isa.get('has_neon') else '✗'} |")
            lines.append(f"| SVE | {'✓' if isa.get('has_sve') else '✗'} |")
        elif "riscv" in arch:
            rvv_status = "✓" if isa.get("has_rvv") else "✗"
            vlen = isa.get("rvv_vlen")
            rvv_str = f"{rvv_status}" + (f" (VLEN={vlen}b)" if vlen else "")
            lines.append(f"| RVV | {rvv_str} |")
            lines.append(f"| RVV 版本 | {isa.get('rvv_spec_version', '—')} |")

        lines.append(f"| HugePages | {'启用' if isa.get('huge_pages_enabled') else '未启用'} |")
        lines.append(f"| NUMA 节点数 | {isa.get('numa_nodes', 1)} |")
        paranoid = isa.get("perf_event_paranoid", 3)
        lines.append(f"| perf_event_paranoid | {paranoid}{'（可采集）' if paranoid <= 1 else '（受限）'} |")

        if isa.get("ddr_channels"):
            lines.append(f"| DDR 通道 | {isa['ddr_channels']} |")
        if isa.get("ddr_speed_mts"):
            lines.append(f"| DDR 速度 | {isa['ddr_speed_mts']} MT/s |")
        lines.append("")
        return lines

    def _format_toolchain_section(self, tc: dict) -> list:
        lines = ["**工具链诊断**", ""]
        lines += ["| 层 | 项目 | 值 |", "|---|------|-----|"]
        lines.append(f"| L3 libc | 类型 | {tc.get('libc_type', '—')} {tc.get('libc_version', '')} |")
        lines.append(f"| L3 | OpenMP | {'可用' if tc.get('openmp_available') else '未检测到'} |")
        lines.append(f"| L4 BLAS | 后端 | {tc.get('blas_backend', 'none')} {tc.get('blas_version', '')} |")
        blas_isa = "✓" if tc.get("blas_isa_match") else "✗"
        kernels = ", ".join(tc.get("blas_isa_kernels", [])) or "—"
        lines.append(f"| L4 BLAS | ISA kernel 匹配 | {blas_isa} ({kernels}) |")
        lines.append(f"| L5 ggml | 后端 | {tc.get('ggml_backend', 'cpu')} |")
        lines.append(f"| L5 ggml | RVV kernel | {'✓' if tc.get('ggml_rvv_enabled') else '✗'} |")
        lines.append(f"| L5 ggml | AVX2 kernel | {'✓' if tc.get('ggml_avx2_enabled') else '✗'} |")
        lines.append(f"| L5 ggml | BLAS 启用 | {'✓' if tc.get('ggml_blas_enabled') else '✗'} |")

        warnings = tc.get("warnings", [])
        if warnings:
            lines += ["", "**工具链诊断警告**", ""]
            for w in warnings:
                lines.append(f"- ⚠️ {w}")
        lines.append("")
        return lines

    def _format_timing_breakdown(self, task_name: str, breakdown: dict) -> list:
        lines = ["", f"**{task_name} — 时序分解（Ollama 服务端）**", ""]
        lines += ["| 阶段 | 值 |", "|------|-----|"]
        label_map = {
            "model_load_ms": ("模型加载", "ms"),
            "prompt_eval_ms": ("Prompt 处理 (prefill)", "ms"),
            "token_gen_ms": ("Token 生成 (decode)", "ms"),
            "network_overhead_ms": ("网络/调度开销", "ms"),
            "prefill_tokens_per_second": ("Prefill TPS", "tok/s"),
            "decode_tokens_per_second": ("Decode TPS", "tok/s"),
            "prefill_decode_ratio": ("Prefill/Decode 比值", ""),
        }
        for key, (label, unit) in label_map.items():
            val = breakdown.get(key)
            if val is not None:
                lines.append(f"| {label} | {val:.2f} {unit} |")
        lines.append("")
        return lines

    def _format_stress_results(self, task_name: str, stress_results: dict) -> list:
        lines = [f"**{task_name} — 并发压测资源峰值**", ""]
        lines += [
            "| 并发度 | QPS | TPS | P95延迟(ms) | GPU util P95(%) |",
            "|--------|-----|-----|-------------|----------------|",
        ]
        for c_key, c_data in sorted(stress_results.items()):
            res = c_data.get("resource_summary", {})
            gpu_p95 = res.get("gpu_util_p95_percent", "—")
            gpu_str = f"{gpu_p95:.0f}" if isinstance(gpu_p95, float) else "—"
            lines.append(
                f"| {c_key} | {c_data['qps']:.2f} | {c_data['tps']:.2f} | "
                f"{c_data['p95_latency_ms']:.0f} | {gpu_str} |"
            )
        lines.append("")
        return lines

    def _format_context_scaling(self, metadata: dict) -> list:
        scaling_data = metadata.get("scaling_data", {})
        nonlinear = metadata.get("scaling_nonlinear", False)
        nonlinear_at = metadata.get("nonlinear_at_context")
        if not scaling_data:
            return []

        nonlinear_str = f"⚠️ 在 ctx={nonlinear_at} 出现超线性增长" if nonlinear else "线性增长（正常）"
        lines = ["", "**Context Length 扩展曲线**", ""]
        lines.append(f"缩放特性: {nonlinear_str}")
        lines += ["", "| Context Length | TTFT (ms) | Decode TPS (tok/s) |",
                  "|---------------|-----------|-------------------|"]
        for ctx_len in sorted(int(k) for k in scaling_data):
            d = scaling_data[ctx_len]
            lines.append(f"| {ctx_len} | {d['ttft_ms']:.1f} | {d['decode_tps']:.2f} |")
        lines.append("")
        return lines

    def _format_sustained_load(self, sustained: dict) -> list:
        tps_windows = sustained.get("tps_windows", [])
        degradation = sustained.get("tps_degradation_pct", 0)
        if not tps_windows:
            return []

        status_str = f"⚠️ TPS 衰减 {degradation:.1f}%（热降频）" if degradation > 20 else f"稳定（衰减 {degradation:.1f}%）"
        lines = ["", "**持续负载测试（热降频检测）**", ""]
        lines.append(f"状态: {status_str}")
        lines += ["", "| 时间窗口 | TPS (tok/s) |", "|---------|------------|"]
        window_s = sustained.get("window_s", 10)
        for i, tps in enumerate(tps_windows):
            t_start = i * window_s
            t_end = (i + 1) * window_s
            lines.append(f"| {t_start}s–{t_end}s | {tps:.2f} |")
        lines.append("")
        return lines

    def _format_bottleneck_report(self, report: dict) -> list:
        bound_type = report.get("bound_type", "unknown")
        confidence = report.get("confidence", "low")
        evidence = report.get("evidence", [])
        recommendations = report.get("recommendations", [])
        secondary = report.get("secondary_issues", [])

        lines = ["", "**瓶颈分析结论**", ""]
        lines.append(f"**主要瓶颈**: `{bound_type}` （置信度: {confidence}）")
        lines.append("")

        if evidence:
            lines.append("*支撑证据：*")
            for e in evidence:
                lines.append(f"- {e}")

        if recommendations:
            lines.append("")
            lines.append("*优化建议：*")
            for r in recommendations:
                lines.append(f"- {r}")

        if secondary:
            lines.append("")
            lines.append("*次要问题：*")
            for s in secondary:
                lines.append(f"- {s}")

        lines.append("")
        return lines
