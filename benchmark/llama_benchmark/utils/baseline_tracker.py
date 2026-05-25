"""性能基线跟踪系统：保存测试结果、对比历史基线、生成趋势分析报告。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class RegressionAlert:
    """单个指标的回归/改善记录。"""

    metric: str
    current: float
    reference: float
    change_pct: float   # 负数表示退步，正数表示改善
    severity: str       # "critical"(>20%) / "warning"(>10%) / "info"(<10%)


@dataclass
class BaselineComparison:
    """当前运行与历史基线的完整对比结果。"""

    device: str
    current_timestamp: str
    reference_timestamp: str
    regressions: List[RegressionAlert] = field(default_factory=list)
    improvements: List[RegressionAlert] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    new_metrics: List[str] = field(default_factory=list)

    def has_regression(self) -> bool:
        """是否存在回归（任意严重程度）。"""
        return len(self.regressions) > 0

    def to_markdown(self) -> str:
        """生成 Markdown 格式的对比摘要。"""
        lines: List[str] = []
        lines.append(f"# 性能基线对比报告 — {self.device}")
        lines.append("")
        lines.append(f"- **当前运行**：{self.current_timestamp}")
        lines.append(f"- **参考基线**：{self.reference_timestamp}")
        lines.append("")

        # 回归表格
        if self.regressions:
            lines.append("## 回归指标")
            lines.append("")
            lines.append("| 指标 | 当前值 | 基线值 | 变化% | 严重程度 |")
            lines.append("|------|--------|--------|-------|----------|")
            for r in self.regressions:
                lines.append(
                    f"| {r.metric} | {r.current:.4g} | {r.reference:.4g} "
                    f"| {r.change_pct:+.1f}% | {r.severity} |"
                )
            lines.append("")
        else:
            lines.append("## 回归指标")
            lines.append("")
            lines.append("_无回归_")
            lines.append("")

        # 改善表格
        if self.improvements:
            lines.append("## 改善指标")
            lines.append("")
            lines.append("| 指标 | 当前值 | 基线值 | 变化% | 严重程度 |")
            lines.append("|------|--------|--------|-------|----------|")
            for r in self.improvements:
                lines.append(
                    f"| {r.metric} | {r.current:.4g} | {r.reference:.4g} "
                    f"| {r.change_pct:+.1f}% | {r.severity} |"
                )
            lines.append("")
        else:
            lines.append("## 改善指标")
            lines.append("")
            lines.append("_无改善_")
            lines.append("")

        # 新增指标
        if self.new_metrics:
            lines.append("## 新增指标（基线中不存在）")
            lines.append("")
            for m in self.new_metrics:
                lines.append(f"- {m}")
            lines.append("")

        # 持平指标
        if self.unchanged:
            lines.append("## 持平指标")
            lines.append("")
            for m in self.unchanged:
                lines.append(f"- {m}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def extract_summary_from_suite_result(suite_result_dict: dict, device_name: str) -> dict:
    """
    从 BenchmarkSuiteResult.to_dict() 中提取 summary.json 格式的关键指标。

    提取规则：
    - 遍历 model_results -> task_results -> metrics
    - timing_breakdown 中的 decode_tokens_per_second / prefill_tokens_per_second / ttft_ms
    - metric name: embed_throughput / accuracy / wer / ndcg_at_10
    - system_info 提取：arch、hw_label、isa.has_rvv（来自第一个 model_result）
    """
    metrics: Dict[str, float] = {}

    # 收集 system_info（取第一个 model_result 的 system_info）
    model_results = suite_result_dict.get("model_results", [])
    raw_sys = model_results[0].get("system_info", {}) if model_results else {}
    system_info: Dict[str, object] = {
        "arch": raw_sys.get("arch", ""),
        "hw_label": raw_sys.get("hw_label", ""),
        "isa_has_rvv": raw_sys.get("isa", {}).get("has_rvv", False),
    }

    # 遍历所有模型结果
    _NAMED_METRICS = {"embed_throughput", "accuracy", "wer", "ndcg_at_10"}

    for mr in model_results:
        model_name: str = mr.get("model_name", "unknown")
        for tr in mr.get("task_results", []):
            # 从 metadata.timing_breakdown 提取性能指标
            timing: dict = tr.get("metadata", {}).get("timing_breakdown", {})
            if timing:
                decode_tps = timing.get("decode_tokens_per_second")
                if decode_tps is not None:
                    metrics[f"{model_name}/decode_tps"] = float(decode_tps)

                prefill_tps = timing.get("prefill_tokens_per_second")
                if prefill_tps is not None:
                    metrics[f"{model_name}/prefill_tps"] = float(prefill_tps)

                # ttft_ms 取最小值（热身后）
                ttft = timing.get("ttft_ms")
                if ttft is not None:
                    key = f"{model_name}/ttft_ms"
                    ttft_val = float(ttft)
                    if key not in metrics or ttft_val < metrics[key]:
                        metrics[key] = ttft_val

            # 从 metrics 列表提取命名指标
            for m in tr.get("metrics", []):
                name: str = m.get("name", "")
                value = m.get("value")
                if value is None:
                    continue
                if name in _NAMED_METRICS:
                    mapped = "ndcg@10" if name == "ndcg_at_10" else name
                    metrics[f"{model_name}/{mapped}"] = float(value)

    # 构造 summary
    run_id: str = suite_result_dict.get("run_id", uuid.uuid4().hex[:8])
    start_time: str = suite_result_dict.get(
        "start_time", datetime.now(timezone.utc).isoformat()
    )
    # 将 ISO 时间戳规范化为 summary 格式（秒精度）
    try:
        ts = datetime.fromisoformat(start_time)
        timestamp = ts.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        timestamp = start_time

    return {
        "device": device_name,
        "timestamp": timestamp,
        "run_id": run_id,
        "system_info": system_info,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class BaselineTracker:
    """性能基线跟踪器：保存结果、加载历史、对比回归、生成趋势报告。"""

    _REGRESSION_CRITICAL = -20.0
    _REGRESSION_WARNING = -10.0

    def __init__(self, baselines_dir: str = "baselines") -> None:
        self.baselines_dir = Path(baselines_dir)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _device_dir(self, device_name: str) -> Path:
        return self.baselines_dir / device_name

    def _run_dir(self, device_name: str, run_ts: str) -> Path:
        return self._device_dir(device_name) / run_ts

    @staticmethod
    def _ts_to_dirname(timestamp: str) -> str:
        """将 ISO 时间戳转换为目录安全字符串（冒号替换为短横线）。"""
        return timestamp.replace(":", "-")

    @staticmethod
    def _dirname_to_ts(dirname: str) -> str:
        """将目录名还原为 ISO 时间戳（短横线还原为冒号，仅限时间部分）。"""
        # 格式：2026-04-13T21-49-27 -> 2026-04-13T21:49:27
        if "T" in dirname:
            date_part, time_part = dirname.split("T", 1)
            time_part = time_part.replace("-", ":")
            return f"{date_part}T{time_part}"
        return dirname

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """原子写入：先写临时文件，再 rename，防止中途崩溃留下损坏文件。"""
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)  # os.rename 在同一文件系统内是原子操作
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def save(self, suite_result_dict: dict, device_name: str) -> Path:
        """
        保存完整结果和摘要，更新 latest.txt，返回保存目录 Path。

        目录名格式：2026-04-13T21-49-27（冒号替换为短横线）
        所有写入均通过原子 tmp→rename 完成，防止中断后产生损坏文件。
        """
        summary = extract_summary_from_suite_result(suite_result_dict, device_name)
        timestamp = summary["timestamp"]
        run_dirname = self._ts_to_dirname(timestamp)

        run_dir = self._run_dir(device_name, run_dirname)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 原子写入完整结果
        self._atomic_write_text(
            run_dir / "result.json",
            json.dumps(suite_result_dict, ensure_ascii=False, indent=2),
        )

        # 原子写入摘要
        self._atomic_write_text(
            run_dir / "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2),
        )

        # 原子更新 latest.txt（latest.txt 损坏会导致 load_latest_summary 返回 None）
        self._atomic_write_text(
            self._device_dir(device_name) / "latest.txt",
            run_dirname,
        )

        return run_dir

    def load_latest_summary(self, device_name: str) -> Optional[dict]:
        """加载最新基线的 summary.json，不存在时返回 None。"""
        latest_path = self._device_dir(device_name) / "latest.txt"
        if not latest_path.exists():
            return None
        run_dirname = latest_path.read_text(encoding="utf-8").strip()
        summary_path = self._run_dir(device_name, run_dirname) / "summary.json"
        if not summary_path.exists():
            return None
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def compare(self, current_summary: dict, device_name: str) -> BaselineComparison:
        """
        当前摘要与最新历史基线对比，返回回归/改善报告。

        若不存在历史基线，返回仅含 new_metrics 的 BaselineComparison。
        """
        current_ts: str = current_summary.get("timestamp", "")
        reference_summary = self.load_latest_summary(device_name)

        if reference_summary is None:
            # 无历史基线：所有当前指标均视为新增
            new_metrics = list(current_summary.get("metrics", {}).keys())
            return BaselineComparison(
                device=device_name,
                current_timestamp=current_ts,
                reference_timestamp="",
                new_metrics=new_metrics,
            )

        reference_ts: str = reference_summary.get("timestamp", "")
        current_metrics: Dict[str, float] = current_summary.get("metrics", {})
        reference_metrics: Dict[str, float] = reference_summary.get("metrics", {})

        regressions: List[RegressionAlert] = []
        improvements: List[RegressionAlert] = []
        unchanged: List[str] = []
        new_metrics: List[str] = []

        for metric, current_val in current_metrics.items():
            if metric not in reference_metrics:
                new_metrics.append(metric)
                continue

            ref_val = reference_metrics[metric]
            if ref_val == 0.0:
                # 无法计算百分比变化
                unchanged.append(metric)
                continue

            change_pct = (current_val - ref_val) / abs(ref_val) * 100.0

            if change_pct < self._REGRESSION_CRITICAL:
                severity = "critical"
            elif change_pct < self._REGRESSION_WARNING:
                severity = "warning"
            elif change_pct < 0.0:
                severity = "info"
            else:
                severity = "info"

            alert = RegressionAlert(
                metric=metric,
                current=current_val,
                reference=ref_val,
                change_pct=change_pct,
                severity=severity,
            )

            if change_pct < 0.0:
                regressions.append(alert)
            elif change_pct > 0.0:
                improvements.append(alert)
            else:
                unchanged.append(metric)

        return BaselineComparison(
            device=device_name,
            current_timestamp=current_ts,
            reference_timestamp=reference_ts,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            new_metrics=new_metrics,
        )

    def list_runs(self, device_name: str) -> List[str]:
        """返回该设备所有历史运行目录名（时间戳），按时间倒序。"""
        device_dir = self._device_dir(device_name)
        if not device_dir.exists():
            return []
        runs = [
            d.name
            for d in device_dir.iterdir()
            if d.is_dir()
        ]
        # 目录名格式与时间排序兼容（字典序即时间序）
        runs.sort(reverse=True)
        return runs

    def generate_trend_markdown(self, device_name: str, last_n: int = 10) -> str:
        """
        生成趋势分析 Markdown 表格。

        每行一个历史运行，每列一个关键指标。
        对每个指标标注趋势：↑ 改善 / ↓ 退步 / — 持平（相对前一次）。
        """
        runs = self.list_runs(device_name)
        if not runs:
            return f"# 趋势分析 — {device_name}\n\n_暂无历史数据_\n"

        # 取最近 last_n 条（list_runs 已倒序，取前 last_n 后再正序显示）
        selected_runs = runs[:last_n]
        selected_runs_asc = list(reversed(selected_runs))

        # 加载所有摘要
        summaries: List[dict] = []
        for run_dirname in selected_runs_asc:
            summary_path = self._run_dir(device_name, run_dirname) / "summary.json"
            if summary_path.exists():
                summaries.append(
                    json.loads(summary_path.read_text(encoding="utf-8"))
                )

        if not summaries:
            return f"# 趋势分析 — {device_name}\n\n_暂无可读取的摘要数据_\n"

        # 收集所有指标名（保持稳定顺序）
        all_metrics: List[str] = []
        seen: set = set()
        for s in summaries:
            for k in s.get("metrics", {}):
                if k not in seen:
                    all_metrics.append(k)
                    seen.add(k)

        lines: List[str] = []
        lines.append(f"# 趋势分析 — {device_name}")
        lines.append("")
        lines.append(f"显示最近 {len(summaries)} 次运行（共 {len(runs)} 次）")
        lines.append("")

        if not all_metrics:
            lines.append("_无指标数据_")
            return "\n".join(lines) + "\n"

        # 表头
        header_cols = ["时间戳"] + all_metrics
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

        # 数据行
        prev_metrics: Optional[Dict[str, float]] = None
        for summary in summaries:
            ts = summary.get("timestamp", "")
            cur_metrics: Dict[str, float] = summary.get("metrics", {})
            row: List[str] = [ts]
            for metric in all_metrics:
                val = cur_metrics.get(metric)
                if val is None:
                    row.append("—")
                    continue
                cell = f"{val:.4g}"
                if prev_metrics is not None and metric in prev_metrics:
                    prev_val = prev_metrics[metric]
                    if prev_val != 0.0:
                        delta_pct = (val - prev_val) / abs(prev_val) * 100.0
                        if delta_pct > 1.0:
                            cell += " ↑"
                        elif delta_pct < -1.0:
                            cell += " ↓"
                        else:
                            cell += " —"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
            prev_metrics = dict(cur_metrics)

        lines.append("")
        lines.append("> ↑ 改善  ↓ 退步  — 持平（相对前一次，阈值 ±1%）")
        lines.append("")

        return "\n".join(lines) + "\n"
