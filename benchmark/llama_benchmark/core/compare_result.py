"""多硬件对比数据模型：加载多份 JSON 报告，生成横向对比结构。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HardwareConfig:
    hw_label: str
    cpu_model: str
    gpu_model: Optional[str]
    report_path: str
    platform: str = ""


@dataclass(frozen=True)
class MetricValue:
    value: float
    pass_fail: str   # "pass" | "fail" | "n/a"
    threshold: Optional[float]
    unit: str = ""
    higher_is_better: bool = True


@dataclass
class CompareReport:
    """N 份硬件报告对齐后的对比结构。

    metric_table[model_name][task_name][metric_name] = List[MetricValue]
    列表顺序与 hardware_configs 一一对应。
    """
    generated_at: str
    hardware_configs: List[HardwareConfig]
    # 三层嵌套：model → task → metric → [值按硬件顺序]
    metric_table: Dict[str, Dict[str, Dict[str, List[Optional[MetricValue]]]]]
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "hardware_configs": [
                {
                    "hw_label": h.hw_label,
                    "cpu_model": h.cpu_model,
                    "gpu_model": h.gpu_model,
                    "report_path": h.report_path,
                    "platform": h.platform,
                }
                for h in self.hardware_configs
            ],
            "metric_table": {
                model: {
                    task: {
                        metric: [
                            {
                                "value": mv.value,
                                "pass_fail": mv.pass_fail,
                                "threshold": mv.threshold,
                                "unit": mv.unit,
                                "higher_is_better": mv.higher_is_better,
                            }
                            if mv is not None else None
                            for mv in values
                        ]
                        for metric, values in tasks.items()
                    }
                    for task, tasks in models.items()
                }
                for model, models in self.metric_table.items()
            },
            "recommendations": self.recommendations,
        }


def load_compare_report(json_paths: List[str]) -> CompareReport:
    """加载多份 benchmark JSON 报告，对齐指标，生成 CompareReport。

    每份 JSON 对应一套硬件环境的测试结果。
    相同模型名 + 任务名 + 指标名的结果会被对齐到同一行。
    """
    from datetime import datetime, timezone

    reports_raw: List[Dict[str, Any]] = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            reports_raw.append(json.load(f))

    hardware_configs = [_extract_hw_config(r, p) for r, p in zip(reports_raw, json_paths)]

    # 收集所有 (model, task, metric) 的唯一组合
    all_keys: set = set()
    for report in reports_raw:
        for mr in report.get("model_results", []):
            model_name = mr.get("model_name", "")
            for task in mr.get("task_results", []):
                task_name = task.get("task_name", "")
                for m in task.get("metrics", []):
                    if m.get("value") is not None:
                        all_keys.add((model_name, task_name, m["name"]))

    # 构建三层 metric_table
    metric_table: Dict[str, Dict[str, Dict[str, List[Optional[MetricValue]]]]] = {}
    for model_name, task_name, metric_name in sorted(all_keys):
        if model_name not in metric_table:
            metric_table[model_name] = {}
        if task_name not in metric_table[model_name]:
            metric_table[model_name][task_name] = {}
        if metric_name not in metric_table[model_name][task_name]:
            metric_table[model_name][task_name][metric_name] = [None] * len(reports_raw)

    for hw_idx, report in enumerate(reports_raw):
        for mr in report.get("model_results", []):
            model_name = mr.get("model_name", "")
            for task in mr.get("task_results", []):
                task_name = task.get("task_name", "")
                for m in task.get("metrics", []):
                    if m.get("value") is None:
                        continue
                    metric_name = m["name"]
                    if model_name not in metric_table:
                        continue
                    if task_name not in metric_table[model_name]:
                        continue
                    if metric_name not in metric_table[model_name][task_name]:
                        continue
                    status = m.get("status", "n/a")
                    metric_table[model_name][task_name][metric_name][hw_idx] = MetricValue(
                        value=float(m["value"]),
                        pass_fail=status if isinstance(status, str) else str(status),
                        threshold=m.get("threshold"),
                        unit=m.get("unit", ""),
                        higher_is_better=m.get("higher_is_better", True),
                    )

    return CompareReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        hardware_configs=hardware_configs,
        metric_table=metric_table,
        recommendations=[],
    )


def _extract_hw_config(report: Dict[str, Any], path: str) -> HardwareConfig:
    """从单份报告中提取硬件配置信息。"""
    # 取第一个 model_result 的 system_info
    hw_label = "unknown"
    cpu_model = ""
    gpu_model = None
    plat = ""

    model_results = report.get("model_results", [])
    if model_results:
        sys_info = model_results[0].get("system_info", {})
        cpu_info = sys_info.get("cpu", {})
        gpu_list = sys_info.get("gpu", [])
        hw_label = sys_info.get("hw_label", "unknown")
        cpu_model = cpu_info.get("brand", "")
        plat = sys_info.get("platform", "")
        if gpu_list:
            gpu_model = gpu_list[0].get("name")

    # 如果 hw_label 仍是 unknown，从路径名推断
    if hw_label == "unknown":
        hw_label = Path(path).stem

    return HardwareConfig(
        hw_label=hw_label,
        cpu_model=cpu_model,
        gpu_model=gpu_model,
        report_path=str(path),
        platform=plat,
    )
