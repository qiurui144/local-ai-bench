"""多硬件对比报告生成器：输出 JSON / HTML / Markdown 三种格式。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.compare_result import CompareReport


class CompareReporter:
    """接收 CompareReport，生成 compare_report.{json,html,md}。"""

    def __init__(self, output_dir: Path = Path("outputs/compare")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, report: "CompareReport") -> Dict[str, Path]:
        """生成全部格式，返回 {format: path} 字典。"""
        from benchmark.llama_benchmark.reporters.recommendation import generate_recommendations

        # 确保推荐已生成
        if not report.recommendations:
            report.recommendations = generate_recommendations(report)

        paths: Dict[str, Path] = {}
        paths["json"] = self._write_json(report)
        paths["md"] = self._write_markdown(report)
        paths["html"] = self._write_html(report)
        return paths

    def _write_json(self, report: "CompareReport") -> Path:
        path = self.output_dir / "compare_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        return path

    def _write_markdown(self, report: "CompareReport") -> Path:
        path = self.output_dir / "compare_report.md"
        lines = self._build_markdown(report)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    def _write_html(self, report: "CompareReport") -> Path:
        path = self.output_dir / "compare_report.html"
        template_path = Path(__file__).parent.parent.parent / "templates" / "compare_report.html.j2"

        if template_path.exists():
            try:
                from jinja2 import Environment, FileSystemLoader, select_autoescape
                env = Environment(
                    loader=FileSystemLoader(str(template_path.parent)),
                    autoescape=select_autoescape(["html"]),
                )
                # 注册 Jinja2 没有内置的 basename 过滤器
                env.filters["basename"] = lambda p: Path(p).name
                tmpl = env.get_template("compare_report.html.j2")
                content = tmpl.render(report=report, report_dict=report.to_dict())
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return path
            except Exception:
                pass

        # 内置简单 HTML（无模板文件时 fallback）
        content = self._build_simple_html(report)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _build_markdown(self, report: "CompareReport") -> List[str]:
        labels = [h.hw_label for h in report.hardware_configs]
        lines: List[str] = [
            "# 硬件对比报告",
            "",
            f"**生成时间**: {report.generated_at}",
            "",
            "## 硬件配置",
            "",
        ]

        # 硬件配置表
        header = "| # | 硬件标签 | CPU | GPU | 报告路径 |"
        sep = "|---|---------|-----|-----|---------|"
        lines += [header, sep]
        for i, hw in enumerate(report.hardware_configs):
            gpu = hw.gpu_model or "—"
            path = Path(hw.report_path).name
            lines.append(f"| {i+1} | `{hw.hw_label}` | {hw.cpu_model[:40]} | {gpu[:40]} | {path} |")

        lines += ["", "## 指标对比", ""]

        # 指标对比表
        for model_name, tasks in report.metric_table.items():
            lines.append(f"### 模型: {model_name}")
            lines.append("")

            header = "| 任务 | 指标 | " + " | ".join(f"`{lbl}`" for lbl in labels) + " |"
            sep = "|------|------|" + "------|" * len(labels)
            lines += [header, sep]

            for task_name, metrics in sorted(tasks.items()):
                for metric_name, values in sorted(metrics.items()):
                    row_cells = []
                    for mv in values:
                        if mv is None:
                            row_cells.append("—")
                        else:
                            indicator = "✓" if mv.pass_fail == "pass" else ("✗" if mv.pass_fail == "fail" else "")
                            row_cells.append(f"{mv.value:.4f} {mv.unit} {indicator}".strip())
                    lines.append(f"| {task_name} | {metric_name} | " + " | ".join(row_cells) + " |")

            lines.append("")

        # 推荐结论
        if report.recommendations:
            lines += ["## 推荐结论", ""]
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return lines

    def _build_simple_html(self, report: "CompareReport") -> str:
        """生成内置 HTML 对比报告（含 Chart.js 柱状图）。"""
        labels = [h.hw_label for h in report.hardware_configs]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

        # 收集供 Chart.js 使用的指标数据（取最重要的几个指标）
        chart_datasets = _extract_chart_data(report, labels, colors)
        chart_json = json.dumps(chart_datasets, ensure_ascii=False)

        rows_html = _build_table_rows(report, labels)
        recs_html = "".join(f"<li>{r}</li>" for r in report.recommendations)

        hw_rows = "".join(
            f"<tr><td>{i+1}</td><td><code>{h.hw_label}</code></td>"
            f"<td>{h.cpu_model}</td><td>{h.gpu_model or '—'}</td>"
            f"<td>{Path(h.report_path).name}</td></tr>"
            for i, h in enumerate(report.hardware_configs)
        )

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>硬件对比报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
  .card {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #2c3e50; color: white; }}
  tr:hover {{ background: #f9f9f9; }}
  .chart-container {{ max-width: 900px; margin: 0 auto; }}
  .rec-list {{ list-style: disc; padding-left: 1.5em; line-height: 1.8; }}
</style>
</head>
<body>
<div class="card">
  <h1>硬件对比报告</h1>
  <p><strong>生成时间：</strong>{report.generated_at}</p>
</div>

<div class="card">
  <h2>硬件配置</h2>
  <table>
    <thead><tr><th>#</th><th>硬件标签</th><th>CPU</th><th>GPU</th><th>报告文件</th></tr></thead>
    <tbody>{hw_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>核心指标对比</h2>
  <div class="chart-container">
    <canvas id="compareChart" height="120"></canvas>
  </div>
</div>

<div class="card">
  <h2>详细指标表</h2>
  <table>
    <thead>
      <tr>
        <th>任务</th><th>指标</th>
        {''.join(f'<th>{lbl}</th>' for lbl in labels)}
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>

<div class="card">
  <h2>推荐结论</h2>
  <ul class="rec-list">{recs_html}</ul>
</div>

<script>
const datasets = {chart_json};
const ctx = document.getElementById('compareChart').getContext('2d');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: datasets.labels,
    datasets: datasets.series,
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: 'top' }},
      title: {{ display: true, text: '核心指标横向对比（已归一化到 [0,1]）' }},
    }},
    scales: {{
      y: {{ beginAtZero: true, max: 1.0 }},
    }},
  }},
}});
</script>
</body>
</html>"""


def _extract_chart_data(
    report: "CompareReport",
    labels: List[str],
    colors: List[str],
) -> Dict[str, Any]:
    """提取关键指标用于 Chart.js（归一化到 [0,1]）。"""
    key_metrics = [
        ("performance", "throughput_tps", True),   # higher is better
        ("performance", "ttft_p50", False),         # lower is better
        ("wer_cer", "wer", False),
        ("wer_cer", "cer", False),
        ("retrieval", "ndcg_at_10", True),
    ]

    chart_labels: List[str] = []
    series_map: Dict[str, List[Optional[float]]] = {}

    for task_kw, metric_kw, higher in key_metrics:
        for model_name, tasks in report.metric_table.items():
            for task_name, metrics in tasks.items():
                if task_kw.lower() not in task_name.lower():
                    continue
                for metric_name, values in metrics.items():
                    if metric_kw.lower() not in metric_name.lower():
                        continue
                    raw_vals = [mv.value if mv else None for mv in values]
                    non_null = [v for v in raw_vals if v is not None]
                    if not non_null:
                        continue
                    min_v, max_v = min(non_null), max(non_null)
                    rng = max_v - min_v if max_v != min_v else 1.0

                    key_label = f"{task_name}/{metric_name}"
                    chart_labels.append(key_label)
                    normalized: List[Optional[float]] = []
                    for v in raw_vals:
                        if v is None:
                            normalized.append(None)
                        elif higher:
                            normalized.append(round((v - min_v) / rng, 3))
                        else:
                            normalized.append(round(1.0 - (v - min_v) / rng, 3))
                    series_map[key_label] = normalized

    # 转置：x 轴为硬件，系列为指标
    transposed_series = [
        {
            "label": lbl,
            "data": series_map[lbl],
            "backgroundColor": colors[j % len(colors)],
        }
        for j, lbl in enumerate(chart_labels)
    ]

    return {"labels": labels, "series": transposed_series}


def _build_table_rows(report: "CompareReport", labels: List[str]) -> str:
    rows = []
    for model_name, tasks in report.metric_table.items():
        for task_name, metrics in sorted(tasks.items()):
            for metric_name, values in sorted(metrics.items()):
                cells = []
                for mv in values:
                    if mv is None:
                        cells.append("<td>—</td>")
                    else:
                        color = (
                            "#2ecc71" if mv.pass_fail == "pass"
                            else "#e74c3c" if mv.pass_fail == "fail"
                            else "#95a5a6"
                        )
                        cells.append(
                            f"<td style='color:{color};font-weight:bold'>"
                            f"{mv.value:.4f} {mv.unit}</td>"
                        )
                rows.append(
                    f"<tr><td>{task_name}</td><td>{metric_name}</td>"
                    + "".join(cells)
                    + "</tr>"
                )
    return "".join(rows)
