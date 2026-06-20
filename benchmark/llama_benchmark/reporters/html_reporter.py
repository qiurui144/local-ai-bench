"""HTML 格式报告生成（Jinja2 模板 + Chart.js）。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.llama_benchmark.reporters.base_reporter import AbstractReporter

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.result import BenchmarkSuiteResult


class HtmlReporter(AbstractReporter):
    def generate(self, result: "BenchmarkSuiteResult") -> Path:
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
        except ImportError:
            raise ImportError("请安装 jinja2: pip install jinja2")

        result.compute_summary()
        template_dir = Path(__file__).parent.parent.parent / "templates"
        template_file = template_dir / "report.html.j2"

        if not template_file.exists():
            # 无模板时生成内置简单 HTML
            return self._generate_simple_html(result)

        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("report.html.j2")
        html_content = template.render(result=result, result_dict=result.to_dict())

        output_path = self._get_output_path(result.run_id, ".html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path

    def _generate_simple_html(self, result: "BenchmarkSuiteResult") -> Path:
        """生成不依赖模板文件的简单 HTML 报告。"""
        import json
        data = result.to_dict()
        s = result.summary

        rows = []
        for mr in result.model_results:
            for task in mr.task_results:
                for metric in task.metrics:
                    if metric.value is None:
                        continue
                    status_color = {
                        "pass": "#2ecc71",
                        "fail": "#e74c3c",
                        "error": "#e67e22",
                        "skip": "#95a5a6",
                    }.get(metric.status.value if hasattr(metric.status, "value") else str(metric.status), "#95a5a6")

                    rows.append(
                        f"<tr>"
                        f"<td>{mr.model_name}</td>"
                        f"<td>{task.task_name}</td>"
                        f"<td>{metric.name}</td>"
                        f"<td>{metric.value:.4f} {metric.unit}</td>"
                        f"<td style='color:{status_color};font-weight:bold'>"
                        f"{metric.status.value if hasattr(metric.status, 'value') else metric.status}"
                        f"</td>"
                        f"</tr>"
                    )

        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Benchmark Report {result.run_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
  .card {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #2c3e50; color: white; }}
  tr:hover {{ background: #f9f9f9; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
  .stat {{ background: #2c3e50; color: white; padding: 15px; border-radius: 6px; text-align: center; }}
  .stat-value {{ font-size: 2em; font-weight: bold; }}
  .stat-label {{ font-size: 0.8em; opacity: 0.8; }}
</style>
</head>
<body>
<div class="card">
  <h1>Benchmark Report</h1>
  <p><strong>Run ID:</strong> {result.run_id} &nbsp;&nbsp;
     <strong>Suite:</strong> {result.suite_name} &nbsp;&nbsp;
     <strong>Date:</strong> {result.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')} &nbsp;&nbsp;
     <strong>Duration:</strong> {result.duration_seconds:.1f}s</p>
</div>
<div class="card">
  <h2>Summary</h2>
  <div class="summary-grid">
    <div class="stat"><div class="stat-value">{s.get('total_models', 0)}</div><div class="stat-label">Models</div></div>
    <div class="stat"><div class="stat-value">{s.get('total_tasks', 0)}</div><div class="stat-label">Tasks</div></div>
    <div class="stat" style="background:#2ecc71"><div class="stat-value">{s.get('passed', 0)}</div><div class="stat-label">Passed</div></div>
    <div class="stat" style="background:#e74c3c"><div class="stat-value">{s.get('failed', 0)}</div><div class="stat-label">Failed</div></div>
    <div class="stat"><div class="stat-value">{s.get('pass_rate', 0):.1%}</div><div class="stat-label">Pass Rate</div></div>
  </div>
</div>
<div class="card">
  <h2>Detailed Results</h2>
  <table>
    <thead><tr><th>Model</th><th>Task</th><th>Metric</th><th>Value</th><th>Status</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>
<div class="card">
  <details><summary>Raw JSON</summary>
  <pre style="overflow:auto;max-height:400px">{json.dumps(data, indent=2, default=str, ensure_ascii=False)}</pre>
  </details>
</div>
</body>
</html>"""

        output_path = self._get_output_path(result.run_id, ".html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path
