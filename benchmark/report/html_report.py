"""benchmark/report/html_report.py
自包含 HTML 报告生成器。支持单模型报告和 compare 报告。
不引入新 Python 依赖（仅 stdlib）。CDN 失败时降级为纯文本 ASCII 表。
"""
from __future__ import annotations

import html
import json

_CHART_JS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"

_VERDICT_COLOR = {
    "PASS": "#22c55e",
    "WARN": "#f59e0b",
    "FAIL": "#ef4444",
    "BLOCKED": "#94a3b8",
    "SKIPPED": "#94a3b8",
    "REPLACEABLE": "#22c55e",
    "NOT_REPLACEABLE": "#ef4444",
    "INCONCLUSIVE": "#f59e0b",
}

_RADAR_LABELS = [
    "accuracy", "translation", "embedding", "rerank", "asr",
    "general_ability", "conditioned", "scenarios", "conversation_drift",
]

_VERDICT_SCORE = {"PASS": 1.0, "WARN": 0.6, "FAIL": 0.2, "BLOCKED": 0.0, "SKIPPED": 0.0}


def _verdict_color(verdict: str) -> str:
    return _VERDICT_COLOR.get(str(verdict).upper(), "#94a3b8")


def _v2s(verdict: str) -> float:
    return _VERDICT_SCORE.get(str(verdict).upper(), 0.0)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _quality_radar_data(report: dict) -> dict:
    """9 个质量轴的雷达图数据；有实际数值时用实际值，否则用 verdict 映射。"""
    bm = report.get("benchmarks", {}) or {}
    values: list[float] = []

    for dim in _RADAR_LABELS:
        block = bm.get(dim) or {}
        verdict = block.get("verdict", "SKIPPED")
        val: float | None = None

        if dim == "accuracy":
            v = (block.get("aggregate") or {}).get("category_precision")
            if v is not None:
                val = _clamp(v)
        elif dim == "translation":
            # bleu 通常 0–100，归一化到 [0,1]
            bleu = block.get("bleu")
            if bleu is None:
                for d_block in (block.get("directions") or {}).values():
                    for t_block in d_block.values():
                        b = (t_block.get("aggregate") or {}).get("bleu")
                        if b is not None:
                            bleu = b
                            break
                    if bleu is not None:
                        break
            if bleu is not None:
                val = _clamp(bleu / 100.0 if bleu > 1.0 else bleu)
        elif dim == "embedding":
            agg = block.get("aggregate") or {}
            r1 = agg.get("recall@1") or agg.get("mrr")
            if r1 is not None:
                val = _clamp(r1)
        elif dim == "rerank":
            agg = block.get("aggregate") or {}
            ndcg = agg.get("ndcg@10") or agg.get("mrr")
            if ndcg is not None:
                val = _clamp(ndcg)
        elif dim == "asr":
            agg = block.get("aggregate") or {}
            cer = agg.get("cer")
            if cer is not None:
                val = _clamp(1.0 - cer)
        elif dim == "general_ability":
            tasks = block.get("tasks") or {}
            accs = [t.get("accuracy", 0) for t in tasks.values()
                    if t.get("accuracy") is not None]
            if accs:
                val = _clamp(sum(accs) / len(accs))
        elif dim == "conditioned":
            # 无明确数值主指标，用 verdict
            pass
        elif dim == "scenarios":
            sc = block.get("scenarios") or {}
            scores: list[float] = []
            for sb in sc.values():
                l1 = sb.get("l1") or {}
                if l1:
                    scores.append(list(l1.values())[0])
            if scores:
                val = _clamp(sum(scores) / len(scores))
        elif dim == "conversation_drift":
            ps = block.get("per_scenario") or {}
            drops = [s.get("max_quality_drop") for s in ps.values()
                     if s.get("max_quality_drop") is not None]
            if drops:
                val = _clamp(1.0 - max(drops))
            else:
                drop = block.get("max_quality_drop")
                if drop is not None:
                    val = _clamp(1.0 - drop)

        values.append(val if val is not None else _v2s(verdict))

    return {"labels": _RADAR_LABELS, "values": values}


def _ascii_table(report: dict) -> str:
    """断网降级用的 ASCII 纯文本摘要。"""
    bm = report.get("benchmarks", {}) or {}
    rows = [("Dimension", "Verdict")]
    for dim in _RADAR_LABELS:
        block = bm.get(dim) or {}
        rows.append((dim, block.get("verdict", "—")))
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    sep = f"+{'-' * (w0 + 2)}+{'-' * (w1 + 2)}+"
    lines = [sep]
    for i, (a, b) in enumerate(rows):
        lines.append(f"| {a:<{w0}} | {b:<{w1}} |")
        if i == 0:
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)


def _perf_table_html(bm: dict) -> str:
    """性能指标表格（TTFT / 吞吐量 / 并发）。"""
    rows: list[tuple[str, str]] = []
    ttft = bm.get("ttft") or {}
    if ttft:
        s = ttft.get("ttft_ms_stats") or {}
        rows.append(("TTFT P50 (ms)", f"{s.get('p50', 0):.0f}"))
        rows.append(("TTFT P95 (ms)", f"{s.get('p95', 0):.0f}"))
        rows.append(("TTFT 错误率", f"{ttft.get('error_rate', 0) * 100:.1f}%"))
    tp = bm.get("throughput") or {}
    if tp:
        rows.append(("聚合 TPS", f"{tp.get('aggregate_tps', 0):.1f}"))
    con = bm.get("concurrency") or {}
    for step in (con.get("steps") or []):
        c = step.get("concurrency", "?")
        rows.append((f"并发{c} 成功率", f"{step.get('success_rate', 0) * 100:.1f}%"))
        rows.append((f"并发{c} TPS", f"{step.get('aggregate_tps', 0):.1f}"))
    if not rows:
        return "<p>无性能数据</p>"
    cells = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    return (
        "<table class='perf'><thead><tr><th>指标</th><th>值</th></tr></thead>"
        f"<tbody>{cells}</tbody></table>"
    )


def _scenarios_bar_data(bm: dict) -> dict:
    """Scenarios 场景逐一 L1 分值。"""
    sc = (bm.get("scenarios") or {}).get("scenarios") or {}
    labels, values, colors = [], [], []
    for name, blk in sc.items():
        l1 = blk.get("l1") or {}
        score = list(l1.values())[0] if l1 else _v2s(blk.get("verdict", "SKIPPED"))
        labels.append(name)
        values.append(round(score, 4))
        colors.append(_verdict_color(blk.get("verdict", "SKIPPED")))
    return {"labels": labels, "values": values, "colors": colors}


def _drift_line_data(bm: dict) -> dict | None:
    """Conversation drift 折线图数据（如存在）。"""
    cd = bm.get("conversation_drift") or {}
    ps = cd.get("per_scenario") or {}
    if not ps:
        return None
    positions: list[int] = sorted({
        int(p) for s in ps.values()
        for p in (s.get("quality_by_position") or {}).keys()
    })
    if not positions:
        return None
    datasets = []
    for name, blk in ps.items():
        qbp = blk.get("quality_by_position") or {}
        pts = [qbp.get(str(p)) for p in positions]
        datasets.append({"label": name, "data": pts})
    return {"positions": positions, "datasets": datasets}


# ── HTML 模板 ─────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:system-ui,sans-serif;margin:0;padding:20px;background:#f8fafc;color:#1e293b}
h1{font-size:1.5rem;margin-bottom:.25rem}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;color:#fff;font-size:.9rem}
.section{background:#fff;border-radius:8px;padding:16px 20px;margin:16px 0;box-shadow:0 1px 3px #0001}
h2{font-size:1.1rem;margin:.5rem 0 1rem}
canvas{max-height:320px}
table.perf{border-collapse:collapse;width:100%}
table.perf th,table.perf td{text-align:left;padding:6px 10px;border-bottom:1px solid #e2e8f0}
table.perf th{background:#f1f5f9;font-weight:600}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.charts{grid-template-columns:1fr}}
.compare-verdict{font-size:1.2rem;font-weight:700;margin:8px 0}
pre.fallback{background:#f1f5f9;padding:12px;border-radius:6px;font-size:.8rem;overflow-x:auto}
"""


def _render_single(report: dict) -> str:
    model = html.escape(str(report.get("model", "unknown")))
    verdict = str(report.get("verdict") or "")
    bm = report.get("benchmarks") or {}

    radar = _quality_radar_data(report)
    bar = _scenarios_bar_data(bm)
    drift = _drift_line_data(bm)

    radar_json = json.dumps(radar["values"])
    radar_labels = json.dumps(radar["labels"])
    bar_json = json.dumps(bar["values"])
    bar_labels = json.dumps(bar["labels"])
    bar_colors = json.dumps(bar["colors"])

    drift_block = ""
    if drift:
        drift_json = json.dumps(drift)
        drift_block = f"""
<div class="section">
  <h2>Conversation Drift 折线图</h2>
  <canvas id="driftChart"></canvas>
  <noscript><pre class="fallback">{html.escape(_ascii_table(report))}</pre></noscript>
</div>
<script>
(function(){{
  var dd={drift_json};
  new Chart(document.getElementById('driftChart'),{{
    type:'line',
    data:{{
      labels:dd.positions,
      datasets:dd.datasets.map(function(d){{return{{label:d.label,data:d.data,fill:false,tension:.3}}}})
    }},
    options:{{responsive:true,plugins:{{legend:{{position:'bottom'}}}}}}
  }});
}})();
</script>"""

    badge_color = _verdict_color(verdict)
    badge_html = (f'<span class="badge" style="background:{badge_color}">'
                  f'{html.escape(verdict)}</span>') if verdict else ""

    ts = html.escape(str(report.get("timestamp", "")))
    hw = report.get("hardware_profile") or {}
    gpu = html.escape(str(hw.get("gpu", "—")))
    hv = html.escape(str(report.get("harness_version", "—")))

    bar_section = ""
    if bar["labels"]:
        bar_section = f"""
<div class="section">
  <h2>Scenarios L1 分值</h2>
  <canvas id="barChart"></canvas>
  <noscript><pre class="fallback">{html.escape(_ascii_table(report))}</pre></noscript>
</div>
<script>
(function(){{
  new Chart(document.getElementById('barChart'),{{
    type:'bar',
    data:{{
      labels:{bar_labels},
      datasets:[{{label:'L1 score',data:{bar_json},backgroundColor:{bar_colors}}}]
    }},
    options:{{responsive:true,scales:{{y:{{min:0,max:1}}}},plugins:{{legend:{{display:false}}}}}}
  }});
}})();
</script>"""

    ascii_fb = html.escape(_ascii_table(report))
    perf_html = _perf_table_html(bm)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{model} Benchmark Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{model} Benchmark Report</h1>
{badge_html}
<p style="color:#64748b;font-size:.85rem">
  {ts} &nbsp;|&nbsp; GPU: {gpu} &nbsp;|&nbsp; harness: {hv}
</p>

<div class="charts">
  <div class="section">
    <h2>质量维度雷达图</h2>
    <canvas id="radarChart"></canvas>
    <noscript><pre class="fallback">{ascii_fb}</pre></noscript>
  </div>
  <div class="section">
    <h2>性能指标</h2>
    {perf_html}
  </div>
</div>

{bar_section}
{drift_block}

<div class="section">
  <h2>断网降级摘要</h2>
  <noscript><pre class="fallback">{ascii_fb}</pre></noscript>
  <pre class="fallback" id="asciiTable" style="display:none">{ascii_fb}</pre>
</div>

<script src="{_CHART_JS}" onerror="document.getElementById('asciiTable').style.display='block'"></script>
<script>
(function(){{
  if(typeof Chart==='undefined')return;
  new Chart(document.getElementById('radarChart'),{{
    type:'radar',
    data:{{
      labels:{radar_labels},
      datasets:[{{
        label:'{model}',
        data:{radar_json},
        fill:true,
        backgroundColor:'rgba(59,130,246,0.15)',
        borderColor:'rgba(59,130,246,0.8)',
        pointBackgroundColor:'rgba(59,130,246,0.8)'
      }}]
    }},
    options:{{responsive:true,scales:{{r:{{min:0,max:1}}}},plugins:{{legend:{{position:'bottom'}}}}}}
  }});
}})();
</script>
</body>
</html>"""


def _render_compare(report: dict) -> str:
    baseline = html.escape(str(report.get("baseline", "baseline")))
    candidate = html.escape(str(report.get("candidate", "candidate")))
    verdict = str(report.get("final_verdict") or report.get("verdict") or "")
    badge_color = _verdict_color(verdict)
    badge_html = (f'<span class="badge" style="background:{badge_color}">'
                  f'{html.escape(verdict)}</span>') if verdict else ""

    b_report = report.get("baseline_report") or {}
    c_report = report.get("candidate_report") or {}

    b_radar = _quality_radar_data(b_report)
    c_radar = _quality_radar_data(c_report)

    labels_json = json.dumps(b_radar["labels"])
    b_vals = json.dumps(b_radar["values"])
    c_vals = json.dumps(c_radar["values"])

    # Performance comparison table
    quality = report.get("quality") or {}
    perf = report.get("performance") or {}
    perf_rows = []
    for k, v in perf.items():
        if isinstance(v, dict):
            b_v = v.get("baseline", "—")
            c_v = v.get("candidate", "—")
        else:
            b_v, c_v = "—", "—"
        perf_rows.append((k, str(b_v), str(c_v)))

    quality_rows = []
    for k, v in quality.items():
        if isinstance(v, dict):
            sig = "★" if v.get("significant") else ""
            b_v = f"{v.get('baseline_mean', 0):.3f}"
            c_v = f"{v.get('candidate_mean', 0):.3f}"
            quality_rows.append((k + sig, b_v, c_v))

    def _rows_html(rows: list) -> str:
        return "".join(
            f"<tr><td>{html.escape(str(a))}</td>"
            f"<td>{html.escape(str(b))}</td>"
            f"<td>{html.escape(str(c))}</td></tr>"
            for a, b, c in rows
        )

    reasons = report.get("reasons") or []
    reasons_html = "".join(f"<li>{html.escape(str(r))}</li>" for r in reasons)

    ascii_b = html.escape(_ascii_table(b_report))
    ascii_c = html.escape(_ascii_table(c_report))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Compare: {baseline} vs {candidate}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>模型对比: {baseline} vs {candidate}</h1>
<div class="compare-verdict">{badge_html}</div>

<div class="section">
  <h2>质量维度雷达图（叠加）</h2>
  <canvas id="radarChart"></canvas>
  <noscript>
    <pre class="fallback">{ascii_b}</pre>
    <pre class="fallback">{ascii_c}</pre>
  </noscript>
</div>

<div class="section">
  <h2>质量指标对比</h2>
  <table class="perf">
    <thead><tr><th>指标</th><th>{baseline}</th><th>{candidate}</th></tr></thead>
    <tbody>{_rows_html(quality_rows)}</tbody>
  </table>
</div>

<div class="section">
  <h2>性能指标对比</h2>
  <table class="perf">
    <thead><tr><th>指标</th><th>{baseline}</th><th>{candidate}</th></tr></thead>
    <tbody>{_rows_html(perf_rows)}</tbody>
  </table>
</div>

<div class="section">
  <h2>判定理由</h2>
  <ul>{reasons_html}</ul>
</div>

<script src="{_CHART_JS}" onerror="
  document.querySelectorAll('noscript').forEach(function(n){{
    var p=document.createElement('div');p.innerHTML=n.textContent;n.after(p);
  }})"></script>
<script>
(function(){{
  if(typeof Chart==='undefined')return;
  new Chart(document.getElementById('radarChart'),{{
    type:'radar',
    data:{{
      labels:{labels_json},
      datasets:[
        {{label:'{baseline}',data:{b_vals},fill:true,
          backgroundColor:'rgba(59,130,246,0.1)',borderColor:'rgba(59,130,246,0.8)'}},
        {{label:'{candidate}',data:{c_vals},fill:true,
          backgroundColor:'rgba(239,68,68,0.1)',borderColor:'rgba(239,68,68,0.8)'}}
      ]
    }},
    options:{{responsive:true,scales:{{r:{{min:0,max:1}}}},plugins:{{legend:{{position:'bottom'}}}}}}
  }});
}})();
</script>
</body>
</html>"""


def generate_html(report: dict) -> str:
    """主入口。根据 report 中的字段判断单模型还是 compare 报告。"""
    if not report:
        return _render_single({})
    mode = report.get("mode") or ""
    if mode == "compare" or ("baseline" in report and "candidate" in report):
        return _render_compare(report)
    return _render_single(report)
