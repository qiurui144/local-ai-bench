#!/usr/bin/env python3
"""Summarize registered models and benchmark evidence into release reports."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPORT_ROOTS = [
    Path("output/reports/amd-win-x86/reports"),
    Path("output/reports/intel-win-x86/reports"),
    Path("output/reports"),
]

MODEL_LIMIT_ALIASES = {
    "llama3.2:3b": "llama3.2-3b-amd-win",
    "qwen2.5:14b": "qwen2.5-14b-amd-win",
    "llama3.2:1b": "llama3.2-1b-intel-win",
    "qwen2.5:7b": "qwen2.5-7b-intel-win",
}

STRUCTURED_OCR_ALIASES = {
    "structured-ocr-amd-directml": "rapidocr-amd-directml",
    "structured-ocr-amd-vitisai": "rapidocr-amd-npu",
    "structured-ocr-amd-rapidocr-cpu": "rapidocr-cpu",
    "structured-ocr-intel-openvino": "rapidocr-intel-openvino",
    "structured-ocr-intel-directml": "rapidocr-intel-directml",
    "structured-ocr-intel-rapidocr-cpu": "rapidocr-cpu",
}


@dataclass
class Evidence:
    reports: list[Path] = field(default_factory=list)
    verdicts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, str] = field(default_factory=dict)


def _ts_key(path: Path) -> str:
    matches = re.findall(r"20\d{6}(?:_\d{6})?", path.name)
    return matches[-1] if matches else path.name


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}%"


def _bench_metrics(name: str, bench: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    if name == "ttft":
        stats = bench.get("ttft_ms_stats", {})
        out["TTFT p50/p95"] = f"{_fmt(stats.get('p50'))}/{_fmt(stats.get('p95'))} ms"
    elif name == "throughput":
        out["TPS"] = _fmt(bench.get("aggregate_tps"), 2)
    elif name == "prefill_decode":
        pp = bench.get("prefill", {}).get("tok_per_sec", {}).get("p50")
        tg = bench.get("decode", {}).get("tok_per_sec", {}).get("p50")
        out["PP/TG"] = f"{_fmt(pp, 2)}/{_fmt(tg, 2)} tok/s"
    elif name == "embedding":
        agg = bench.get("aggregate", {})
        lat = bench.get("performance", {}).get("latency", {}).get("single_query_latency_ms_stats", {})
        out["embed"] = f"hit@1 {_fmt(agg.get('hit@1'), 3)}, nDCG {_fmt(agg.get('ndcg@10'), 3)}, p50 {_fmt(lat.get('p50'))} ms"
    elif name == "rerank":
        agg = bench.get("aggregate", {})
        lat = agg.get("query_rerank_latency_ms_stats", {})
        out["rerank"] = f"nDCG {_fmt(agg.get('ndcg@10'), 3)}, MRR {_fmt(agg.get('mrr'), 3)}, p50 {_fmt(lat.get('p50'))} ms"
    elif name == "ocr":
        agg = bench.get("aggregate", {})
        lat = agg.get("latency_ms_stats", {})
        out["ocr"] = f"CER {_pct(agg.get('cer'))}, NED {_pct(agg.get('ned'))}, p50 {_fmt(lat.get('p50'))} ms"
    elif name == "asr":
        agg = bench.get("aggregate", {})
        out["asr"] = f"CER {_pct(agg.get('cer'))}, RTF {_fmt(agg.get('rtf_mean'), 3)}"
    elif name in {"general_ability", "translation", "conditioned", "scenarios", "conversation_drift"}:
        out[name] = bench.get("verdict", "")
    elif name == "concurrency":
        steps = bench.get("steps", [])
        if steps:
            best = max(steps, key=lambda s: s.get("aggregate_tps", 0))
            out["concurrency"] = f"c{best.get('concurrency')} {_fmt(best.get('aggregate_tps'), 2)} tok/s"
    elif name == "stability":
        out["stability"] = f"drift {_fmt(bench.get('latency_drift_ratio'), 2)}"
    return {k: v for k, v in out.items() if v}


def _model_limit_metrics(data: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    ok_conc = [x for x in data.get("concurrency", []) if x.get("success_rate") == 1.0]
    if ok_conc:
        best = max(ok_conc, key=lambda x: x.get("aggregate_decode_tps", 0))
        out["limit concurrency"] = f"c{best.get('concurrency')} {_fmt(best.get('aggregate_decode_tps'), 2)} tok/s"
    ok_ctx = [x for x in data.get("context", []) if x.get("ok")]
    if ok_ctx:
        out["max context"] = f"{max(x.get('target_prompt_tokens_approx', 0) for x in ok_ctx) // 1024}k"
    return out


def collect_evidence() -> dict[str, Evidence]:
    evidence: dict[str, Evidence] = {}
    paths: list[Path] = []
    for root in REPORT_ROOTS:
        if root.exists():
            paths.extend(root.glob("*.json"))
    for path in sorted(set(paths), key=_ts_key):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = data.get("model") or data.get("model_name") or path.stem
        model = MODEL_LIMIT_ALIASES.get(model, model)
        model = STRUCTURED_OCR_ALIASES.get(model, model)
        ev = evidence.setdefault(model, Evidence())
        ev.reports.append(path)
        if data.get("benchmark") == "structured_ocr":
            ev.verdicts["structured_ocr"] = data.get("verdict", "")
            lat = data.get("latency_ms_stats", {})
            ev.metrics["structured OCR"] = (
                f"field {_pct(data.get('field_accuracy'))}, CER {_pct(data.get('text_cer'))}, p50 {_fmt(lat.get('p50'))} ms"
            )
            continue
        if data.get("benchmark") == "ollama_model_limits":
            ev.verdicts["model_limits"] = "MEASURED"
            ev.metrics.update(_model_limit_metrics(data))
            continue
        for bench_name, bench in (data.get("benchmarks") or {}).items():
            if not isinstance(bench, dict):
                continue
            verdict = bench.get("verdict")
            if verdict:
                ev.verdicts[bench_name] = verdict
            ev.metrics.update(_bench_metrics(bench_name, bench))
    return evidence


def _capabilities(model: dict[str, Any]) -> str:
    caps = [k.removesuffix("_capable") for k, v in model.items() if k.endswith("_capable") and v]
    if model.get("is_vlm") or model.get("task_type") == "vlm":
        caps.append("vlm")
    if not caps and model.get("task_type") == "text_only":
        caps.append("llm")
    return ",".join(dict.fromkeys(caps)) or "-"


def _row_status(ev: Evidence | None) -> str:
    if not ev or not ev.verdicts:
        return "REGISTERED"
    priority = ["FAIL", "BLOCKED", "WARN", "SKIP", "SKIPPED", "PASS", "MEASURED"]
    values = set(ev.verdicts.values())
    for item in priority:
        if item in values:
            return item
    return ",".join(sorted(values))


def _allowed_dimensions(model: dict[str, Any]) -> set[str]:
    caps = set(_capabilities(model).split(","))
    allowed = {"model_limits"}
    if "embedding" in caps:
        allowed.add("embedding")
    if "rerank" in caps:
        allowed.add("rerank")
    if "ocr" in caps:
        allowed.update({"ocr", "structured_ocr"})
    if "asr" in caps:
        allowed.add("asr")
    if "translation" in caps or "llm" in caps:
        allowed.update({
            "ttft", "throughput", "prefill_decode", "concurrency", "stability",
            "translation", "general_ability", "conditioned", "scenarios",
            "conversation_drift",
        })
    if "vlm" in caps:
        allowed.update({
            "accuracy", "ttft", "throughput", "prefill_decode", "concurrency",
            "stability", "general_ability", "conditioned", "scenarios",
            "conversation_drift",
        })
    return allowed


def _filtered_evidence(model: dict[str, Any], ev: Evidence | None) -> Evidence | None:
    if ev is None:
        return None
    allowed = _allowed_dimensions(model)
    verdicts = {k: v for k, v in ev.verdicts.items() if k in allowed}
    metrics = {}
    for key, value in ev.metrics.items():
        metric_dim = key.split()[0]
        if metric_dim == "embed":
            metric_dim = "embedding"
        if (
            metric_dim in allowed
            or key in {"TTFT p50/p95", "TPS", "PP/TG", "limit concurrency", "max context"}
            or (key == "structured OCR" and "structured_ocr" in allowed)
        ):
            metrics[key] = value
    return Evidence(reports=ev.reports, verdicts=verdicts, metrics=metrics)


def render_report(models: list[dict[str, Any]], evidence: dict[str, Evidence], lang: str) -> str:
    zh = lang == "zh"
    title = "全模型矩阵评测结果与最佳选型" if zh else "Full Model Matrix Results and Selection"
    lines = [f"# {title}", "", "Updated: 2026-06-19 20:10 CST.", ""]
    lines += [
        "## 选型结论" if zh else "## Selection",
        "",
    ]
    if zh:
        lines += [
            "- AMD Windows 默认 LLM：`qwen2.5-7b-amd-win`，质量/吞吐均衡；高参数上限用 `qwen2.5-14b-amd-win`，但吞吐约 7.67 tok/s。",
            "- Intel Windows 默认 LLM：`qwen2.5-7b-intel-win` 用于更高质量，`qwen2.5-3b-intel-win` 用于轻量默认回归。",
            "- 高并发/长上下文轻量 LLM：AMD `llama3.2-3b-amd-win`，Intel `llama3.2-1b-intel-win`，两者均验证 32k 上下文和 32 并发稳定。",
            "- Embedding：AMD 首选 `qwen3-embedding-0.6b-amd`，Intel 首选 `qwen3-embedding-0.6b-intel-win`；AMD `bge-m3-amd` 可作多语言替代。",
            "- Reranker：两端默认 `bge-reranker-base-*-win`；`bge-reranker-v2-m3-*-win` 质量同过但 CPU 延迟约 3.7 倍。",
            "- OCR：AMD 首选 `rapidocr-amd-directml`；Intel 首选 `rapidocr-intel-openvino`；Intel DirectML OCR 不可用。",
            "- ASR：两端均使用 `sensevoice-small-*-win`，均已 PASS。",
            "- VLM：`llava:7b` 路径可运行但质量 fixture FAIL，不建议作为当前最佳 VLM。",
        ]
    else:
        lines += [
            "- AMD Windows default LLM: `qwen2.5-7b-amd-win`; use `qwen2.5-14b-amd-win` only when parameter count matters more than speed.",
            "- Intel Windows default LLM: `qwen2.5-7b-intel-win` for quality, `qwen2.5-3b-intel-win` for lightweight regression.",
            "- Lightweight high-concurrency/long-context LLMs: AMD `llama3.2-3b-amd-win`, Intel `llama3.2-1b-intel-win`.",
            "- Embedding: AMD `qwen3-embedding-0.6b-amd`, Intel `qwen3-embedding-0.6b-intel-win`; AMD `bge-m3-amd` is the multilingual alternative.",
            "- Reranker: use `bge-reranker-base-*-win` by default; `bge-reranker-v2-m3-*-win` passes but is about 3.7x slower on CPU.",
            "- OCR: AMD `rapidocr-amd-directml`; Intel `rapidocr-intel-openvino`; Intel DirectML OCR is not usable.",
            "- ASR: `sensevoice-small-*-win` passes on both platforms.",
            "- VLM: `llava:7b` runs, but fixture quality fails; it is not the current best VLM choice.",
        ]
    lines += ["", "## Matrix", ""]
    headers = ["Model", "Target", "Provider", "Role", "Caps", "Status", "Verdicts", "Key metrics", "Latest report"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for model in models:
        name = model.get("name", "")
        ev = _filtered_evidence(model, evidence.get(name))
        verdicts = ", ".join(f"{k}:{v}" for k, v in sorted((ev.verdicts if ev else {}).items())) or "-"
        metrics = "; ".join(f"{k} {v}" for k, v in sorted((ev.metrics if ev else {}).items())) or "-"
        latest = str(sorted(ev.reports, key=_ts_key)[-1]) if ev and ev.reports else "-"
        lines.append("| " + " | ".join([
            name,
            model.get("target") or "local/reference",
            model.get("provider", ""),
            model.get("role", ""),
            _capabilities(model),
            _row_status(ev),
            verdicts.replace("|", "/"),
            metrics.replace("|", "/"),
            latest,
        ]) + " |")
    lines += [
        "",
        "## Evidence Rule" if not zh else "## 证据规则",
        "",
    ]
    if zh:
        lines += [
            "- 表格基于 `models.yaml` 全量注册模型生成。",
            "- 每个模型聚合 `output/reports/**.json` 中可解析的最新/专项证据。",
            "- `REGISTERED` 表示模型已注册但当前工作区没有可复核 JSON 结果。",
            "- 旧生成式 reranker 代理结果不作为当前 reranker 通过依据；当前采用 `local_reranker` CrossEncoder。",
        ]
    else:
        lines += [
            "- The table is generated from all registered models in `models.yaml`.",
            "- Evidence is aggregated from parseable JSON files under `output/reports/**`.",
            "- `REGISTERED` means the model is configured but no local JSON evidence exists in this workspace.",
            "- Historical generative rerank proxy reports are not used as current pass evidence; current rerank uses `local_reranker` CrossEncoder.",
        ]
    return "\n".join(lines) + "\n"


def render_json_summary(models: list[dict[str, Any]], evidence: dict[str, Evidence]) -> dict[str, Any]:
    rows = []
    for model in models:
        name = model.get("name", "")
        ev = _filtered_evidence(model, evidence.get(name))
        rows.append({
            "model": name,
            "target": model.get("target") or "local/reference",
            "provider": model.get("provider", ""),
            "role": model.get("role", ""),
            "capabilities": _capabilities(model),
            "status": _row_status(ev),
            "verdicts": ev.verdicts if ev else {},
            "metrics": ev.metrics if ev else {},
            "latest_report": str(sorted(ev.reports, key=_ts_key)[-1]) if ev and ev.reports else "",
        })
    return {
        "generated_at": "2026-06-19T20:10:00+08:00",
        "source_models": "models.yaml",
        "evidence_roots": [str(p) for p in REPORT_ROOTS],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models.yaml")
    parser.add_argument("--out-zh", default="reports/2026-06-19-all-model-matrix-results.zh.md")
    parser.add_argument("--out-en", default="reports/2026-06-19-all-model-matrix-results.en.md")
    parser.add_argument("--out-json", default="reports/2026-06-19-all-model-matrix-results.json")
    args = parser.parse_args()

    models = yaml.safe_load(Path(args.models).read_text(encoding="utf-8"))["models"]
    evidence = collect_evidence()
    Path(args.out_zh).write_text(render_report(models, evidence, "zh"), encoding="utf-8")
    Path(args.out_en).write_text(render_report(models, evidence, "en"), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(render_json_summary(models, evidence), ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out_zh)
    print(args.out_en)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
