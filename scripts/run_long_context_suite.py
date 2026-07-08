#!/usr/bin/env python3
"""Run the long-context edge subset against an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import ModelConfig  # noqa: E402
from benchmark.long_context.runner import run_long_context  # noqa: E402


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    return json.loads(raw) if raw else {}


def _render_markdown(block: dict) -> str:
    lines = [
        f"# {block.get('model')} Long-Context Edge Suite",
        "",
        f"- Mode: `{block.get('mode')}`",
        f"- Required for: {block.get('required_for')}",
        f"- Verdict: **{block.get('verdict')}**",
        "",
        "Sources:",
    ]
    for name, url in (block.get("sources") or {}).items():
        lines.append(f"- {name}: {url}")
    for reason in block.get("verdict_reasons") or []:
        lines.append(f"- {reason}")
    lines += ["", "## Summary", ""]
    summary = block.get("summary") or {}
    latency = summary.get("latency") or {}
    lines.append(f"- Measured cases: {summary.get('measured_cases', 0)}")
    if summary.get("blocked_suites"):
        lines.append(f"- Blocked suites: {', '.join(summary['blocked_suites'])}")
    if latency:
        lines.append(
            f"- Latency mean/p95/max: {latency.get('mean_s')} / "
            f"{latency.get('p95_s')} / {latency.get('max_s')} s"
        )
    lines += ["", "## Suites", ""]
    for key, suite in (block.get("suites") or {}).items():
        lines.append(f"### {key}")
        lines.append(f"- Verdict: {suite.get('verdict')}")
        if suite.get("reason"):
            lines.append(f"- Reason: {suite.get('reason')}")
        if "recall" in suite:
            lines.append(f"- Recall: {suite.get('recall')} ({suite.get('measured')} measured)")
        if "score" in suite:
            lines.append(f"- Score: {suite.get('score')} ({suite.get('measured')} measured)")
        if suite.get("type_scores"):
            rendered = ", ".join(f"{k}={v}" for k, v in sorted(suite["type_scores"].items()))
            lines.append(f"- Type scores: {rendered}")
        if "accuracy" in suite:
            lines.append(f"- Accuracy: {suite.get('accuracy')} ({suite.get('measured')} measured)")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="model id sent in API payload")
    parser.add_argument("--model-name", default="", help="friendly report name")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt-prefix", default="")
    parser.add_argument("--chat-template-kwargs", default="")
    parser.add_argument("--provider", default="generic")
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--context-lengths", default="1024,3072")
    parser.add_argument("--depth-percents", default="10,50,90")
    parser.add_argument("--longbench-datasets", default="passage_retrieval_en,passage_count")
    parser.add_argument("--leval-tasks", default="quality,coursera")
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--questions-per-document", type=int, default=1)
    parser.add_argument("--airplane-manual-case-limit", type=int, default=12)
    parser.add_argument("--airplane-manual-context-tokens", type=int, default=0,
                        help="Context-token budget for aviation manual windows; defaults to max-input-tokens.")
    parser.add_argument("--airplane-manual-prompt-budget-safety", type=float, default=0.70,
                        help="Conservative prompt budget multiplier for aviation manual windows.")
    parser.add_argument("--skip-suites", default="", help="comma-separated suite keys to skip")
    parser.add_argument("--case-result-log", default="",
                        help="JSONL checkpoint path; defaults to <out-dir>/long-context-cases.jsonl")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(
        name=args.model_name or args.model,
        provider=args.provider,
        model_id=args.model,
        base_url_override=args.base_url.rstrip("/"),
        task_type="text_only",
        prompt_prefix=args.prompt_prefix,
        chat_template_kwargs=_parse_json(args.chat_template_kwargs),
    )
    case_log = Path(args.case_result_log) if args.case_result_log else out_dir / "long-context-cases.jsonl"
    cfg = {
        "max_input_tokens": args.max_input_tokens,
        "timeout_s": args.timeout_s,
        "progress_log": True,
        "case_result_log": str(case_log),
        "skip_suites": [x.strip() for x in args.skip_suites.split(",") if x.strip()],
        "suites": {
            "needle_in_a_haystack": {
                "context_lengths": [int(x) for x in args.context_lengths.split(",") if x.strip()],
                "depth_percents": [int(x) for x in args.depth_percents.split(",") if x.strip()],
            },
            "longbench": {
                "datasets": [x.strip() for x in args.longbench_datasets.split(",") if x.strip()],
                "samples_per_dataset": args.samples_per_dataset,
            },
            "leval": {
                "tasks": [x.strip() for x in args.leval_tasks.split(",") if x.strip()],
                "samples_per_task": args.samples_per_task,
                "questions_per_document": args.questions_per_document,
            },
            "aviation_manuals": {
                "case_limit": args.airplane_manual_case_limit,
                "target_context_tokens": args.airplane_manual_context_tokens or args.max_input_tokens,
                "prompt_budget_safety": args.airplane_manual_prompt_budget_safety,
                "depth_percents": [int(x) for x in args.depth_percents.split(",") if x.strip()],
            },
        },
    }
    block = run_long_context(model_cfg, cfg, ROOT)
    block["timestamp"] = datetime.now().isoformat()
    (out_dir / "long-context-result.json").write_text(
        json.dumps(block, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "long-context-result.md").write_text(_render_markdown(block), encoding="utf-8")
    print(json.dumps({
        "model": block.get("model"),
        "verdict": block.get("verdict"),
        "summary": block.get("summary"),
    }, ensure_ascii=False))
    return 0 if block.get("verdict") in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
