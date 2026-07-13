#!/usr/bin/env python3
"""Controlled K3 conditioned-quality probe with explicit per-call timeouts."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.conditioned.context_corpus import build_context, load_cail_paragraphs, load_needles


QUESTION_SUFFIX = "\n\n仅根据上文事实回答下列问题,只给出答案本身:\n问题:"


def norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def chat(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    timeout_s: float,
    prompt_prefix: str,
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_prefix + prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=timeout_s,
        )
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = time.perf_counter() - started
    item: dict[str, Any] = {
        "ok": response.status_code == 200,
        "status_code": response.status_code,
        "elapsed_s": round(elapsed, 3),
    }
    try:
        body = response.json()
    except Exception as exc:
        item["ok"] = False
        item["error"] = f"json_decode: {exc}"
        item["body_prefix"] = response.text[:1000]
        return item
    message = (body.get("choices") or [{}])[0].get("message") or {}
    item["content"] = message.get("content") or ""
    item["finish_reason"] = (body.get("choices") or [{}])[0].get("finish_reason")
    item["usage"] = body.get("usage")
    return item


def summarize(rows: list[dict[str, Any]], probes: list[dict[str, Any]]) -> dict[str, Any]:
    by_role = {"task": {"correct": 0, "total": 0}, "needle": {"correct": 0, "total": 0}}
    errors = 0
    latencies = []
    for row in rows:
        role = row["role"]
        by_role[role]["total"] += 1
        if row.get("error") or not row.get("ok"):
            errors += 1
        if row.get("correct"):
            by_role[role]["correct"] += 1
        if row.get("elapsed_s") is not None:
            latencies.append(float(row["elapsed_s"]))
    return {
        "n": len(probes),
        "errors": errors,
        "task_accuracy": round(by_role["task"]["correct"] / max(1, by_role["task"]["total"]), 3),
        "needle_recall": round(by_role["needle"]["correct"] / max(1, by_role["needle"]["total"]), 3),
        "latency_s": {
            "max": round(max(latencies), 3) if latencies else None,
            "p50": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3-30B-A3B-Q4_0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contexts", default="1024,3072")
    parser.add_argument("--timeout-s", type=float, default=240)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-prefix", default="/no_think\n")
    parser.add_argument("--chat-template-kwargs", default='{"enable_thinking":false}')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes = load_needles(Path("datasets/conditioned/needles.jsonl"))
    if probes is None:
        raise SystemExit("missing datasets/conditioned/needles.jsonl")
    paragraphs = load_cail_paragraphs()
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    chat_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs.strip() else {}

    result: dict[str, Any] = {
        "benchmark": "conditioned_control",
        "model": args.model,
        "base_url": args.base_url,
        "contexts": {},
        "timeout_s": args.timeout_s,
        "max_tokens": args.max_tokens,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for target in contexts:
        facts = [(p["id"], float(p["depth"]), p["fact"]) for p in probes]
        context = build_context(target, facts, paragraphs)
        rows: list[dict[str, Any]] = []
        for probe in probes:
            prompt = context.text + QUESTION_SUFFIX + probe["question"]
            call = chat(
                args.base_url,
                args.model,
                prompt,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
                prompt_prefix=args.prompt_prefix,
                chat_template_kwargs=chat_kwargs,
            )
            content = call.get("content") or ""
            row = {
                "probe_id": probe["id"],
                "role": probe["role"],
                "question": probe["question"],
                "answer": probe["answer"],
                "ok": call.get("ok"),
                "status_code": call.get("status_code"),
                "elapsed_s": call.get("elapsed_s"),
                "finish_reason": call.get("finish_reason"),
                "usage": call.get("usage"),
                "content_prefix": content[:200],
                "correct": norm(probe["answer"]) in norm(content),
                "error": call.get("error"),
            }
            rows.append(row)
            with (args.out_dir / "trace.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"context_tokens": target, **row}, ensure_ascii=False) + "\n")
        result["contexts"][str(target)] = {
            "target_tokens": target,
            "estimated_tokens": context.est_tokens,
            "insertions": context.insertions,
            "summary": summarize(rows, probes),
            "rows": rows,
        }

    verdict = "PASS"
    reasons: list[str] = []
    for ctx, block in result["contexts"].items():
        summary = block["summary"]
        if summary["errors"]:
            verdict = "FAIL"
            reasons.append(f"{ctx}: {summary['errors']}/{summary['n']} requests errored")
        if summary["needle_recall"] < 0.5:
            verdict = "FAIL"
            reasons.append(f"{ctx}: needle_recall {summary['needle_recall']} < 0.5")
    result["verdict"] = verdict
    result["verdict_reasons"] = reasons

    (args.out_dir / "conditioned-control.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# K3 Conditioned Control",
        "",
        f"- model: `{args.model}`",
        f"- timeout_s: `{args.timeout_s}`",
        f"- verdict: `{verdict}`",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines += ["", "| context | task_accuracy | needle_recall | errors | latency_p50_s | latency_max_s |"]
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for ctx, block in result["contexts"].items():
        s = block["summary"]
        lines.append(
            f"| {ctx} | {s['task_accuracy']} | {s['needle_recall']} | "
            f"{s['errors']}/{s['n']} | {s['latency_s']['p50']} | {s['latency_s']['max']} |"
        )
    (args.out_dir / "conditioned-control.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
