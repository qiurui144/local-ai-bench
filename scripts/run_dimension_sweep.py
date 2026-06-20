#!/usr/bin/env python3
"""Run selected benchmark dimensions while ignoring per-model quick-test skips."""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import load_benchmarks_config, load_models  # noqa: E402
import run_benchmark as rb  # noqa: E402


REPORTS = ROOT / "output" / "reports"


DEFAULT_SWEEP_OVERRIDES = {
    "concurrency": {
        "concurrencies": [1, 2, 4, 8],
        "duration_s": 10,
    },
    "stability": {
        "duration_s": 180,
        "sample_interval_s": 10,
    },
    "translation": {
        "flores": {"split": "devtest", "num_samples": 12},
        "directions": ["zh->en", "en->zh"],
        "levels": ["l1", "l3"],
        "run_comet": False,
        "ttft_samples": 2,
        "throughput_duration_s": 15,
    },
    "general_ability": {
        "tasks": {
            "gsm8k": {"num_samples": 12, "split": "test"},
            "mmlu": {
                "per_subject": 4,
                "subjects": [
                    "professional_law",
                    "logical_fallacies",
                    "computer_security",
                    "elementary_mathematics",
                ],
            },
            "hellaswag": {"num_samples": 12, "split": "validation"},
        },
        "thresholds": {
            "gsm8k_min": 0.20,
            "mmlu_min": 0.20,
            "hellaswag_min": 0.20,
        },
    },
    "conditioned": {
        "context_ladder": [1024, 4096, 8192],
        "answer_margin_tokens": 512,
        "max_tokens": 64,
    },
    "scenarios": {
        "num_cases": 3,
        "judge_model": None,
    },
    "conversation_drift": {
        "num_cases": 2,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _default(obj):
    return rb._default(obj)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--dimensions", required=True, help="Comma-separated benchmark dimensions")
    parser.add_argument("--tag", default="dimension-sweep")
    parser.add_argument("--profile", choices=["windows-full"], default="windows-full")
    args = parser.parse_args()

    wanted_models = [x.strip() for x in args.models.split(",") if x.strip()]
    wanted_dims = [x.strip() for x in args.dimensions.split(",") if x.strip()]
    unknown = set(wanted_dims) - set(rb.DIMENSIONS)
    if unknown:
        raise SystemExit(f"unknown dimensions: {sorted(unknown)}")

    all_models = {m.name: m for m in load_models(ROOT / "models.yaml")}
    missing = [m for m in wanted_models if m not in all_models]
    if missing:
        raise SystemExit(f"unknown models: {missing}")

    golden = json.loads((ROOT / "golden" / "expectations.json").read_text(encoding="utf-8"))
    bench_cfg = load_benchmarks_config(ROOT / "models.yaml")
    if args.profile == "windows-full":
        bench_cfg = _deep_merge(bench_cfg, DEFAULT_SWEEP_OVERRIDES)

    skip = set(rb.DIMENSIONS) - set(wanted_dims)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    has_fail = False
    has_warn = False

    for model_name in wanted_models:
        model_cfg = copy.deepcopy(all_models[model_name])
        model_cfg.benchmarks = copy.deepcopy(model_cfg.benchmarks or {})
        model_cfg.benchmarks["skip"] = []
        started = time.monotonic()
        result = rb.run_all_for_model(model_cfg, golden, skip, bench_cfg)
        result["sweep"] = {
            "tag": args.tag,
            "profile": args.profile,
            "requested_dimensions": wanted_dims,
            "duration_s": round(time.monotonic() - started, 3),
        }
        results.append(result)

        stem = f"{model_name}_{args.tag}_{timestamp}"
        (REPORTS / f"{stem}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_default),
            encoding="utf-8",
        )
        (REPORTS / f"{stem}.md").write_text(rb.render_markdown(result), encoding="utf-8")
        print(f"wrote {REPORTS / f'{stem}.json'}")

        for dim in rb.QUALITY_DIMS:
            verdict = result.get("benchmarks", {}).get(dim, {}).get("verdict")
            if verdict == "FAIL":
                has_fail = True
            elif verdict in {"WARN", "BLOCKED"}:
                has_warn = True

    summary = {
        "tag": args.tag,
        "timestamp": timestamp,
        "models": wanted_models,
        "dimensions": wanted_dims,
        "results": [
            {
                "model": r.get("model"),
                "duration_s": r.get("sweep", {}).get("duration_s"),
                "benchmarks": {
                    k: v.get("verdict", "MEASURED") if isinstance(v, dict) else "MEASURED"
                    for k, v in (r.get("benchmarks") or {}).items()
                },
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    (REPORTS / f"{args.tag}_{timestamp}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 2 if has_fail else (1 if has_warn else 0)


if __name__ == "__main__":
    raise SystemExit(main())
