"""Translation latency / throughput per language pair.

Reuses the project performance idiom (``benchmark/performance.py``):
streaming requests for TTFT, sustained sync requests for tokens/s. The only
difference from the generic performance suite is that requests carry real
translation prompts (L1 single-sentence), and results are reported **per
language pair** so zh->en and en->zh can be compared directly.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Optional, Sequence

from common import ModelConfig, infer_stream, infer_sync, summarize_latencies

from . import prompts
from .datasets import TranslationPair

logger = logging.getLogger(__name__)


def run_translation_ttft(
    model_cfg: ModelConfig,
    pairs: Sequence[TranslationPair],
    *,
    samples: int = 5,
    max_tokens: int = 200,
) -> dict:
    """Per-language-pair TTFT P50/P95 over ``samples`` streamed translations."""
    if not pairs:
        return {"benchmark": "translation_ttft", "model": model_cfg.name, "skipped": True}

    src_lang, tgt_lang = pairs[0].src_lang, pairs[0].tgt_lang
    ttfts: list[float] = []
    totals: list[float] = []
    errors = 0
    for i in range(samples):
        p = pairs[i % len(pairs)]
        res = infer_stream(
            model_cfg,
            prompt=prompts.l1_single_sentence(p.src, src_lang, tgt_lang),
            image_path=None,
            max_tokens=max_tokens,
        )
        if res.ok and res.ttft_ms > 0:
            ttfts.append(res.ttft_ms)
            totals.append(res.latency_ms)
        else:
            errors += 1
        logger.info("  [TTFT %d/%d] %s %s->%s ttft=%.0fms",
                    i + 1, samples, model_cfg.name, src_lang, tgt_lang, res.ttft_ms)

    return {
        "benchmark": "translation_ttft",
        "model": model_cfg.name,
        "lang_pair": f"{src_lang}->{tgt_lang}",
        "samples": samples,
        "ttft_ms_stats": summarize_latencies(ttfts),
        "total_latency_ms_stats": summarize_latencies(totals),
        "errors": errors,
        "error_rate": errors / samples if samples else 0,
    }


def run_translation_throughput(
    model_cfg: ModelConfig,
    pairs: Sequence[TranslationPair],
    *,
    duration_s: float = 60.0,
    max_tokens: int = 256,
) -> dict:
    """Per-language-pair sustained tokens/s over ``duration_s`` seconds."""
    if not pairs:
        return {"benchmark": "translation_throughput", "model": model_cfg.name, "skipped": True}

    src_lang, tgt_lang = pairs[0].src_lang, pairs[0].tgt_lang
    deadline = time.monotonic() + duration_s
    total_output = total_input = 0
    latencies: list[float] = []
    tps_per_req: list[float] = []
    errors = n = 0

    while time.monotonic() < deadline:
        p = pairs[n % len(pairs)]
        res = infer_sync(
            model_cfg,
            prompt=prompts.l1_single_sentence(p.src, src_lang, tgt_lang),
            image_path=None,
            max_tokens=max_tokens,
        )
        n += 1
        if res.ok:
            total_output += res.output_tokens
            total_input += res.input_tokens
            latencies.append(res.latency_ms)
            if res.tokens_per_sec > 0:
                tps_per_req.append(res.tokens_per_sec)
        else:
            errors += 1

    wall = duration_s if n > 0 else 1
    return {
        "benchmark": "translation_throughput",
        "model": model_cfg.name,
        "lang_pair": f"{src_lang}->{tgt_lang}",
        "duration_s": duration_s,
        "requests": n,
        "errors": errors,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "aggregate_tps": total_output / wall,
        "per_request_tps_stats": {
            "p50": statistics.median(tps_per_req) if tps_per_req else 0,
            "p95": sorted(tps_per_req)[int(len(tps_per_req) * 0.95)] if tps_per_req else 0,
        },
        "latency_stats_ms": summarize_latencies(latencies),
    }


def run_translation_performance(
    model_cfg: ModelConfig,
    pairs_by_direction: dict[str, Sequence[TranslationPair]],
    *,
    ttft_samples: int = 5,
    throughput_duration_s: float = 60.0,
    skip: Optional[set[str]] = None,
) -> dict:
    """Run TTFT + throughput for each language direction.

    ``pairs_by_direction`` maps a label (e.g. ``"zh->en"``) to its pair list.
    """
    skip = skip or set()
    out: dict = {"benchmark": "translation_performance", "model": model_cfg.name, "directions": {}}
    for label, pairs in pairs_by_direction.items():
        block: dict = {}
        if "ttft" not in skip:
            block["ttft"] = run_translation_ttft(model_cfg, pairs, samples=ttft_samples)
        if "throughput" not in skip:
            block["throughput"] = run_translation_throughput(
                model_cfg, pairs, duration_s=throughput_duration_s
            )
        out["directions"][label] = block
    return out
