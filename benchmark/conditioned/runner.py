"""conditioned 维度:上下文阶梯(质量+needle+性能)+ prefix-cache 冷热 A/B。

判定(spec §5):quality drop>0.20 WARN / >0.35 FAIL;最深档 needle_recall<0.5
WARN;缓存冷热输出不一致 FAIL(缓存改变答案是正确性 bug);全档 SKIPPED →
BLOCKED;单档全错记 error 不崩整跑。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

from common import infer_stream, infer_sync

from benchmark.conditioned.context_corpus import (
    build_context,
    load_cail_paragraphs,
    load_needles,
)

logger = logging.getLogger(__name__)

DEFAULT_LADDER = [1024, 4096, 8192, 16384, 32768]
ANSWER_MARGIN_TOKENS = 512
QUESTION_SUFFIX = "\n\n仅根据上文事实回答下列问题,只给出答案本身:\n问题:"


def _label(tokens: int) -> str:
    return f"{tokens // 1024}k"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def _effective_max_len(model_cfg):
    """models.yaml max_model_len 优先;否则探 vLLM /v1/models 的 max_model_len。"""
    if getattr(model_cfg, "max_model_len", None):
        return int(model_cfg.max_model_len)
    try:
        r = httpx.get(f"{model_cfg.base_url}/models", timeout=5.0)
        if r.status_code == 200:
            v = (r.json().get("data") or [{}])[0].get("max_model_len")
            return int(v) if v else None
    except Exception:
        pass
    return None


def _run_rung(model_cfg, target: int, probes: list[dict], paragraphs: list[str],
              max_tokens: int) -> dict:
    facts = [(p["id"], float(p["depth"]), p["fact"]) for p in probes]
    ctx = build_context(target, facts, paragraphs)
    correct = {"task": 0, "needle": 0}
    total = {"task": 0, "needle": 0}
    errors = 0
    ttft_ms = tps = 0.0
    prompt_tokens_actual = 0
    for k, p in enumerate(probes):
        prompt = ctx.text + QUESTION_SUFFIX + p["question"]
        fn = infer_stream if k == 0 else infer_sync   # 首题流式测 TTFT/TPS
        r = fn(model_cfg, prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        total[p["role"]] += 1
        if not r.ok:
            errors += 1
            continue
        if k == 0:
            ttft_ms, tps = r.ttft_ms, r.tokens_per_sec
            prompt_tokens_actual = r.input_tokens
        if _norm(p["answer"]) in _norm(r.content):
            correct[p["role"]] += 1
    return {
        "task_accuracy": round(correct["task"] / total["task"], 3) if total["task"] else 0.0,
        "needle_recall": round(correct["needle"] / total["needle"], 3) if total["needle"] else 0.0,
        "ttft_ms": round(ttft_ms, 1), "tps": round(tps, 1),
        "prompt_tokens_target": target,
        "prompt_tokens_actual": prompt_tokens_actual,
        "tokens_estimation": "approx(1.6 chars/token)",
        "errors": errors, "n": len(probes),
    }


def _run_cache_ab(model_cfg, paragraphs: list[str], max_tokens: int) -> dict:
    # 专用前缀:倒序段落 → 与阶梯前缀必然不同,第 1 次请求真 cold
    ctx = build_context(1024, [], list(reversed(paragraphs)))
    prompt = ctx.text + "\n\n请用一句话概括上文的主要内容。"
    runs = []
    for _ in range(2):
        r = infer_stream(model_cfg, prompt=prompt, max_tokens=max_tokens,
                         temperature=0.0, seed=0)
        if not r.ok:
            return {"error": r.error}
        runs.append(r)
    cold, warm = runs
    speedup = round(cold.ttft_ms / warm.ttft_ms, 2) if warm.ttft_ms > 0 else 0.0
    return {"ttft_cold_ms": round(cold.ttft_ms, 1), "ttft_warm_ms": round(warm.ttft_ms, 1),
            "speedup": speedup,
            "output_consistent": cold.content.strip() == warm.content.strip()}


def run_conditioned(model_cfg, cfg: dict, root: Path) -> dict:
    out = {"benchmark": "conditioned", "model": model_cfg.name,
           "context_ladder": {}, "cache": None,
           "verdict": "PASS", "verdict_reasons": []}
    probes = load_needles(root / (cfg.get("needles_file") or "datasets/conditioned/needles.jsonl"))
    if probes is None:
        out.update(verdict="BLOCKED",
                   verdict_reasons=["needles file missing — build datasets/conditioned/needles.jsonl"])
        return out
    try:
        paragraphs = load_cail_paragraphs()
    except Exception as e:
        out.update(verdict="BLOCKED",
                   verdict_reasons=[f"CAIL corpus load failed: {e} — offline? run scripts/prepare_offline.sh"])
        return out

    thr = cfg.get("thresholds") or {}
    drop_warn = thr.get("quality_drop_warn", 0.20)
    drop_fail = thr.get("quality_drop_fail", 0.35)
    recall_min = thr.get("needle_recall_min", 0.50)
    max_tokens = cfg.get("max_tokens", 64)
    ladder = cfg.get("context_ladder", DEFAULT_LADDER)
    margin = cfg.get("answer_margin_tokens", ANSWER_MARGIN_TOKENS)
    max_len = _effective_max_len(model_cfg)

    completed: list[tuple[int, dict]] = []
    for target in ladder:
        label = _label(target)
        if max_len is not None and target + margin > max_len:
            out["context_ladder"][label] = {
                "verdict": "SKIPPED", "reason": f"exceeds model max_len {max_len}"}
            continue
        rung = _run_rung(model_cfg, target, probes, paragraphs, max_tokens)
        out["context_ladder"][label] = rung
        if rung["errors"] < rung["n"]:
            completed.append((target, rung))
        else:
            out["verdict_reasons"].append(
                f"[{label}] all {rung['n']} requests errored — server rejected long input?")

    if ladder and not completed:
        skipped_all = all(b.get("verdict") == "SKIPPED"
                          for b in out["context_ladder"].values())
        out["verdict"] = "BLOCKED"
        out["verdict_reasons"].append(
            "no rung completed" + (" — all exceed model max_len" if skipped_all else ""))
        return out

    if len(completed) >= 2:
        lo, hi = completed[0], completed[-1]
        drop = round(lo[1]["task_accuracy"] - hi[1]["task_accuracy"], 3)
        out["quality_degradation"] = {"from": _label(lo[0]), "to": _label(hi[0]), "drop": drop}
        if drop > drop_fail:
            out["verdict"] = "FAIL"
            out["verdict_reasons"].append(f"quality drop {drop} > {drop_fail}")
        elif drop > drop_warn and out["verdict"] == "PASS":
            out["verdict"] = "WARN"
            out["verdict_reasons"].append(f"quality drop {drop} > {drop_warn}")
        # needle 与 quality-drop 是两个独立信号:理由必报,PASS→WARN 升级,
        # 已 WARN 不吞理由、已 FAIL 不降级。
        if hi[1]["needle_recall"] < recall_min:
            if out["verdict"] == "PASS":
                out["verdict"] = "WARN"
            out["verdict_reasons"].append(
                f"needle_recall {hi[1]['needle_recall']} @ {_label(hi[0])} < {recall_min}")

    cache = _run_cache_ab(model_cfg, paragraphs, max_tokens)
    out["cache"] = cache
    if cache.get("error"):
        # 冷热一致性不可证 → 至少 WARN(不降已有 FAIL);若连阶梯也零完成,
        # 则整跑零测量 → BLOCKED(空跑绝不 PASS)。
        if not completed:
            out["verdict"] = "BLOCKED"
            out["verdict_reasons"].append(
                f"empty run: no ladder rung completed and cache A/B errored: "
                f"{cache['error']} — nothing measured")
        else:
            if out["verdict"] == "PASS":
                out["verdict"] = "WARN"
            out["verdict_reasons"].append(
                f"cache A/B errored: {cache['error']} — "
                "cold/warm consistency unverifiable")
    elif not cache["output_consistent"]:
        out["verdict"] = "FAIL"
        out["verdict_reasons"].append(
            "cache warm output differs from cold — caching changed the answer (correctness bug)")
    elif cache["speedup"] < 1.2:
        out["verdict_reasons"].append(
            f"prefix-cache speedup {cache['speedup']}≈1 — check vLLM enable_prefix_caching")
    return out
