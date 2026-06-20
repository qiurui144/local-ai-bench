"""单模型报告的 per-dimension markdown 节(自 run_benchmark.render_markdown 下沉)。

每个 `render_<dim>(block) -> list[str]` 对空 block 的行为与原实现逐字一致:
accuracy/ttft/throughput/concurrency/stability 头部恒输出,其余整节条件输出;
accuracy 不带前导空行(原实现位于头部 lines 列表尾),其余节带。
"""
from __future__ import annotations


def render_accuracy(acc: dict) -> list[str]:
    lines = ["## 准确性", ""]
    if acc.get("skipped"):
        lines.append(f"- SKIPPED: {acc.get('reason')}")
    elif acc.get("status") == "blocked":
        lines.append(f"- BLOCKED: {acc.get('reason')}")
        missing = acc.get("missing_images") or []
        if missing:
            lines.append(f"- Missing fixture images: {', '.join(str(x) for x in missing[:12])}")
    elif acc:
        agg = acc.get("aggregate", {})
        lines += [
            f"- **判定**: {acc.get('verdict','?')}",
            f"- 分类 precision: {agg.get('category_precision',0)*100:.1f}%",
            f"- 实体 recall: {agg.get('entity_recall',0)*100:.1f}%",
            f"- 事实 recall: {agg.get('fact_recall',0)*100:.1f}%",
            f"- must_not_say 违反: {agg.get('must_not_say_violations',0)}",
            f"- 输出 token 平均: {agg.get('output_tokens_stats',{}).get('avg',0):.0f}",
            f"- 截断率: {agg.get('truncation_rate',0)*100:.1f}%",
            f"- 错误率: {agg.get('error_rate',0)*100:.1f}%",
        ]
        for r in acc.get("verdict_reasons", []):
            lines.append(f"  - {r}")
    return lines


def render_ttft(ttft: dict) -> list[str]:
    if not ttft:
        return []
    s = ttft.get("ttft_ms_stats", {})
    return [
        "", "## 首 Token (TTFT)", "",
        f"- P50 TTFT: {s.get('p50',0):.0f} ms",
        f"- P95 TTFT: {s.get('p95',0):.0f} ms",
        f"- 样本数: {ttft.get('samples',0)}  错误率: {ttft.get('error_rate',0)*100:.1f}%",
    ]


def render_throughput(tp: dict) -> list[str]:
    if not tp:
        return []
    return [
        "", "## 吞吐量", "",
        f"- 聚合 TPS: **{tp.get('aggregate_tps',0):.1f}** tokens/s",
        f"- 每请求 P50 TPS: {tp.get('per_request_tps_stats',{}).get('p50',0):.1f}",
        f"- 请求数: {tp.get('requests',0)}  错误: {tp.get('errors',0)}",
        f"- 总输入 tokens: {tp.get('total_input_tokens',0)}",
        f"- 总输出 tokens: {tp.get('total_output_tokens',0)}",
    ]


def render_prefill_decode(pd: dict) -> list[str]:
    lines: list[str] = []
    if pd:
        pp = pd.get("prefill", {})
        tg = pd.get("decode", {})
        lines += [
            "", "## PP / TG 分阶段吞吐", "",
            f"- **PP (prefill) tok/s**: {pp.get('tok_per_sec',{}).get('p50',0):.1f} "
            f"(mean {pp.get('tok_per_sec',{}).get('mean',0):.1f}, "
            f"avg prompt {pp.get('avg_prompt_tokens',0):.0f} tok)",
            f"- **TG (decode) tok/s**: {tg.get('tok_per_sec',{}).get('p50',0):.1f} "
            f"(mean {tg.get('tok_per_sec',{}).get('mean',0):.1f}, "
            f"avg decode {tg.get('avg_decode_tokens',0):.0f} tok)",
            f"- 测得样本: {pd.get('measured',0)}/{pd.get('samples',0)}  "
            f"无 usage 跳过: {pd.get('no_usage_samples',0)}  错误: {pd.get('errors',0)}",
        ]
    return lines


def render_concurrency(con: dict) -> list[str]:
    if not con or not con.get("steps"):
        return []
    lines = ["", "## 并发稳定性", "",
             "| 并发 | 成功率 | P50 ms | P95 ms | 聚合 TPS |", "|---|---|---|---|---|"]
    for step in con["steps"]:
        s = step.get("latency_stats_ms", {})
        lines.append(
            f"| {step['concurrency']} "
            f"| {step['success_rate']*100:.1f}% "
            f"| {s.get('p50',0):.0f} | {s.get('p95',0):.0f} "
            f"| {step.get('aggregate_tps',0):.1f} |"
        )
    return lines


def render_stability(stab: dict) -> list[str]:
    if not stab:
        return []
    return [
        "", "## 长时间稳定性（30 min）", "",
        f"- 判定: {stab.get('drift_verdict','?')}",
        f"- 前 5min P95: {stab.get('first_5min_p95_ms',0):.0f} ms",
        f"- 最后 5min P95: {stab.get('last_5min_p95_ms',0):.0f} ms",
        f"- 漂移比: {stab.get('latency_drift_ratio',1):.2f}×",
        f"- 错误率: {stab.get('error_rate',0)*100:.1f}%",
        f"- 样本数: {stab.get('total_samples',0)}",
    ]


def render_translation(tr: dict) -> list[str]:
    lines: list[str] = []
    if tr:
        lines += ["", "## 翻译（zh<->en）", "", f"- 判定: {tr.get('verdict','?')}", ""]
        lines += ["| 方向 | 任务 | BLEU | chrF | COMET | 术语率 |",
                  "|---|---|---|---|---|---|"]
        for direction, blocks in tr.get("directions", {}).items():
            for task, block in blocks.items():
                agg = block.get("aggregate", {})
                comet = agg.get("comet", {}) or {}
                comet_s = f"{comet['score']:.3f}" if comet.get("available") else "skip(GPU)"
                terms = agg.get("terminology") or {}
                term_s = f"{terms['term_match_rate']*100:.0f}%" if terms else "-"
                lines.append(
                    f"| {direction} | {task} | {agg.get('bleu',0):.1f} "
                    f"| {agg.get('chrf',0):.1f} | {comet_s} | {term_s} |"
                )
        for r in tr.get("verdict_reasons", []):
            lines.append(f"  - {r}")
    return lines


def render_embedding(emb: dict) -> list[str]:
    lines: list[str] = []
    if emb and not emb.get("skipped"):
        agg = emb.get("aggregate", {})
        val = agg.get("validation", {})
        perf = emb.get("performance", {})
        lat = perf.get("latency", {}).get("single_query_latency_ms_stats", {})
        mem = perf.get("memory", {})
        lines += [
            "", "## Embedding 检索", "",
            f"- **判定**: {emb.get('verdict','?')}",
            f"- hit@1: {agg.get('hit@1',0):.3f}",
            f"- recall@1 / @5 / @10: {agg.get('recall@1',0):.3f} / "
            f"{agg.get('recall@5',0):.3f} / {agg.get('recall@10',0):.3f}",
            f"- MRR: {agg.get('mrr',0):.4f}  nDCG@10: {agg.get('ndcg@10',0):.4f}",
            f"- 单条 embed 延迟 P50 (常驻): {lat.get('p50',0):.1f} ms",
            f"- 数值校验: {'OK' if val.get('ok') else 'FAIL'} "
            f"(zero={val.get('zero_vectors',0)} nan={val.get('nan_vectors',0)} "
            f"inf={val.get('inf_vectors',0)} dim={val.get('dim',0)})",
        ]
        if mem.get("available"):
            lines.append(
                f"- RSS: 常驻查询 {mem.get('resident_query_rss_mb',0):.0f} MB / "
                f"批量 {mem.get('batch_rss_mb',0):.0f} MB"
            )
        else:
            lines.append(f"- RSS: 未采集 ({mem.get('reason','')})")
        for r in emb.get("verdict_reasons", []):
            lines.append(f"  - {r}")
    return lines


def render_rerank(rr: dict) -> list[str]:
    lines: list[str] = []
    if rr and not rr.get("skipped"):
        agg = rr.get("aggregate", {})
        sep = agg.get("score_separation", {})
        plat = agg.get("single_pair_latency_ms_stats", {})
        lines += [
            "", "## Reranker 重排", "",
            f"- **判定**: {rr.get('verdict','?')}",
            f"- nDCG@10: {agg.get('ndcg@10',0):.4f}  MRR: {agg.get('mrr',0):.4f}",
            f"- recall@1 / @5: {agg.get('recall@1',0):.3f} / {agg.get('recall@5',0):.3f}",
            f"- 单 pair 延迟 P50: {plat.get('p50',0):.0f} ms  (pairs={agg.get('num_pairs',0)})",
            f"- 分数分离: pos {sep.get('pos_mean',0):.2f} vs neg {sep.get('neg_mean',0):.2f}",
        ]
        for r in rr.get("verdict_reasons", []):
            lines.append(f"  - {r}")
    return lines


def render_asr(asr: dict) -> list[str]:
    lines: list[str] = []
    if asr:
        lines += ["", "## ASR (CER/WER/RTF)", ""]
        if asr.get("status") == "blocked":
            lines.append(f"- BLOCKED: {asr.get('reason')}")
        else:
            agg = asr.get("aggregate", {})
            lines += [
                f"- **判定**: {asr.get('verdict','?')}",
                f"- CER: {agg.get('cer',0)*100:.2f}%  WER: {agg.get('wer',0)*100:.2f}%",
                f"- RTF mean: {agg.get('rtf_mean',0):.4f}",
                f"- 样本: {agg.get('num_samples',0)}  音频时长: {agg.get('total_audio_s',0):.1f} s",
                f"- 空输出: {agg.get('empty_output_count',0)}  错误: {agg.get('error_count',0)}",
            ]
            for r in asr.get("verdict_reasons", []):
                lines.append(f"  - {r}")
    return lines


def render_ocr(ocr: dict) -> list[str]:
    lines: list[str] = []
    if ocr:
        lines += ["", "## OCR (CER/NED/延迟)", ""]
        if ocr.get("status") == "blocked":
            lines.append(f"- BLOCKED: {ocr.get('reason')}")
        else:
            agg = ocr.get("aggregate", {})
            lat = agg.get("latency_ms_stats", {})
            lines += [
                f"- **判定**: {ocr.get('verdict', '?')}",
                f"- CER: {agg.get('cer', 0)*100:.2f}%  NED: {agg.get('ned', 0)*100:.2f}%",
                f"- 延迟 P50/P95: {lat.get('p50', 0):.0f} / {lat.get('p95', 0):.0f} ms",
                f"- 样本: {agg.get('num_samples', 0)}  后端: {agg.get('backend', '?')}",
                f"- 空输出: {agg.get('empty_output_count', 0)}  错误: {agg.get('error_count', 0)}",
            ]
            for r in ocr.get("verdict_reasons", []):
                lines.append(f"  - {r}")
    return lines


def render_general_ability(ga: dict) -> list[str]:
    if not ga:
        return []
    lines = ["", "## 通用能力 (gsm8k/mmlu/hellaswag)", "",
             f"- **判定**: {ga.get('verdict','?')}"]
    for name, t in (ga.get("tasks") or {}).items():
        if t.get("verdict") == "BLOCKED":
            lines.append(f"- **{name}**: BLOCKED — {t.get('reason')}")
        else:
            lines.append(f"- **{name}**: {t.get('verdict')}  accuracy "
                         f"{t.get('accuracy',0):.3f} (n={t.get('n',0)}, "
                         f"errors={t.get('errors',0)})")
    for r in ga.get("verdict_reasons", []):
        lines.append(f"  - {r}")
    return lines


def render_conditioned(cd: dict) -> list[str]:
    if not cd:
        return []
    lines = ["", "## 条件化能力 (context ladder / cache A/B)", "",
             f"- **判定**: {cd.get('verdict','?')}"]
    ladder = cd.get("context_ladder") or {}
    if ladder:
        lines += ["", "| 档位 | task_acc | needle_recall | TTFT ms | TPS | errors |",
                  "|---|---|---|---|---|---|"]
        for label, b in ladder.items():
            if b.get("verdict") == "SKIPPED":
                lines.append(f"| {label} | SKIPPED ({b.get('reason')}) | - | - | - | - |")
            else:
                lines.append(f"| {label} | {b.get('task_accuracy',0):.2f} "
                             f"| {b.get('needle_recall',0):.2f} | {b.get('ttft_ms',0):.0f} "
                             f"| {b.get('tps',0):.1f} | {b.get('errors',0)} |")
    qd = cd.get("quality_degradation")
    if qd:
        lines.append(f"- 质量衰减: {qd['from']}→{qd['to']} drop {qd['drop']}")
    cache = cd.get("cache") or {}
    if cache and not cache.get("error"):
        lines.append(f"- 缓存冷热: cold {cache.get('ttft_cold_ms',0):.0f}ms / warm "
                     f"{cache.get('ttft_warm_ms',0):.0f}ms = {cache.get('speedup',0)}× "
                     f"一致性 {'OK' if cache.get('output_consistent') else 'FAIL'}")
    for r in cd.get("verdict_reasons", []):
        lines.append(f"  - {r}")
    return lines


def render_scenarios(sc: dict) -> list[str]:
    lines: list[str] = []
    if sc:
        lines += ["", "## 真实场景 (S1/S2/S3)", ""]
        lines.append(f"- **判定**: {sc.get('verdict','?')}  judge: {sc.get('judge_model')}")
        cal = sc.get("judge_calibration") or {}
        if cal:
            lines.append(f"- judge 校准: agreement {cal.get('anchor_agreement',0):.2f} "
                         f"({'PASS' if cal.get('passed') else 'FAIL'})")
        for name, blk in (sc.get("scenarios") or {}).items():
            v = blk.get("verdict", "?")
            lines.append(f"- **{name}**: {v}")
            if blk.get("l1"):
                l1 = "  ".join(f"{k}={val:.2f}" for k, val in blk["l1"].items())
                lines.append(f"  - L1: {l1}")
            if blk.get("l2_judge"):
                j = blk["l2_judge"]
                lines.append(f"  - L2 judge: {j.get('mean',0):.2f} ± {j.get('std',0):.2f} (N={j.get('seeds')})")
            if blk.get("provenance"):
                lines.append(f"  - provenance: {blk['provenance']}")
            for r in blk.get("verdict_reasons", []):
                lines.append(f"  - {r}")
        for r in sc.get("verdict_reasons", []):
            lines.append(f"  - {r}")
    return lines


def render_conversation_drift(cd: dict) -> list[str]:
    lines: list[str] = []
    if cd:
        lines += ["", "## 对话漂移 (conversation_drift)", ""]
        overall = cd.get("overall_drift_verdict", "?")
        positions = cd.get("positions_tested", [])
        lines.append(f"- **整体漂移判定**: {overall}  verdict: {cd.get('verdict','?')}")
        lines.append(f"- 测试位置(prior turns): {positions}")
        for name, blk in (cd.get("per_scenario") or {}).items():
            v = blk.get("verdict", "?")
            drop = blk.get("max_quality_drop")
            metric = blk.get("primary_metric", "")
            drop_str = f"  max_drop={drop:.4f}" if drop is not None else ""
            lines.append(f"- **{name}**: {v}{drop_str}  ({metric})")
            qbp = blk.get("quality_by_position") or {}
            if qbp:
                pos_str = "  ".join(f"@{p}={v}" for p, v in qbp.items() if v is not None)
                if pos_str:
                    lines.append(f"  - 质量@位置: {pos_str}")
            for r in blk.get("verdict_reasons", []):
                lines.append(f"  - {r}")
        for r in cd.get("verdict_reasons", []):
            lines.append(f"- {r}")
    return lines
