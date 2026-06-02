"""主 benchmark 入口 —— 对 models.yaml 中声明的模型跑完整测试套件

使用：
  python run_benchmark.py --model qwen3-vl-8b-instruct
  python run_benchmark.py --model all              # 跑所有已启动的模型
  python run_benchmark.py --model qwen3-vl-8b-instruct \\
      --skip stability                             # 跳过 30 分钟稳定性

输出：
  output/reports/{model}_{timestamp}.json       # 机器可读
  output/reports/{model}_{timestamp}.md          # 人类可读
  output/reports/matrix_{timestamp}.md           # 所有模型对比表（all 模式）

退出码：
  0 全部 PASS
  1 有 WARN
  2 任一模型 FAIL
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from common import (
    ModelConfig,
    get_vram_info,
    load_benchmarks_config,
    load_models,
    wait_model_ready,
)

sys.path.insert(0, str(Path(__file__).parent / "benchmark"))
from benchmark.accuracy import run_accuracy
from benchmark.performance import (
    run_concurrency,
    run_prefill_decode,
    run_stability,
    run_throughput,
    run_ttft,
)
from benchmark.translation.accuracy import run_translation
from benchmark.translation.datasets import load_custom_jsonl, load_flores
from benchmark.translation.performance import run_translation_performance
from benchmark.embedding.accuracy import run_embedding
from benchmark.embedding.datasets import load_retrieval
from benchmark.embedding.performance import run_embedding_performance
from benchmark.rerank.accuracy import run_rerank
from benchmark.asr.runner import run_asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
GOLDEN = ROOT / "golden" / "expectations.json"
REPORTS = ROOT / "output" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def _default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def run_all_for_model(
    model_cfg: ModelConfig,
    golden: dict,
    skip: set[str],
    bench_cfg: dict | None = None,
) -> dict:
    """对单个模型跑全量 benchmark"""
    bench_cfg = bench_cfg if bench_cfg is not None else load_benchmarks_config(ROOT / "models.yaml")
    results: dict = {
        "model": model_cfg.name,
        "hf_repo": model_cfg.hf_repo,
        "quantization": model_cfg.quantization,
        "hardware_min": model_cfg.hardware_min,
        "timestamp": datetime.datetime.now().isoformat(),
        "vram_snapshot": get_vram_info(),
        "benchmarks": {},
    }

    # 预检：端点是否就绪
    logger.info("检查 %s (port %d) 就绪...", model_cfg.name, model_cfg.port)
    if not wait_model_ready(model_cfg, timeout_s=60.0):
        logger.error("  %s 未就绪，跳过", model_cfg.name)
        results["error"] = "model_not_ready"
        return results

    # 准确性
    if "accuracy" not in skip:
        logger.info("▶ accuracy")
        results["benchmarks"]["accuracy"] = run_accuracy(model_cfg, golden, FIXTURES)

    # TTFT
    if "ttft" not in skip:
        logger.info("▶ ttft")
        results["benchmarks"]["ttft"] = run_ttft(model_cfg, FIXTURES, samples=5)

    # 吞吐
    if "throughput" not in skip:
        logger.info("▶ throughput")
        results["benchmarks"]["throughput"] = run_throughput(
            model_cfg, FIXTURES, duration_s=60.0
        )

    # PP / TG 分阶段吞吐（prefill vs decode tok/s，llama-bench 法）
    if "prefill_decode" not in skip:
        logger.info("▶ prefill_decode (PP/TG)")
        results["benchmarks"]["prefill_decode"] = run_prefill_decode(
            model_cfg, FIXTURES, samples=5, decode_tokens=128
        )

    # 并发
    if "concurrency" not in skip:
        logger.info("▶ concurrency")
        results["benchmarks"]["concurrency"] = run_concurrency(
            model_cfg, FIXTURES,
            concurrencies=[1, 5, 10, 30, 50],
            duration_s=60.0,
        )

    # 稳定性（默认 30min 太长，加 skip 开关）
    if "stability" not in skip:
        logger.info("▶ stability (30 min)")
        results["benchmarks"]["stability"] = run_stability(
            model_cfg, FIXTURES, duration_s=1800.0, sample_interval_s=5.0
        )

    # 翻译（zh<->en，L1/L2/L3 + 延迟）。仅 translation_capable 模型；用 hint 字段判定。
    tr_cfg = bench_cfg.get("translation", {})
    if "translation" not in skip and _is_translation_capable(model_cfg):
        logger.info("▶ translation (zh<->en)")
        results["benchmarks"]["translation"] = _run_translation_dimension(model_cfg, tr_cfg)

    # Embedding 检索质量 + 延迟/内存。仅 embedding_capable 模型（用 hint 判定）。
    emb_cfg = bench_cfg.get("embedding", {})
    if "embedding" not in skip and _model_hint(model_cfg, "embedding_capable"):
        logger.info("▶ embedding (retrieval recall/MRR/nDCG)")
        results["benchmarks"]["embedding"] = _run_embedding_dimension(model_cfg, emb_cfg)

    # Reranker 检索重排质量 + 单 pair 延迟。仅 rerank_capable 模型。
    rr_cfg = bench_cfg.get("rerank", {})
    if "rerank" not in skip and _model_hint(model_cfg, "rerank_capable"):
        logger.info("▶ rerank (nDCG/MRR + per-pair latency)")
        results["benchmarks"]["rerank"] = _run_rerank_dimension(model_cfg, rr_cfg)

    # ASR (中文 CER/WER/RTF)。仅 asr_capable 模型；缺数据/后端则 graceful BLOCKED。
    asr_cfg = bench_cfg.get("asr", {})
    if "asr" not in skip and _model_hint(model_cfg, "asr_capable"):
        logger.info("▶ asr (CER/WER/RTF)")
        results["benchmarks"]["asr"] = _run_asr_dimension(model_cfg, asr_cfg)

    results["vram_after"] = get_vram_info()
    return results


def _model_hint(model_cfg: ModelConfig, key: str) -> bool:
    """读 models.yaml 上某个布尔 hint 字段（不在 ModelConfig 上，重读 yaml）。"""
    try:
        import yaml

        data = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
        for m in data.get("models", []):
            if m.get("name") == model_cfg.name:
                return bool(m.get(key, False))
    except Exception:
        pass
    return False


def _is_translation_capable(model_cfg: ModelConfig) -> bool:
    """读 models.yaml 的 translation_capable hint。"""
    return _model_hint(model_cfg, "translation_capable")


def _run_embedding_dimension(model_cfg: ModelConfig, emb_cfg: dict) -> dict:
    """Embedding 检索质量 + 延迟/内存。数据集缺失时回退内置合成检索集。"""
    corpus = emb_cfg.get("corpus", "datasets/retrieval/cmteb_zh_subset.jsonl")
    corpus_path = ROOT / corpus
    num_samples = emb_cfg.get("num_samples")
    thresholds = emb_cfg.get("thresholds")
    queries = load_retrieval(corpus_path if corpus_path.exists() else None,
                             num_samples=num_samples)
    out = run_embedding(model_cfg, queries, thresholds=thresholds)
    out["performance"] = run_embedding_performance(
        model_cfg, queries, samples=emb_cfg.get("latency_samples", 12)
    )
    return out


def _run_rerank_dimension(model_cfg: ModelConfig, rr_cfg: dict) -> dict:
    """Reranker 重排质量 + 单 pair 延迟。复用同一检索集。"""
    corpus = rr_cfg.get("corpus", "datasets/retrieval/cmteb_zh_subset.jsonl")
    corpus_path = ROOT / corpus
    num_samples = rr_cfg.get("num_samples")
    thresholds = rr_cfg.get("thresholds")
    queries = load_retrieval(corpus_path if corpus_path.exists() else None,
                             num_samples=num_samples)
    return run_rerank(model_cfg, queries, thresholds=thresholds)


def _run_asr_dimension(model_cfg: ModelConfig, asr_cfg: dict) -> dict:
    """ASR CER/WER/RTF。缺 manifest 或 onnx 后端时 graceful BLOCKED。"""
    manifest = asr_cfg.get("manifest", "datasets/asr/manifest.jsonl")
    manifest_path = ROOT / manifest
    return run_asr(
        model_cfg,
        manifest_path=manifest_path if manifest_path.exists() else None,
        audio_root=ROOT / asr_cfg.get("audio_root", "datasets/asr") if asr_cfg.get("audio_root") else None,
        asr_model_dir=asr_cfg.get("model_dir"),
        num_samples=asr_cfg.get("num_samples"),
        thresholds=asr_cfg.get("thresholds"),
    )


def _run_translation_dimension(model_cfg: ModelConfig, tr_cfg: dict) -> dict:
    """对单模型跑翻译质量（每方向 × L1/L2/L3）+ 每方向延迟。"""
    flores = tr_cfg.get("flores", {})
    num_samples = flores.get("num_samples", 100)
    split = flores.get("split", "devtest")
    thresholds = tr_cfg.get("thresholds", {})
    run_comet = tr_cfg.get("run_comet", True)
    custom_path = ROOT / tr_cfg.get("custom_corpus", "datasets/translation/custom_zh_en.jsonl")

    out: dict = {"benchmark": "translation", "model": model_cfg.name,
                 "directions": {}, "verdict": "PASS", "verdict_reasons": []}
    pairs_by_dir: dict[str, list] = {}

    for direction in tr_cfg.get("directions", ["zh->en", "en->zh"]):
        src_lang, tgt_lang = direction.split("->")
        flores_pairs = load_flores(src_lang, tgt_lang, split=split, num_samples=num_samples)
        pairs_by_dir[direction] = flores_pairs

        dir_block: dict = {}
        # L1 single-sentence (Flores)
        dir_block["l1_flores"] = run_translation(
            model_cfg, flores_pairs, level="l1", thresholds=thresholds, run_comet=run_comet
        )
        # L3 terminology (custom corpus filtered to this direction + has glossary)
        try:
            custom = [p for p in load_custom_jsonl(custom_path)
                      if p.src_lang == src_lang and p.tgt_lang == tgt_lang and p.glossary]
        except Exception:
            custom = []
        if custom:
            dir_block["l3_terminology"] = run_translation(
                model_cfg, custom, level="l3", thresholds=thresholds, run_comet=False
            )
        out["directions"][direction] = dir_block

        for block in dir_block.values():
            if block.get("verdict") == "FAIL":
                out["verdict"] = "FAIL"
            elif block.get("verdict") == "WARN" and out["verdict"] != "FAIL":
                out["verdict"] = "WARN"
            out["verdict_reasons"] += [f"[{direction}] {r}" for r in block.get("verdict_reasons", [])]

    # latency per direction (TTFT + tok/s)
    out["performance"] = run_translation_performance(
        model_cfg, pairs_by_dir,
        ttft_samples=tr_cfg.get("ttft_samples", 5),
        throughput_duration_s=tr_cfg.get("throughput_duration_s", 60.0),
    )
    return out


def render_markdown(result: dict) -> str:
    """单模型报告 Markdown"""
    m = result["model"]
    bm = result.get("benchmarks", {})
    acc = bm.get("accuracy") or {}
    ttft = bm.get("ttft") or {}
    tp = bm.get("throughput") or {}
    con = bm.get("concurrency") or {}
    stab = bm.get("stability") or {}
    vram = result.get("vram_after", {})

    lines = [
        f"# {m} Benchmark",
        "",
        f"- HF repo: `{result.get('hf_repo','')}`",
        f"- Quantization: `{result.get('quantization')}`",
        f"- Hardware min: {result.get('hardware_min','')}",
        f"- Time: {result.get('timestamp','')}",
        f"- VRAM after run: {vram.get('used_mb','?')}MB / {vram.get('total_mb','?')}MB"
        if vram else "- VRAM: 未采集",
        "",
        "## 准确性",
        "",
    ]
    if acc.get("skipped"):
        lines.append(f"- SKIPPED: {acc.get('reason')}")
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
    lines += ["", "## 首 Token (TTFT)", ""]
    if ttft:
        s = ttft.get("ttft_ms_stats", {})
        lines += [
            f"- P50 TTFT: {s.get('p50',0):.0f} ms",
            f"- P95 TTFT: {s.get('p95',0):.0f} ms",
            f"- 样本数: {ttft.get('samples',0)}  错误率: {ttft.get('error_rate',0)*100:.1f}%",
        ]
    lines += ["", "## 吞吐量", ""]
    if tp:
        lines += [
            f"- 聚合 TPS: **{tp.get('aggregate_tps',0):.1f}** tokens/s",
            f"- 每请求 P50 TPS: {tp.get('per_request_tps_stats',{}).get('p50',0):.1f}",
            f"- 请求数: {tp.get('requests',0)}  错误: {tp.get('errors',0)}",
            f"- 总输入 tokens: {tp.get('total_input_tokens',0)}",
            f"- 总输出 tokens: {tp.get('total_output_tokens',0)}",
        ]
    pd = bm.get("prefill_decode") or {}
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
    lines += ["", "## 并发稳定性", ""]
    if con and con.get("steps"):
        lines += ["| 并发 | 成功率 | P50 ms | P95 ms | 聚合 TPS |", "|---|---|---|---|---|"]
        for step in con["steps"]:
            s = step.get("latency_stats_ms", {})
            lines.append(
                f"| {step['concurrency']} "
                f"| {step['success_rate']*100:.1f}% "
                f"| {s.get('p50',0):.0f} | {s.get('p95',0):.0f} "
                f"| {step.get('aggregate_tps',0):.1f} |"
            )
    lines += ["", "## 长时间稳定性（30 min）", ""]
    if stab:
        lines += [
            f"- 判定: {stab.get('drift_verdict','?')}",
            f"- 前 5min P95: {stab.get('first_5min_p95_ms',0):.0f} ms",
            f"- 最后 5min P95: {stab.get('last_5min_p95_ms',0):.0f} ms",
            f"- 漂移比: {stab.get('latency_drift_ratio',1):.2f}×",
            f"- 错误率: {stab.get('error_rate',0)*100:.1f}%",
            f"- 样本数: {stab.get('total_samples',0)}",
        ]

    tr = bm.get("translation") or {}
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

    emb = bm.get("embedding") or {}
    if emb and not emb.get("skipped"):
        agg = emb.get("aggregate", {})
        val = agg.get("validation", {})
        perf = emb.get("performance", {})
        lat = perf.get("latency", {}).get("single_query_latency_ms_stats", {})
        mem = perf.get("memory", {})
        lines += [
            "", "## Embedding 检索", "",
            f"- **判定**: {emb.get('verdict','?')}",
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

    rr = bm.get("rerank") or {}
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

    asr = bm.get("asr") or {}
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
    return "\n".join(lines)


def render_matrix(all_results: list[dict]) -> str:
    """多模型横向对比"""
    lines = [
        "# 模型矩阵对比",
        "",
        "| Model | 分类 precision | 实体 recall | TTFT P95 | TPS | 并发 50 成功率 | 稳定性漂移 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        name = r["model"]
        bm = r.get("benchmarks", {})
        acc = bm.get("accuracy", {}).get("aggregate", {}) if bm.get("accuracy") else {}
        ttft_p95 = bm.get("ttft", {}).get("ttft_ms_stats", {}).get("p95", 0)
        tps = bm.get("throughput", {}).get("aggregate_tps", 0)
        con_steps = bm.get("concurrency", {}).get("steps", [])
        con_50 = next((s for s in con_steps if s.get("concurrency") == 50), {})
        drift = bm.get("stability", {}).get("latency_drift_ratio", 0)

        lines.append(
            f"| {name} "
            f"| {acc.get('category_precision',0)*100:.1f}% "
            f"| {acc.get('entity_recall',0)*100:.1f}% "
            f"| {ttft_p95:.0f}ms "
            f"| {tps:.1f} "
            f"| {con_50.get('success_rate',0)*100:.1f}% "
            f"| {drift:.2f}× |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="模型名（见 models.yaml）或 all")
    parser.add_argument("--skip", default="",
                        help="逗号分隔跳过的 benchmark: accuracy,ttft,throughput,"
                             "prefill_decode,concurrency,stability,translation,"
                             "embedding,rerank,asr")
    parser.add_argument("--golden", default=str(GOLDEN))
    args = parser.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    bench_cfg = load_benchmarks_config(ROOT / "models.yaml")
    models = load_models(ROOT / "models.yaml")
    if args.model != "all":
        models = [m for m in models if m.name == args.model]
        if not models:
            logger.error("未知模型: %s。可选: %s", args.model,
                         [m.name for m in load_models(ROOT / 'models.yaml')])
            return 2

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results: list[dict] = []
    has_fail = False
    has_warn = False

    for m in models:
        logger.info("═══ %s ═══", m.name)
        result = run_all_for_model(m, golden, skip, bench_cfg)
        all_results.append(result)

        # 判定（质量维度任一 FAIL 即整体 FAIL）
        for dim in ("accuracy", "translation", "embedding", "rerank", "asr"):
            verdict = result.get("benchmarks", {}).get(dim, {}).get("verdict")
            if verdict == "FAIL":
                has_fail = True
            elif verdict == "WARN":
                has_warn = True

        # 保存单模型报告
        stem = f"{m.name}_{timestamp}"
        (REPORTS / f"{stem}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=_default),
            encoding="utf-8",
        )
        (REPORTS / f"{stem}.md").write_text(render_markdown(result), encoding="utf-8")
        logger.info("报告保存: %s", REPORTS / f"{stem}.md")

    # 矩阵报告
    if len(all_results) > 1:
        matrix_md = render_matrix(all_results)
        (REPORTS / f"matrix_{timestamp}.md").write_text(matrix_md, encoding="utf-8")
        print("\n" + matrix_md)

    return 2 if has_fail else (1 if has_warn else 0)


if __name__ == "__main__":
    sys.exit(main())
