"""translation 维度编排(质量 L1/L3 × 方向 + 延迟),自 run_benchmark 下沉。"""
from __future__ import annotations

from pathlib import Path

from benchmark.translation.accuracy import run_translation
from benchmark.translation.datasets import load_custom_jsonl, load_flores
from benchmark.translation.performance import run_translation_performance


def run_translation_dimension(model_cfg, tr_cfg: dict, root: Path) -> dict:
    """对单模型跑翻译质量（每方向 × L1/L2/L3）+ 每方向延迟。"""
    flores = tr_cfg.get("flores", {})
    num_samples = flores.get("num_samples", 100)
    split = flores.get("split", "devtest")
    thresholds = tr_cfg.get("thresholds", {})
    run_comet = tr_cfg.get("run_comet", True)
    custom_path = root / tr_cfg.get("custom_corpus", "datasets/translation/custom_zh_en.jsonl")

    out: dict = {"benchmark": "translation", "model": model_cfg.name,
                 "directions": {}, "dataset_sources": {},
                 "verdict": "PASS", "verdict_reasons": []}
    pairs_by_dir: dict[str, list] = {}

    for direction in tr_cfg.get("directions", ["zh->en", "en->zh"]):
        src_lang, tgt_lang = direction.split("->")
        flores_pairs = load_flores(src_lang, tgt_lang, split=split, num_samples=num_samples)
        pairs_by_dir[direction] = flores_pairs

        # 数据 provenance 必须进报告:builtin 合成 fallback 的分数不能冒充
        # Flores-200 PASS(per §6.3 baseline 诚信)。
        srcs = sorted({p.source for p in flores_pairs})
        out["dataset_sources"][direction] = srcs
        if "builtin" in srcs:
            if out["verdict"] == "PASS":
                out["verdict"] = "WARN"
            out["verdict_reasons"].append(
                f"[{direction}] L1 scored on builtin synthetic fallback "
                f"({len(flores_pairs)} pairs), NOT Flores-200"
            )

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
