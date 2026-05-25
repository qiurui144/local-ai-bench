"""Docling 解析准确率 benchmark：字段级 F1 + 表格 cell-level F1。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    SampleResult,
    TaskResult,
)
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


def _cell_f1(pred_cells: List[Dict], ref_cells: List[Dict]) -> float:
    """计算表格 cell-level F1。

    每个 cell 用 (row, col, normalized_text) 作为唯一键。
    """

    def normalize(text: str) -> str:
        return text.strip().lower()

    pred_set: Set[tuple] = {
        (c.get("row", 0), c.get("col", 0), normalize(c.get("text", "")))
        for c in pred_cells
    }
    ref_set: Set[tuple] = {
        (c.get("row", 0), c.get("col", 0), normalize(c.get("text", "")))
        for c in ref_cells
    }

    if not pred_set and not ref_set:
        return 1.0

    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _field_f1(pred: Dict[str, Any], ref: Dict[str, Any]) -> float:
    """计算结构化字段级 F1（文本字段精确匹配）。"""
    def extract_fields(d: Dict, prefix: str = "") -> Set[str]:
        fields = set()
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, str):
                fields.add(f"{key}={v.strip().lower()}")
            elif isinstance(v, (int, float)):
                fields.add(f"{key}={v}")
        return fields

    pred_fields = extract_fields(pred)
    ref_fields = extract_fields(ref)

    if not pred_fields and not ref_fields:
        return 1.0
    tp = len(pred_fields & ref_fields)
    precision = tp / len(pred_fields) if pred_fields else 0.0
    recall = tp / len(ref_fields) if ref_fields else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_parse_accuracy(
    backend,
    config: BenchmarkTaskConfig,
    model_name: str,
) -> TaskResult:
    """执行文档解析准确率评测。

    数据集格式：dataset_path 下每个子目录包含：
    - document.pdf（或 .docx）
    - annotation.json（标注的结构化结果）
    """
    start_time = time.time()

    if not config.dataset_path or not Path(config.dataset_path).exists():
        logger.warning("Docling 测试集路径未配置或不存在，跳过")
        return TaskResult(
            task_name="parse_accuracy",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=0.0,
            status=BenchmarkStatus.SKIP,
            error_message="数据集路径未配置",
        )

    dataset_path = Path(config.dataset_path)
    test_cases = _find_test_cases(dataset_path)
    if config.num_samples:
        test_cases = test_cases[: config.num_samples]

    if not test_cases:
        return TaskResult(
            task_name="parse_accuracy",
            model_name=model_name,
            metrics=[],
            num_samples=0,
            duration_seconds=0.0,
            status=BenchmarkStatus.SKIP,
            error_message="未找到测试样本",
        )

    field_f1_scores: List[float] = []
    table_f1_scores: List[float] = []
    sample_results: List[SampleResult] = []

    for doc_path, annotation_path in test_cases:
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        infer_start = time.perf_counter_ns()
        try:
            parsed = backend.parse(str(doc_path))
        except Exception as e:
            logger.warning(f"解析失败 {doc_path}: {e}")
            parsed = {"text": "", "tables": [], "metadata": {}}
        latency_ms = (time.perf_counter_ns() - infer_start) / 1_000_000

        # 字段 F1
        ref_fields = annotation.get("fields", {})
        pred_fields = {"text_preview": parsed["text"][:200]}
        ff1 = _field_f1(pred_fields, ref_fields) if ref_fields else 1.0
        field_f1_scores.append(ff1)

        # 表格 F1
        ref_tables = annotation.get("tables", [])
        pred_tables = parsed.get("tables", [])
        if ref_tables:
            tf1_scores = []
            for rt in ref_tables:
                best = 0.0
                for pt in pred_tables:
                    best = max(best, _cell_f1(pt.get("cells", []), rt.get("cells", [])))
                tf1_scores.append(best)
            tf1 = sum(tf1_scores) / len(tf1_scores)
        else:
            tf1 = 1.0
        table_f1_scores.append(tf1)

        sample_results.append(
            SampleResult(
                sample_id=doc_path.name,
                input=str(doc_path),
                expected=str(annotation_path),
                predicted=parsed["text"][:100],
                correct=ff1 >= 0.8,
                latency_ms=latency_ms,
                metadata={"field_f1": ff1, "table_f1": tf1},
            )
        )

    avg_field_f1 = sum(field_f1_scores) / len(field_f1_scores) if field_f1_scores else 0.0
    avg_table_f1 = sum(table_f1_scores) / len(table_f1_scores) if table_f1_scores else 0.0

    field_threshold = config.thresholds.get("field_f1", ThresholdConfig())
    table_threshold = config.thresholds.get("cell_f1", ThresholdConfig())

    field_status = BenchmarkStatus.PASS if field_threshold.check(avg_field_f1) else BenchmarkStatus.FAIL
    table_status = BenchmarkStatus.PASS if table_threshold.check(avg_table_f1) else BenchmarkStatus.FAIL
    overall = (
        BenchmarkStatus.FAIL
        if BenchmarkStatus.FAIL in (field_status, table_status)
        else BenchmarkStatus.PASS
    )

    return TaskResult(
        task_name="parse_accuracy",
        model_name=model_name,
        metrics=[
            MetricResult(
                name="field_f1",
                value=round(avg_field_f1, 4),
                higher_is_better=True,
                threshold=field_threshold.min_value,
                status=field_status,
            ),
            MetricResult(
                name="table_cell_f1",
                value=round(avg_table_f1, 4),
                higher_is_better=True,
                threshold=table_threshold.min_value,
                status=table_status,
            ),
        ],
        num_samples=len(test_cases),
        duration_seconds=time.time() - start_time,
        status=overall,
        sample_results=sample_results,
    )


def _find_test_cases(dataset_path: Path) -> List[Tuple[Path, Path]]:
    """扫描数据集目录，返回 (document_path, annotation_path) 对。"""
    cases = []
    for item_dir in sorted(dataset_path.iterdir()):
        if not item_dir.is_dir():
            continue
        annotation = item_dir / "annotation.json"
        if not annotation.exists():
            continue
        for ext in ("*.pdf", "*.docx", "*.pptx"):
            docs = list(item_dir.glob(ext))
            if docs:
                cases.append((docs[0], annotation))
                break
    return cases
