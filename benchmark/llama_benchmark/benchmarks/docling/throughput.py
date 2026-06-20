"""文档解析吞吐量测试：pages/s @并发1/4/8 + 峰值内存。

主要用于硬件选型：
  - 对比 CPU-only vs GPU 的吞吐差距
  - 验证多核 CPU 并发扩展性（ProcessPool）
  - 量化 VRAM 占用（通过 pynvml 采样）
"""

from __future__ import annotations

import os
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Dict, List, Optional

from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    TaskResult,
)
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.system_info import get_system_info

logger = get_logger(__name__)


def run_throughput(
    backend,
    config: BenchmarkTaskConfig,
    model_name: str,
    document_paths: Optional[List[str]] = None,
) -> TaskResult:
    """执行文档解析吞吐量测试。

    Args:
        document_paths: 测试文档路径列表。为 None 时使用合成测试（生成临时 PDF）。
    """
    start_time = time.time()

    # 准备测试文档
    if not document_paths:
        document_paths = _generate_synthetic_pdfs(
            n=config.num_samples or 10,
            pages_per_doc=config.extra.get("pages_per_doc", 5),
        )
        _cleanup_after = True
    else:
        _cleanup_after = False

    num_docs = len(document_paths)
    concurrency_levels = config.extra.get("concurrency_levels", [1, 4, 8])

    metrics: List[MetricResult] = []
    throughput_by_concurrency: Dict[int, float] = {}

    # 峰值内存基线（测试前）
    mem_before = _get_memory_mb()
    vram_before = _get_vram_mb()

    for concurrency in concurrency_levels:
        tps = _measure_throughput(backend, document_paths, concurrency)
        throughput_by_concurrency[concurrency] = tps
        logger.info(f"[{model_name}] 并发={concurrency}: {tps:.2f} pages/s")

        metrics.append(
            MetricResult(
                name=f"throughput_pages_per_sec_c{concurrency}",
                value=round(tps, 3),
                unit="pages/s",
                higher_is_better=True,
                metadata={"concurrency": concurrency},
            )
        )

    # 峰值内存增量（测试后 - 测试前）
    mem_after = _get_memory_mb()
    vram_after = _get_vram_mb()
    peak_rss_mb = max(0.0, mem_after - mem_before)
    peak_vram_mb = max(0.0, vram_after - vram_before)

    metrics.extend([
        MetricResult(
            name="peak_rss_delta_mb",
            value=round(peak_rss_mb, 1),
            unit="MB",
            higher_is_better=False,
        ),
        MetricResult(
            name="peak_vram_delta_mb",
            value=round(peak_vram_mb, 1),
            unit="MB",
            higher_is_better=False,
        ),
    ])

    # 并发扩展效率（最高并发 / 单线程的比值，理想=concurrency倍）
    if 1 in throughput_by_concurrency and len(concurrency_levels) > 1:
        max_c = max(concurrency_levels)
        if throughput_by_concurrency[1] > 0:
            scale_efficiency = (
                throughput_by_concurrency.get(max_c, 0)
                / throughput_by_concurrency[1]
                / max_c
            )
            metrics.append(
                MetricResult(
                    name="concurrency_scale_efficiency",
                    value=round(scale_efficiency, 3),
                    unit="",
                    higher_is_better=True,
                    metadata={"base_concurrency": 1, "max_concurrency": max_c},
                )
            )

    # 判断是否通过阈值
    threshold = config.thresholds.get(
        "throughput_pages_per_sec", ThresholdConfig()
    )
    baseline_tps = throughput_by_concurrency.get(1, 0.0)
    status = BenchmarkStatus.PASS if threshold.check(baseline_tps) else BenchmarkStatus.FAIL

    # 清理合成文档
    if _cleanup_after:
        for p in document_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    return TaskResult(
        task_name="doc_throughput",
        model_name=model_name,
        metrics=metrics,
        num_samples=num_docs,
        duration_seconds=time.time() - start_time,
        status=status,
        metadata={
            "concurrency_levels": concurrency_levels,
            "num_docs": num_docs,
            "system_info": get_system_info(),
        },
    )


def _measure_throughput(backend, document_paths: List[str], concurrency: int) -> float:
    """测量指定并发度下的吞吐量（pages/s）。

    使用线程池（因为 parse() 通常是 I/O + CPU 混合，且各 backend 可能持有 GIL）。
    若 backend 支持多进程，可在 config.extra 中设置 use_process_pool=true。
    """
    from concurrent.futures import ThreadPoolExecutor

    total_pages = 0
    errors = 0
    start = time.perf_counter_ns()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(backend.parse, p): p for p in document_paths}
        for future in as_completed(futures):
            try:
                result = future.result()
                pages = result.get("metadata", {}).get("num_pages", 1)
                total_pages += max(1, pages)
            except Exception as e:
                errors += 1
                logger.debug(f"parse 失败: {e}")

    elapsed_s = (time.perf_counter_ns() - start) / 1e9
    if elapsed_s <= 0 or total_pages <= 0:
        return 0.0

    logger.debug(
        f"并发={concurrency}: {total_pages} pages / {elapsed_s:.2f}s = "
        f"{total_pages/elapsed_s:.2f} pages/s (errors={errors})"
    )
    return total_pages / elapsed_s


def _get_memory_mb() -> float:
    """获取当前进程 RSS 内存（MB）。"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _get_vram_mb() -> float:
    """获取当前 GPU VRAM 使用量（MB），不可用时返回 0。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return 0.0


def _generate_synthetic_pdfs(n: int, pages_per_doc: int = 5) -> List[str]:
    """生成合成 PDF 用于吞吐测试（使用 reportlab 或 fpdf2）。

    若无 PDF 生成库，生成包含文本的临时文件。
    """
    import tempfile

    paths = []
    tmp_dir = tempfile.mkdtemp(prefix="llama_bench_throughput_")

    for i in range(n):
        pdf_path = Path(tmp_dir) / f"synthetic_{i:04d}.pdf"
        _write_synthetic_pdf(pdf_path, pages_per_doc, doc_id=i)
        if pdf_path.exists():
            paths.append(str(pdf_path))

    return paths


def _write_synthetic_pdf(output_path: Path, num_pages: int, doc_id: int) -> None:
    """写入合成 PDF，优先使用 reportlab，降级到 fpdf2，最终降级到纯文本。"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4

        c = canvas.Canvas(str(output_path), pagesize=A4)
        for p in range(num_pages):
            c.drawString(72, 750, f"Document {doc_id} - Page {p + 1}")
            c.drawString(72, 700,
                "This is a synthetic test document for throughput benchmarking. "
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
            c.showPage()
        c.save()
        return
    except ImportError:
        pass

    try:
        from fpdf import FPDF

        pdf = FPDF()
        for p in range(num_pages):
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 10, f"Document {doc_id} - Page {p + 1}", ln=True)
            pdf.multi_cell(0, 8,
                "This is a synthetic test document for throughput benchmarking.")
        pdf.output(str(output_path))
        return
    except ImportError:
        pass

    # 最终降级：写纯文本文件（PyMuPDF 等可处理，但 AI 解析器可能跳过）
    with open(output_path.with_suffix(".txt"), "w") as f:
        for p in range(num_pages):
            f.write(f"Document {doc_id} - Page {p + 1}\n")
            f.write("Synthetic test content for throughput benchmarking.\n\n")
