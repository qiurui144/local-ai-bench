"""CLI 入口：llama-bench 命令。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from benchmark.llama_benchmark.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="llama-bench",
    help="多模态 AI 模型自动化验收测试框架（LLM / Whisper / Embedding / Rerank / Docling）",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


@app.command("run")
def run_benchmarks(
    models_config: Path = typer.Option(
        Path("configs/models.yaml"),
        "--models", "-m",
        help="模型配置文件路径",
        exists=True,
    ),
    benchmarks_config: Path = typer.Option(
        Path("configs/benchmarks.yaml"),
        "--benchmarks", "-b",
        help="Benchmark 配置文件路径",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output", "-o",
        help="报告输出目录",
    ),
    formats: List[str] = typer.Option(
        ["json", "html", "markdown"],
        "--format", "-f",
        help="报告格式（可多次指定：-f json -f html）",
    ),
    model_filter: Optional[List[str]] = typer.Option(
        None, "--model",
        help="只运行指定模型名（可多次指定）",
    ),
    task_filter: Optional[List[str]] = typer.Option(
        None, "--task",
        help="只运行指定任务（如 mmlu,gsm8k）",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="仅验证配置，不执行 benchmark",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """运行模型验收测试并生成报告。"""
    configure_logging(log_level)

    from benchmark.llama_benchmark.core.config import AppConfig
    from benchmark.llama_benchmark.core.registry import create_runner
    from benchmark.llama_benchmark.core.result import BenchmarkSuiteResult, ModelBenchmarkResult
    from benchmark.llama_benchmark.reporters.composite_reporter import CompositeReporter
    from benchmark.llama_benchmark.utils.system_info import get_system_info

    console.print(f"[bold cyan]llama-bench[/] 开始加载配置...")

    try:
        config = AppConfig.load(str(models_config), str(benchmarks_config))
    except Exception as e:
        console.print(f"[red]配置加载失败: {e}[/]")
        raise typer.Exit(1)

    # 过滤模型
    models = config.models
    if model_filter:
        models = [m for m in models if m.name in model_filter]
        if not models:
            console.print(f"[red]未找到指定模型: {model_filter}[/]")
            raise typer.Exit(1)

    console.print(f"[green]已加载 {len(models)} 个模型配置[/]")

    if dry_run:
        console.print("\n[yellow]--dry-run 模式，列出将执行的任务：[/]\n")
        _show_dry_run_table(models, config)
        return

    suite_start = datetime.utcnow()
    model_results: List[ModelBenchmarkResult] = []

    # 远程设备：建立 SSH 隧道并采集远端系统信息；本地设备：直接采集本机信息
    remote_session = None
    device_cfg = config.ollama.device
    if device_cfg is not None:
        from benchmark.llama_benchmark.utils.remote_device import RemoteDeviceConfig, RemoteDeviceSession
        remote_cfg = RemoteDeviceConfig(
            host=device_cfg.host,
            user=device_cfg.user,
            password=device_cfg.password,
            key_file=device_cfg.key_file,
            ssh_port=device_cfg.ssh_port,
            ollama_remote_port=device_cfg.ollama_remote_port,
            local_tunnel_port=device_cfg.local_tunnel_port,
            name=device_cfg.name,
            arch_hint=device_cfg.arch_hint,
        )
        console.print(
            f"[cyan]远程设备 {device_cfg.host} ({device_cfg.name or 'remote'})："
            f" 建立 SSH 隧道并采集系统信息...[/]"
        )
        remote_session = RemoteDeviceSession(remote_cfg)
        remote_session._open_tunnel()
        sys_info = remote_session.collect_system_info()
        # 更新 config 中的 Ollama URL，让 Runner 直接复用已开启的隧道
        config.ollama.base_url = remote_session.ollama_base_url
        config.ollama.device = None  # 避免 Runner 重复开隧道
    else:
        sys_info = get_system_info()

    try:
        for model_cfg in models:
            console.print(f"\n[bold]处理模型: {model_cfg.name}[/] ({model_cfg.type.value})")
            model_start = datetime.utcnow()

            try:
                runner = create_runner(model_cfg, config)
                task_results = runner.run_safe()
            except Exception as e:
                console.print(f"[red]  Runner 创建失败: {e}[/]")
                task_results = []

            model_results.append(
                ModelBenchmarkResult(
                    model_name=model_cfg.name,
                    model_type=model_cfg.type.value,
                    backend=model_cfg.backend.value,
                    task_results=task_results,
                    start_time=model_start,
                    end_time=datetime.utcnow(),
                    system_info=sys_info,
                )
            )

            # 打印模型结果摘要
            for task in task_results:
                icon = "[green]PASS[/]" if task.status.value == "pass" else (
                    "[red]FAIL[/]" if task.status.value == "fail" else "[yellow]ERROR[/]"
                )
                console.print(f"  {task.task_name}: {icon}")
    finally:
        if remote_session is not None:
            remote_session._close_tunnel()

    suite_result = BenchmarkSuiteResult(
        suite_name="llama-benchmark",
        model_results=model_results,
        start_time=suite_start,
        end_time=datetime.utcnow(),
    )
    suite_result.compute_summary()

    # 生成报告
    reporter = CompositeReporter(output_dir, formats)
    reporter.generate(suite_result)

    s = suite_result.summary
    console.print(f"\n[bold green]测试完成！[/]")
    console.print(
        f"  通过: {s['passed']} / {s['total_tasks']} | "
        f"失败: {s['failed']} | "
        f"错误: {s['errored']} | "
        f"通过率: {s['pass_rate']:.1%}"
    )
    console.print(f"  报告已保存至: {output_dir}/")

    if s["failed"] > 0 or s["errored"] > 0:
        raise typer.Exit(1)


@app.command("list-models")
def list_models(
    models_config: Path = typer.Option(
        Path("configs/models.yaml"), "--models", "-m", exists=True
    ),
) -> None:
    """列出配置文件中定义的所有模型。"""
    import yaml
    with open(models_config) as f:
        data = yaml.safe_load(f)

    table = Table(title="已配置模型")
    table.add_column("名称", style="cyan")
    table.add_column("类型")
    table.add_column("后端")
    table.add_column("模型/路径")

    for m in data.get("models", []):
        model_ref = m.get("ollama_model") or m.get("path") or "—"
        table.add_row(m["name"], m["type"], m["backend"], str(model_ref))

    console.print(table)


@app.command("list-tasks")
def list_tasks() -> None:
    """列出所有可用的 benchmark 任务。"""
    tasks = {
        "LLM": ["mmlu (57学科多选，logprob)", "gsm8k (数学推理)", "hellaswag (常识推理)", "performance (TTFT/TPOT)"],
        "Whisper": ["wer_cer (WER/CER, LibriSpeech)"],
        "Embedding": ["retrieval (NDCG@10, MTEB)", "similarity (Spearman, STS)"],
        "Rerank": ["rerank_<dataset> (NDCG/MRR/MAP, BEIR)"],
        "Docling": ["parse_accuracy (field_f1, table_cell_f1)"],
    }
    for model_type, task_list in tasks.items():
        console.print(f"\n[bold]{model_type}[/]")
        for t in task_list:
            console.print(f"  • {t}")


@app.command("validate-config")
def validate_config(
    models_config: Path = typer.Option(Path("configs/models.yaml"), exists=True),
    benchmarks_config: Path = typer.Option(Path("configs/benchmarks.yaml"), exists=True),
) -> None:
    """验证配置文件格式合法性。"""
    from benchmark.llama_benchmark.core.config import AppConfig
    try:
        config = AppConfig.load(str(models_config), str(benchmarks_config))
        console.print(f"[green]配置验证通过！共 {len(config.models)} 个模型。[/]")
    except Exception as e:
        console.print(f"[red]配置验证失败: {e}[/]")
        raise typer.Exit(1)


@app.command("compare")
def compare_reports(
    reports: List[Path] = typer.Argument(..., help="2-N 份 benchmark JSON 报告路径"),
    output_dir: Path = typer.Option(
        Path("outputs/compare"),
        "--output-dir", "-o",
        help="对比报告输出目录",
    ),
) -> None:
    """对比多份 benchmark 运行结果，生成横向对比报告。

    支持 2 到 N 套硬件报告。单个报告时退化为简单摘要显示。
    输出：compare_report.{json,html,md}
    """
    # 验证文件存在
    for p in reports:
        if not p.exists():
            console.print(f"[red]文件不存在: {p}[/]")
            raise typer.Exit(1)

    if len(reports) == 1:
        console.print("[yellow]只提供了 1 份报告，无法进行横向对比。请至少提供 2 份报告。[/]")
        raise typer.Exit(1)

    from benchmark.llama_benchmark.core.compare_result import load_compare_report
    from benchmark.llama_benchmark.reporters.compare_reporter import CompareReporter
    from benchmark.llama_benchmark.reporters.recommendation import generate_recommendations

    console.print(f"[bold cyan]加载 {len(reports)} 份报告...[/]")
    report_paths = [str(p) for p in reports]

    try:
        compare_report = load_compare_report(report_paths)
    except Exception as e:
        console.print(f"[red]报告加载失败: {e}[/]")
        raise typer.Exit(1)

    compare_report.recommendations = generate_recommendations(compare_report)

    reporter = CompareReporter(output_dir=output_dir)
    try:
        paths = reporter.generate(compare_report)
    except Exception as e:
        console.print(f"[red]报告生成失败: {e}[/]")
        raise typer.Exit(1)

    # 终端打印摘要对比表（保留原有两份对比的表格体验）
    labels = [h.hw_label for h in compare_report.hardware_configs]
    table = Table(title=f"指标对比（{' vs '.join(labels)}）")
    table.add_column("任务/指标", style="cyan")
    for lbl in labels:
        table.add_column(lbl[:20])

    for model_name, tasks in compare_report.metric_table.items():
        for task_name, metrics in sorted(tasks.items()):
            for metric_name, values in sorted(metrics.items()):
                row = [f"{model_name}/{task_name}/{metric_name}"]
                for mv in values:
                    if mv is None:
                        row.append("—")
                    else:
                        color = "green" if mv.pass_fail == "pass" else (
                            "red" if mv.pass_fail == "fail" else "white"
                        )
                        row.append(f"[{color}]{mv.value:.4f} {mv.unit}[/]")
                table.add_row(*row)

    console.print(table)

    if compare_report.recommendations:
        console.print("\n[bold]推荐结论：[/]")
        for rec in compare_report.recommendations:
            console.print(f"  • {rec}")

    console.print(f"\n[green]报告已保存至:[/]")
    for fmt, path in paths.items():
        console.print(f"  {fmt}: {path}")


def _show_dry_run_table(models, config) -> None:
    table = Table(title="Dry Run - 待执行任务")
    table.add_column("模型", style="cyan")
    table.add_column("类型")
    table.add_column("后端")
    table.add_column("任务")

    for m in models:
        bench_cfg = config.get_benchmark_config(m.type)
        tasks = []
        if hasattr(bench_cfg, "mmlu") and bench_cfg.mmlu.enabled:
            tasks.append("mmlu")
        if hasattr(bench_cfg, "gsm8k") and bench_cfg.gsm8k.enabled:
            tasks.append("gsm8k")
        if hasattr(bench_cfg, "hellaswag") and bench_cfg.hellaswag.enabled:
            tasks.append("hellaswag")
        if hasattr(bench_cfg, "performance") and bench_cfg.performance.enabled:
            tasks.append("performance")
        if hasattr(bench_cfg, "wer_cer") and bench_cfg.wer_cer.enabled:
            tasks.append("wer_cer")
        if hasattr(bench_cfg, "retrieval") and bench_cfg.retrieval.enabled:
            tasks.append("retrieval")
        if hasattr(bench_cfg, "similarity") and bench_cfg.similarity.enabled:
            tasks.append("similarity")
        if hasattr(bench_cfg, "parse_accuracy") and bench_cfg.parse_accuracy.enabled:
            tasks.append("parse_accuracy")
        if hasattr(bench_cfg, "tasks") and bench_cfg.tasks.enabled:
            tasks.append("rerank")

        table.add_row(m.name, m.type.value, m.backend.value, ", ".join(tasks))

    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
