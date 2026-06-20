"""Smoke tests for the llama_benchmark typer CLI entry point.

Covers the RELEASE.md known issue "CLI unusable as shipped": typer/rich
must be importable from requirements, defaults must resolve to the
bundled benchmark/llama_configs/ files, and a missing config must yield
an actionable error (exit 2, no traceback) instead of a usage error.
"""

from typer.testing import CliRunner

from benchmark.llama_benchmark.cli import (
    _DEFAULT_BENCHMARKS_CONFIG,
    _DEFAULT_MODELS_CONFIG,
    app,
)

runner = CliRunner()


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "llama-bench" in result.output


def test_default_config_paths_exist():
    assert _DEFAULT_MODELS_CONFIG.is_file(), _DEFAULT_MODELS_CONFIG
    assert _DEFAULT_BENCHMARKS_CONFIG.is_file(), _DEFAULT_BENCHMARKS_CONFIG


def test_run_with_missing_config_gives_actionable_error():
    result = runner.invoke(app, ["run", "--models", "/nonexistent/models.yaml"])
    assert result.exit_code == 2
    # Actionable: names the missing path and where the bundled configs live.
    assert "配置文件不存在" in result.output
    assert "/nonexistent/models.yaml" in result.output
    assert "llama_configs" in result.output
    assert "Traceback" not in result.output


def test_validate_config_with_missing_benchmarks_gives_actionable_error():
    result = runner.invoke(
        app, ["validate-config", "--benchmarks", "/nonexistent/benchmarks.yaml"]
    )
    assert result.exit_code == 2
    assert "配置文件不存在" in result.output
    assert "Traceback" not in result.output


def test_validate_config_defaults_pass():
    """The bundled llama_configs load cleanly through AppConfig."""
    result = runner.invoke(app, ["validate-config"])
    assert result.exit_code == 0, result.output
    assert "配置验证通过" in result.output
