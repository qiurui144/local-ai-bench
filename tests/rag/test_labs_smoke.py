"""Smoke tests for the runnable labs advertised in the README.

The README promises ``python -m benchmark.rag.labs.labN`` works for all
labs; each one is executed here as ``__main__`` and must complete without
raising. All labs are pure-CPU and fast (<1s measured), so none are skipped.
"""
from __future__ import annotations

import contextlib
import io
import pkgutil
import runpy

import benchmark.rag.labs as labs_pkg
import pytest

LAB_MODULES = sorted(
    m.name for m in pkgutil.iter_modules(labs_pkg.__path__) if m.name.startswith("lab")
)


def test_all_advertised_labs_discovered():
    assert len(LAB_MODULES) == 8, f"expected 8 labs per README, found: {LAB_MODULES}"


@pytest.mark.parametrize("lab_name", LAB_MODULES)
def test_lab_runs_as_main(lab_name):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        runpy.run_module(f"benchmark.rag.labs.{lab_name}", run_name="__main__")
    assert stdout.getvalue().strip(), f"{lab_name} produced no output"
