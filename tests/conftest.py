"""Pytest conftest for local-ai-bench tests.

Adds the repo root to sys.path so `import benchmark.rigor.*` etc. works
without installing the package.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
