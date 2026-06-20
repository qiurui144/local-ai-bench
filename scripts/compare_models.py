#!/usr/bin/env python3
"""Thin CLI for benchmark.compare (spec D2). Exit: 0 REPLACEABLE / 1 INCONCLUSIVE / 2 NOT_REPLACEABLE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.compare import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
