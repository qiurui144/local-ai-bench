"""高精度计时工具。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def measure_ms() -> Generator[list, None, None]:
    """上下文管理器，测量代码块执行时间（毫秒）。

    用法::

        with measure_ms() as elapsed:
            do_something()
        print(f"耗时: {elapsed[0]:.2f} ms")
    """
    result = [0.0]
    start = time.perf_counter_ns()
    try:
        yield result
    finally:
        result[0] = (time.perf_counter_ns() - start) / 1_000_000


def now_ms() -> float:
    """返回当前时间戳（毫秒）。"""
    return time.perf_counter_ns() / 1_000_000


class Stopwatch:
    """可复用的秒表，支持多次计时。"""

    def __init__(self) -> None:
        self._start: int = 0
        self._laps: list[float] = []

    def start(self) -> "Stopwatch":
        self._start = time.perf_counter_ns()
        return self

    def lap(self) -> float:
        """记录一圈时间（ms）并返回。"""
        elapsed = (time.perf_counter_ns() - self._start) / 1_000_000
        self._laps.append(elapsed)
        self._start = time.perf_counter_ns()
        return elapsed

    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self._start) / 1_000_000

    @property
    def laps(self) -> list[float]:
        return list(self._laps)
