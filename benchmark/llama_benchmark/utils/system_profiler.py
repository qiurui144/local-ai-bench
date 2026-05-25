"""Linux perf 性能计数器采集：IPC / L3 缓存命中率 / 分支预测命中率。

覆盖 L6 层分析：推理运行时的 CPU 微架构行为。
需要 perf_event_paranoid ≤ 1 或 root 权限。
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PerfSnapshot:
    ipc: Optional[float] = None                    # instructions per cycle
    l3_cache_miss_rate: Optional[float] = None     # %
    branch_miss_rate: Optional[float] = None       # %
    instructions: Optional[int] = None
    cycles: Optional[int] = None
    cache_misses: Optional[int] = None
    cache_references: Optional[int] = None
    branch_misses: Optional[int] = None
    branches: Optional[int] = None
    duration_ms: float = 0.0
    available: bool = False
    unavailable_reason: str = ""

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if k == "unavailable_reason":
                if v:
                    d[k] = v
                continue
            if v is not None and v is not False:
                d[k] = v
        d["available"] = self.available
        return d


def sample_perf_during(pid: int, duration_s: float = 5.0) -> PerfSnapshot:
    """
    执行 perf stat 采集指定进程的性能计数器。

    perf_event_paranoid > 1 时直接返回 available=False 并附说明。
    """
    snapshot = PerfSnapshot(duration_ms=duration_s * 1000)

    unavail_reason = _check_perf_available()
    if unavail_reason:
        snapshot.unavailable_reason = unavail_reason
        return snapshot

    events = "instructions,cycles,cache-misses,cache-references,branch-misses,branches"
    cmd = [
        "perf", "stat",
        "-p", str(pid),
        "-e", events,
        "--", "sleep", str(duration_s),
    ]

    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration_s + 15,
        )
        snapshot.duration_ms = (time.perf_counter() - t0) * 1000
        # perf stat 输出到 stderr
        output = result.stderr + result.stdout
        _parse_perf_output(snapshot, output)
        snapshot.available = True

    except FileNotFoundError:
        snapshot.unavailable_reason = "perf 命令不存在，请安装: apt install linux-tools-generic"
    except subprocess.TimeoutExpired:
        snapshot.unavailable_reason = "perf stat 超时"
    except Exception as e:
        snapshot.unavailable_reason = str(e)

    return snapshot


def _check_perf_available() -> str:
    """检查 perf 是否可用，返回不可用原因（空字符串表示可用）。"""
    try:
        result = subprocess.run(
            ["which", "perf"],
            capture_output=True,
            timeout=3,
        )
        if result.returncode != 0:
            return "perf 未安装，请: apt install linux-tools-generic"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "perf 未安装，请: apt install linux-tools-generic"

    try:
        paranoid = int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
        if paranoid > 1:
            return (
                f"perf_event_paranoid={paranoid}（>1），"
                "需要: echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
            )
    except (OSError, ValueError):
        pass

    return ""


def _parse_perf_output(snapshot: PerfSnapshot, output: str) -> None:
    """解析 perf stat 输出，提取各计数器值和派生指标。"""
    patterns = {
        "instructions": r"([\d,]+)\s+instructions",
        "cycles": r"([\d,]+)\s+cycles",
        "cache_misses": r"([\d,]+)\s+cache-misses",
        "cache_references": r"([\d,]+)\s+cache-references",
        "branch_misses": r"([\d,]+)\s+branch-misses",
        "branches": r"([\d,]+)\s+branches\b(?!-)",
    }
    for field_name, pattern in patterns.items():
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            try:
                setattr(snapshot, field_name, int(m.group(1).replace(",", "")))
            except ValueError:
                pass

    # IPC（两种格式）
    ipc_match = re.search(r"([\d.]+)\s+insn per cycle", output)
    if ipc_match:
        try:
            snapshot.ipc = float(ipc_match.group(1))
        except ValueError:
            pass
    elif snapshot.instructions and snapshot.cycles and snapshot.cycles > 0:
        snapshot.ipc = round(snapshot.instructions / snapshot.cycles, 3)

    # L3 cache miss rate
    if snapshot.cache_misses is not None and snapshot.cache_references:
        if snapshot.cache_references > 0:
            snapshot.l3_cache_miss_rate = round(
                snapshot.cache_misses / snapshot.cache_references * 100, 2
            )

    # Branch miss rate
    if snapshot.branch_misses is not None and snapshot.branches:
        if snapshot.branches > 0:
            snapshot.branch_miss_rate = round(
                snapshot.branch_misses / snapshot.branches * 100, 2
            )


def get_ollama_pid() -> Optional[int]:
    """从 /proc 遍历找 ollama serve 进程的 PID。"""
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return None

    for pid_dir in proc_dir.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                errors="replace"
            )
            if "ollama" in cmdline.lower() and "serve" in cmdline.lower():
                return int(pid_dir.name)
        except (OSError, ValueError):
            continue
    return None
