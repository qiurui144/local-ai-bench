"""Reproducibility snapshotting.

A reproducible benchmark records *enough state* that, six months later,
someone (perhaps you, with no memory of the run) can recreate the same
numbers within multi-seed noise. That requires four buckets:

1. Code state             - git SHA + dirty flag + diff for uncommitted edits
2. Python environment     - interpreter version + every installed package
3. Hardware / OS          - CPU model, RAM, GPU, kernel, OS
4. Data inputs            - dataset path + SHA256 of every file used

This module captures all four and writes a `reproducibility.json` alongside
the run output. Pair it with multi_seed_runner.write_manifest() to produce
audit-grade run artifacts.

References
----------
- Pineau, J. et al. (2021). Improving Reproducibility in Machine Learning
  Research (A Report from the NeurIPS 2019 Reproducibility Program).
  JMLR 22.
- ACL Reproducibility Checklist 2020+.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Code state
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Best-effort subprocess wrapper, returning stdout or None on failure."""
    try:
        out = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@dataclass
class CodeState:
    git_sha: Optional[str]
    git_branch: Optional[str]
    git_dirty: bool
    git_diff_excerpt: Optional[str]
    repo_root: Optional[str]

    @classmethod
    def capture(cls, repo_root: Optional[Path] = None) -> "CodeState":
        cwd = Path(repo_root) if repo_root else Path.cwd()
        sha = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
        status = _run(["git", "status", "--porcelain"], cwd=cwd)
        dirty = bool(status)
        diff = None
        if dirty:
            d = _run(["git", "diff", "--stat"], cwd=cwd)
            if d and len(d) <= 8192:
                diff = d
            elif d:
                diff = d[:8192] + "\n... (truncated)"
        return cls(
            git_sha=sha,
            git_branch=branch,
            git_dirty=dirty,
            git_diff_excerpt=diff,
            repo_root=str(cwd),
        )


# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------


@dataclass
class PythonEnv:
    python_version: str
    python_executable: str
    pip_freeze: List[str]
    pip_top_level: List[str]

    @classmethod
    def capture(cls) -> "PythonEnv":
        freeze = _run([sys.executable, "-m", "pip", "freeze"]) or ""
        # 'pip list --not-required' returns just top-level deps; useful for
        # human-readable reports.
        top = _run([sys.executable, "-m", "pip", "list", "--not-required"]) or ""
        return cls(
            python_version=sys.version,
            python_executable=sys.executable,
            pip_freeze=[ln for ln in freeze.splitlines() if ln.strip()],
            pip_top_level=[ln for ln in top.splitlines() if ln.strip()],
        )


# ---------------------------------------------------------------------------
# Hardware / OS
# ---------------------------------------------------------------------------


@dataclass
class HardwareSpec:
    hostname: str
    cpu_model: Optional[str]
    cpu_count_logical: int
    cpu_count_physical: Optional[int]
    ram_total_bytes: Optional[int]
    os_name: str
    os_release: str
    kernel: str
    architecture: str
    gpu_summary: Optional[str]

    @classmethod
    def capture(cls) -> "HardwareSpec":
        cpu_model = None
        # Try /proc/cpuinfo on Linux.
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            try:
                for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
            except Exception:
                pass
        if not cpu_model:
            cpu_model = platform.processor() or None

        cpu_count_logical = os.cpu_count() or 1
        cpu_count_physical: Optional[int] = None
        try:
            import psutil  # type: ignore

            cpu_count_physical = psutil.cpu_count(logical=False)
            ram_total = psutil.virtual_memory().total
        except ImportError:
            ram_total = None
            # Fall back to /proc/meminfo.
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                try:
                    for line in meminfo.read_text(encoding="utf-8").splitlines():
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            ram_total = kb * 1024
                            break
                except Exception:
                    pass

        gpu_summary = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        if not gpu_summary:
            gpu_summary = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])

        return cls(
            hostname=socket.gethostname(),
            cpu_model=cpu_model,
            cpu_count_logical=cpu_count_logical,
            cpu_count_physical=cpu_count_physical,
            ram_total_bytes=ram_total,
            os_name=platform.system(),
            os_release=platform.release(),
            kernel=platform.version(),
            architecture=platform.machine(),
            gpu_summary=gpu_summary,
        )


# ---------------------------------------------------------------------------
# Data inputs
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a single file. Safe for multi-GB datasets."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DataInputs:
    files: List[Dict[str, Any]]

    @classmethod
    def capture(cls, paths: Sequence[Path]) -> "DataInputs":
        records: List[Dict[str, Any]] = []
        for p in paths:
            p_obj = Path(p)
            if not p_obj.exists():
                records.append({"path": str(p_obj), "missing": True})
                continue
            if p_obj.is_dir():
                for sub in sorted(p_obj.rglob("*")):
                    if sub.is_file():
                        records.append(
                            {
                                "path": str(sub),
                                "size_bytes": sub.stat().st_size,
                                "sha256": sha256_file(sub),
                            }
                        )
            else:
                records.append(
                    {
                        "path": str(p_obj),
                        "size_bytes": p_obj.stat().st_size,
                        "sha256": sha256_file(p_obj),
                    }
                )
        return cls(files=records)


# ---------------------------------------------------------------------------
# Top-level snapshot
# ---------------------------------------------------------------------------


@dataclass
class ReproducibilitySnapshot:
    timestamp_unix: float
    code: CodeState
    python_env: PythonEnv
    hardware: HardwareSpec
    data_inputs: DataInputs
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def capture(
        cls,
        data_paths: Sequence[Path] = (),
        repo_root: Optional[Path] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "ReproducibilitySnapshot":
        return cls(
            timestamp_unix=time.time(),
            code=CodeState.capture(repo_root=repo_root),
            python_env=PythonEnv.capture(),
            hardware=HardwareSpec.capture(),
            data_inputs=DataInputs.capture(data_paths),
            extra=dict(extra or {}),
        )
