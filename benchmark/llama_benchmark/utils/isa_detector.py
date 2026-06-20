"""ISA 特性检测：解析 /proc/cpuinfo flags，识别 AVX2/AVX512/RVV/NEON 等指令集扩展。

覆盖 L1/L2 层分析：硬件 ISA 能力 + 内核是否正确启用相关扩展。
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ISAProfile:
    arch: str = ""                          # "x86_64" | "aarch64" | "riscv64"
    cpu_flags: List[str] = field(default_factory=list)  # 原始 flags

    # x86 SIMD
    has_avx2: bool = False
    has_avx512f: bool = False
    has_avx512vnni: bool = False
    has_amx_int8: bool = False
    has_fma: bool = False

    # ARM
    has_neon: bool = False
    has_sve: bool = False
    has_sve2: bool = False

    # RISC-V
    has_rvv: bool = False
    rvv_vlen: Optional[int] = None          # 向量寄存器位宽（bits）
    rvv_spec_version: Optional[str] = None  # "v1.0" | "v0.7"

    # 内核状态
    kernel_version: str = ""
    huge_pages_enabled: bool = False
    numa_nodes: int = 1
    perf_event_paranoid: int = 3            # 默认最严格

    # 内存
    ddr_channels: Optional[int] = None
    ddr_speed_mts: Optional[int] = None     # MT/s

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if v is not None and v != [] and v != "":
                d[k] = v
        # Always include key boolean fields even if False
        for key in ("has_avx2", "has_avx512f", "has_rvv", "has_neon"):
            d[key] = getattr(self, key)
        return d


def detect_isa() -> ISAProfile:
    """从 /proc/cpuinfo 解析 ISA 特性，读取内核状态信息。"""
    arch = platform.machine().lower()
    profile = ISAProfile(arch=arch)

    try:
        profile.kernel_version = platform.release()
    except Exception:
        pass

    _parse_cpuinfo(profile)

    if profile.has_rvv:
        profile.rvv_vlen = _detect_rvv_vlen()

    profile.huge_pages_enabled = _check_huge_pages()
    profile.numa_nodes = _count_numa_nodes()
    profile.perf_event_paranoid = _read_perf_event_paranoid()

    ddr_channels, ddr_speed = _get_ddr_info()
    profile.ddr_channels = ddr_channels
    profile.ddr_speed_mts = ddr_speed

    return profile


def _parse_cpuinfo(profile: ISAProfile) -> None:
    """解析 /proc/cpuinfo 提取 flags 和 ISA 扩展。"""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    # x86: "flags" line; ARM: "Features" line; RISC-V: "isa" line
    flags_match = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.MULTILINE | re.IGNORECASE)
    features_match = re.search(r"^Features\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    isa_match = re.search(r"^isa\s*:\s*(.+)$", cpuinfo, re.MULTILINE | re.IGNORECASE)
    misa_match = re.search(r"^misa\s*:\s*(.+)$", cpuinfo, re.MULTILINE | re.IGNORECASE)

    flags_line = ""
    if flags_match:
        flags_line = flags_match.group(1)
    elif features_match:
        flags_line = features_match.group(1)
    elif isa_match:
        flags_line = isa_match.group(1)

    profile.cpu_flags = flags_line.split() if flags_line else []
    flags_set = set(profile.cpu_flags)

    arch_lower = profile.arch

    if "x86" in arch_lower or arch_lower in ("i386", "i686", "amd64"):
        profile.has_avx2 = "avx2" in flags_set
        profile.has_avx512f = "avx512f" in flags_set
        profile.has_avx512vnni = "avx512_vnni" in flags_set
        profile.has_amx_int8 = "amx_int8" in flags_set
        profile.has_fma = "fma" in flags_set

    elif "aarch64" in arch_lower or "arm" in arch_lower:
        profile.has_neon = "asimd" in flags_set or "neon" in flags_set
        profile.has_sve = "sve" in flags_set
        profile.has_sve2 = "sve2" in flags_set

    elif "riscv" in arch_lower:
        isa_str = (isa_match.group(1) if isa_match else "") or flags_line
        # ISA string 如 "rv64imafdcvh_zicsr_..." 或 "rv64gc_v1p0"
        profile.has_rvv = bool(re.search(r"_v\b|_zve|rv64.{0,6}v", isa_str, re.IGNORECASE))

        # 也检查 sysfs
        sysfs_isa_path = Path("/sys/devices/system/cpu/cpu0/riscv_isa")
        if sysfs_isa_path.exists():
            try:
                sysfs_isa = sysfs_isa_path.read_text(errors="replace").strip()
                if re.search(r"_v\b|_zve|_zvl", sysfs_isa, re.IGNORECASE):
                    profile.has_rvv = True
            except OSError:
                pass

        # 解析 RVV spec version
        ver_match = re.search(r"_v(\d+)p(\d+)", isa_str)
        if ver_match:
            profile.rvv_spec_version = f"v{ver_match.group(1)}.{ver_match.group(2)}"
        elif profile.has_rvv:
            profile.rvv_spec_version = "v1.0"

        # misa V bit（bit 21）
        if misa_match:
            try:
                val = int(misa_match.group(1).strip(), 16)
                if val & (1 << 21):
                    profile.has_rvv = True
            except ValueError:
                if "v" in misa_match.group(1).lower():
                    profile.has_rvv = True


def _detect_rvv_vlen() -> Optional[int]:
    """检测 RISC-V 向量寄存器位宽 (VLEN)。"""
    # 尝试从 sysfs 读取 vlenb（字节）
    vlenb_paths = [
        "/sys/devices/system/cpu/cpu0/riscv/vlenb",
        "/proc/sys/abi/riscv_v_vlenb",
    ]
    for path in vlenb_paths:
        try:
            val = int(Path(path).read_text().strip())
            return val * 8  # bytes → bits
        except (OSError, ValueError):
            pass

    # 尝试从 /proc/cpuinfo "vlenb" 字段
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(errors="replace")
        m = re.search(r"vlenb\s*:\s*(\d+)", cpuinfo, re.IGNORECASE)
        if m:
            return int(m.group(1)) * 8
    except OSError:
        pass

    return None


def _check_huge_pages() -> bool:
    """检查是否启用 HugePages。"""
    try:
        meminfo = Path("/proc/meminfo").read_text()
        m = re.search(r"HugePages_Total:\s*(\d+)", meminfo)
        if m:
            return int(m.group(1)) > 0
    except OSError:
        pass
    return False


def _count_numa_nodes() -> int:
    """统计 NUMA 节点数量。"""
    try:
        numa_dir = Path("/sys/devices/system/node")
        if numa_dir.exists():
            nodes = [d for d in numa_dir.iterdir() if d.name.startswith("node") and d.is_dir()]
            return max(len(nodes), 1)
    except OSError:
        pass
    return 1


def _read_perf_event_paranoid() -> int:
    """读取 perf_event_paranoid 值（≤1 才能在非 root 下采集计数器）。"""
    try:
        val = Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
        return int(val)
    except (OSError, ValueError):
        return 3


def _get_ddr_info():
    """尝试从 dmidecode 读取 DDR 通道数和速度（需要 root 或 sudo）。"""
    channels = None
    speed_mts = None
    try:
        result = subprocess.run(
            ["dmidecode", "-t", "memory"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            # 计算已使用的内存槽数量
            slots = re.findall(r"Size:\s+(\d+ [MG]B)", result.stdout)
            channels = len(slots) if slots else None
            # 读取最高速度
            speed_matches = re.findall(r"Speed:\s+(\d+)\s+MT/s", result.stdout)
            if speed_matches:
                speeds = [int(s) for s in speed_matches if int(s) > 0]
                if speeds:
                    speed_mts = max(speeds)
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        pass
    return channels, speed_mts
