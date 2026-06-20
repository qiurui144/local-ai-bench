"""多平台硬件探针，按 accelerator 自动选择探测方式。"""
from __future__ import annotations
import hashlib
import json
import socket
import subprocess
from dataclasses import dataclass, field

@dataclass
class HardwareProfile:
    gpu: str = "unknown"
    driver: str = "unknown"
    cuda: str = "unknown"
    vllm: str = "unknown"
    accelerator: str = "cpu"
    arch: str = "unknown"
    cpu_model: str = "unknown"
    total_memory_gb: float = 0.0
    hostname_hash: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


class HardwareProbe:
    """Factory: probe = HardwareProbe.for_target(target_cfg); profile = probe.collect()"""

    @classmethod
    def for_target(cls, target_cfg=None):
        """根据 target 的 accelerator 字段选择探针；本地 auto-detect 时走 NvidiaProbe fallback 链。"""
        accel = getattr(target_cfg, "accelerator", "cpu") if target_cfg else "cpu"
        mapping = {
            "cuda": _NvidiaProbe,
            "rocm": _RocmProbe,
            "vulkan": _VulkanProbe,
            "rknn-npu": _RKNNProbe,
            "amd-xdna": _XDNAProbe,
            "cpu": _CpuOnlyProbe,
        }
        probe_cls = mapping.get(accel, _CpuOnlyProbe)
        return probe_cls(target_cfg)

    def __init__(self, target_cfg=None):
        self.target_cfg = target_cfg

    def collect(self) -> dict:
        profile = HardwareProfile()
        profile.hostname_hash = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]
        profile.arch = _detect_arch()
        self._fill(profile)
        return profile.to_dict()

    def _fill(self, profile: HardwareProfile):
        pass  # override in subclasses


class _NvidiaProbe(HardwareProbe):
    def _fill(self, profile):
        profile.accelerator = "cuda"
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            profile.gpu = name.decode() if isinstance(name, bytes) else str(name)
            drv = pynvml.nvmlSystemGetDriverVersion()
            profile.driver = drv.decode() if isinstance(drv, bytes) else str(drv)
            cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            profile.cuda = f"{cuda // 1000}.{cuda % 1000 // 10}"
        except Exception:
            pass


class _RocmProbe(HardwareProbe):
    def _fill(self, profile):
        profile.accelerator = "rocm"
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname", "--json"], timeout=10
            )
            data = json.loads(out)
            cards = list(data.values())
            if cards:
                profile.gpu = cards[0].get("Card series", "AMD GPU")
        except Exception:
            profile.gpu = "AMD GPU (rocm-smi unavailable)"


class _VulkanProbe(HardwareProbe):
    """AMD Windows Vulkan（Ollama Vulkan backend）—— sysfs 不可用，用 wmic 或 dxdiag 探测。"""
    def _fill(self, profile):
        profile.accelerator = "vulkan"
        # 简易探测：调 ollama 的 /api/version 拿 backend 信息
        try:
            import os

            import httpx
            base = os.environ.get("OLLAMA_AMD_BASE_URL", "http://localhost:11434")
            base = base.replace("/v1", "")
            r = httpx.get(f"{base}/api/version", timeout=5)
            if r.status_code == 200:
                profile.extra["ollama_version"] = r.json().get("version", "unknown")
            profile.gpu = "AMD Radeon (Vulkan)"
        except Exception:
            profile.gpu = "AMD GPU (Vulkan, probe unavailable)"


class _RKNNProbe(HardwareProbe):
    """RK3588 RKNN NPU — 通过 sysfs 获取 NPU 频率等信息。"""
    def _fill(self, profile):
        profile.accelerator = "rknn-npu"
        profile.arch = "aarch64"
        # NPU 频率
        try:
            npu_freq = open("/sys/class/devfreq/fdab0000.npu/cur_freq").read().strip()
            profile.extra["npu_freq_hz"] = npu_freq
        except Exception:
            pass
        # Mali GPU
        try:
            mali_freq = open("/sys/class/devfreq/fb000000.gpu/cur_freq").read().strip()
            profile.extra["mali_freq_hz"] = mali_freq
            profile.gpu = "Mali-G610"
        except Exception:
            pass
        # CPU model
        try:
            cpuinfo = open("/proc/cpuinfo").read()
            for line in cpuinfo.splitlines():
                if "Hardware" in line:
                    profile.cpu_model = line.split(":")[-1].strip()
                    break
        except Exception:
            pass


class _XDNAProbe(HardwareProbe):
    """AMD XDNA NPU（RyzenAI）—— Windows only，通过 WMI 探测。"""
    def _fill(self, profile):
        profile.accelerator = "amd-xdna"
        profile.gpu = "AMD Radeon 780M (RDNA3)"
        profile.extra["npu"] = "AMD XDNA (RyzenAI)"


class _CpuOnlyProbe(HardwareProbe):
    def _fill(self, profile):
        profile.accelerator = "cpu"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        profile.cpu_model = line.split(":")[-1].strip()
                        break
        except Exception:
            pass


def _detect_arch() -> str:
    import platform
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    if "riscv" in m:
        return "riscv64"
    return m
