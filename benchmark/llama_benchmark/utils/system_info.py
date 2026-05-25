"""系统信息采集：CPU / 内存 / GPU，含短标签生成。"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_system_info() -> Dict[str, Any]:
    """采集当前系统的 CPU、内存、GPU 信息，并包含 ISA 特性和工具链配置。"""
    cpu = _get_cpu_info()
    memory = _get_memory_info()
    gpus = _get_gpu_info()

    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu": cpu,
        "memory": memory,
        "gpu": gpus,
        "hw_label": _generate_hw_label(cpu, gpus),
    }

    # L1/L2: ISA 特性检测（总是运行，轻量）
    try:
        from benchmark.llama_benchmark.utils.isa_detector import detect_isa
        isa_profile = detect_isa()
        info["isa"] = isa_profile.to_dict()
    except Exception:
        pass

    # L3-L5: 工具链检查（总是运行，可能稍慢但有 timeout 保护）
    try:
        from benchmark.llama_benchmark.utils.toolchain_inspector import inspect_toolchain
        toolchain_profile = inspect_toolchain()
        info["toolchain"] = toolchain_profile.to_dict()
    except Exception:
        pass

    return info


def _get_cpu_info() -> Dict[str, Any]:
    import psutil

    info: Dict[str, Any] = {
        "brand": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "arch": platform.machine(),
    }

    # CPU 最大频率
    try:
        freq = psutil.cpu_freq()
        if freq:
            info["freq_max_mhz"] = round(freq.max, 0)
    except Exception:
        pass

    # L3 缓存（Linux only）
    l3_kb = _read_l3_cache_kb()
    if l3_kb is not None:
        info["cache_l3_mb"] = round(l3_kb / 1024, 1)

    return info


def _read_l3_cache_kb() -> Optional[int]:
    """从 sysfs 读取 L3 缓存大小（Linux only）。"""
    try:
        for idx in range(8):
            cache_path = Path(f"/sys/devices/system/cpu/cpu0/cache/index{idx}/level")
            size_path = Path(f"/sys/devices/system/cpu/cpu0/cache/index{idx}/size")
            if not cache_path.exists() or not size_path.exists():
                continue
            if cache_path.read_text().strip() == "3":
                size_str = size_path.read_text().strip()  # e.g. "32768K"
                if size_str.endswith("K"):
                    return int(size_str[:-1])
                elif size_str.endswith("M"):
                    return int(size_str[:-1]) * 1024
    except Exception:
        pass
    return None


def _get_memory_info() -> Dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 2),
        "available_gb": round(vm.available / (1024**3), 2),
    }


def _get_gpu_info() -> List[Dict[str, Any]]:
    gpus = _get_nvidia_gpu_info()
    if gpus:
        return gpus

    gpus = _get_amd_gpu_info()
    return gpus


def _get_nvidia_gpu_info() -> List[Dict[str, Any]]:
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)

            gpu: Dict[str, Any] = {
                "index": i,
                "vendor": "nvidia",
                "name": name,
                "memory_total_gb": round(mem.total / (1024**3), 2),
                "memory_free_gb": round(mem.free / (1024**3), 2),
            }

            # Compute capability
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                gpu["compute_capability"] = f"{major}.{minor}"
            except Exception:
                pass

            # TDP（设计功耗）
            try:
                tdp_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
                gpu["tdp_w"] = round(tdp_mw / 1000, 0)
            except Exception:
                pass

            # 显存带宽（按型号查表，pynvml 无直接接口）
            bw = _lookup_gpu_bandwidth_gbps(name)
            if bw is not None:
                gpu["memory_bandwidth_gbps"] = bw

            gpus.append(gpu)
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return gpus


def _get_amd_gpu_info() -> List[Dict[str, Any]]:
    """尝试通过 rocm-smi 获取 AMD GPU 信息。"""
    gpus = []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return gpus

        import json
        data = json.loads(result.stdout)
        for card_id, card_data in data.items():
            if not card_id.startswith("card"):
                continue
            gpu: Dict[str, Any] = {
                "index": int(card_id.replace("card", "")),
                "vendor": "amd",
                "name": card_data.get("Card series", card_data.get("Card model", "AMD GPU")),
            }
            # VRAM total
            vram_str = card_data.get("VRAM Total Memory (B)", "")
            if vram_str:
                try:
                    gpu["memory_total_gb"] = round(int(vram_str) / (1024**3), 2)
                except ValueError:
                    pass
            gpus.append(gpu)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return gpus


# 常见 GPU 显存带宽查表（GB/s）
_GPU_BANDWIDTH_TABLE: Dict[str, float] = {
    "H100 SXM": 3350.0,
    "H100 PCIe": 2000.0,
    "H100": 2000.0,
    "A100 SXM": 2039.0,
    "A100-SXM": 2039.0,
    "A100 PCIe": 1935.0,
    "A100": 1935.0,
    "A800": 2039.0,
    "A40": 696.0,
    "A30": 933.0,
    "A10": 600.0,
    "A10G": 600.0,
    "V100 SXM": 900.0,
    "V100": 900.0,
    "RTX 4090": 1008.0,
    "RTX 4080": 717.0,
    "RTX 4070": 504.0,
    "RTX 3090": 936.0,
    "RTX 3080": 760.0,
    "RTX 3070": 448.0,
    "T4": 320.0,
    "L40S": 864.0,
    "L40": 864.0,
    "L4": 300.0,
    "MI300X": 5300.0,
    "MI250X": 3276.0,
    "MI100": 1229.0,
    "RX 7900 XTX": 960.0,
}


def _lookup_gpu_bandwidth_gbps(gpu_name: str) -> Optional[float]:
    """按 GPU 名称关键字匹配显存带宽。"""
    name_upper = gpu_name.upper()
    for key, bw in _GPU_BANDWIDTH_TABLE.items():
        if key.upper() in name_upper:
            return bw
    return None


def _generate_hw_label(cpu: Dict[str, Any], gpus: List[Dict[str, Any]]) -> str:
    """生成紧凑硬件标签，用于报告标题和对比图表的系列名。

    格式：{CPU短名}-{核数}C[-{GPU型号}-{显存}G]
    示例：Intel-Xeon-8C-A100-80G, Apple-M2-8C-CPU-only
    """
    # CPU 短名
    brand = cpu.get("brand", "") or platform.processor() or "CPU"
    # 提取品牌和型号关键词（去掉频率等噪声）
    brand_clean = re.sub(r"\s+@\s+[\d.]+GHz.*", "", brand).strip()
    brand_clean = re.sub(r"\(R\)|\(TM\)|Intel|AMD|Apple|CPU|Processor", "", brand_clean, flags=re.IGNORECASE).strip()
    brand_clean = re.sub(r"\s+", "-", brand_clean.strip())
    # 只取前 2-3 个词
    parts = [p for p in brand_clean.split("-") if p][:3]
    cpu_label = "-".join(parts) if parts else "CPU"

    cores = cpu.get("physical_cores") or cpu.get("logical_cores", "?")
    label = f"{cpu_label}-{cores}C"

    if gpus:
        gpu = gpus[0]
        gpu_name = gpu.get("name", "GPU")
        # 提取型号关键词
        gpu_clean = re.sub(r"NVIDIA|GeForce|Quadro|Tesla|AMD|Radeon|^GPU\s*", "", gpu_name, flags=re.IGNORECASE).strip()
        gpu_clean = re.sub(r"\s+", "-", gpu_clean).strip("-")
        vram_gb = gpu.get("memory_total_gb", 0)
        if vram_gb:
            label += f"-{gpu_clean}-{int(vram_gb)}G"
        else:
            label += f"-{gpu_clean}"
    else:
        label += "-CPU-only"

    # 截断到合理长度
    return label[:48]
