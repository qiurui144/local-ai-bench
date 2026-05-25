"""工具链溯源检查：验证 llama.cpp/Ollama 是否正确启用了 ISA 扩展（RVV/AVX/BLAS）。

覆盖 L3/L4/L5 层分析：
  L3 — C 运行时（glibc/musl）、libstdc++、OpenMP
  L4 — BLAS 库（OpenBLAS/MKL）是否针对当前 ISA 编译
  L5 — llama.cpp 编译标志（GGML_RVV/GGML_CUDA/GGML_BLAS）是否实际生效
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ToolchainProfile:
    # L3 — C 运行时
    libc_type: str = "unknown"              # "glibc" | "musl" | "unknown"
    libc_version: str = ""
    libstdcpp_version: Optional[str] = None
    openmp_available: bool = False

    # L4 — BLAS
    blas_backend: str = "none"              # "openblas" | "mkl" | "blis" | "none"
    blas_version: Optional[str] = None
    blas_isa_match: bool = False            # BLAS 是否含当前 ISA 的 kernel
    blas_isa_kernels: List[str] = field(default_factory=list)

    # L5 — llama.cpp/Ollama 编译标志（从二进制符号推断）
    ggml_backend: str = "cpu"              # "cpu" | "cuda" | "rocm" | "metal"
    ggml_rvv_enabled: bool = False
    ggml_avx2_enabled: bool = False
    ggml_avx512_enabled: bool = False
    ggml_blas_enabled: bool = False
    ggml_symbols_found: List[str] = field(default_factory=list)

    # llama-cpp-python
    llama_cpp_python_version: Optional[str] = None
    llama_cpp_python_compiled_for: Optional[str] = None

    # 自动生成的诊断警告
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if v is not None and v != [] and v != "":
                d[k] = v
        # Always include critical fields
        for key in ("libc_type", "blas_backend", "ggml_backend",
                    "ggml_rvv_enabled", "ggml_avx2_enabled", "ggml_blas_enabled"):
            d[key] = getattr(self, key)
        return d


def inspect_toolchain(ollama_binary: str = "ollama") -> ToolchainProfile:
    """
    分析 llama.cpp/Ollama 的工具链配置。

    分析链：
    1. ldd {binary} → 解析 .so 依赖（libggml / libopenblas / libcublas）
    2. nm -D {libggml*.so} → 验证 ISA kernel 符号（RVV/AVX）
    3. strings {libopenblas.so} → BLAS kernel 列表
    4. ldd --version → libc 类型和版本
    5. python3 -c "import llama_cpp" → llama-cpp-python 版本
    """
    profile = ToolchainProfile()

    _detect_libc(profile)

    ollama_path = shutil.which(ollama_binary)
    if ollama_path:
        _analyze_ollama_binary(profile, ollama_path)
    else:
        _analyze_ggml_libs_directly(profile)

    if profile.blas_backend == "openblas":
        _analyze_openblas(profile)

    _detect_llama_cpp_python(profile)
    _generate_warnings(profile)

    return profile


def _detect_libc(profile: ToolchainProfile) -> None:
    """检测 libc 类型和版本。"""
    try:
        result = subprocess.run(
            ["ldd", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout + result.stderr).lower()
        if "gnu" in output or "glibc" in output:
            profile.libc_type = "glibc"
            m = re.search(r"(\d+\.\d+)", output)
            if m:
                profile.libc_version = m.group(1)
        elif "musl" in output:
            profile.libc_type = "musl"
            m = re.search(r"(\d+\.\d+\.\d+)", output)
            if m:
                profile.libc_version = m.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback：检查已知路径
    if profile.libc_type == "unknown":
        glibc_paths = [
            "/lib/libc.so.6",
            "/lib/x86_64-linux-gnu/libc.so.6",
            "/lib/aarch64-linux-gnu/libc.so.6",
            "/lib/riscv64-linux-gnu/libc.so.6",
        ]
        musl_paths = [
            "/lib/libc.musl-x86_64.so.1",
            "/lib/ld-musl-riscv64.so.1",
            "/lib/ld-musl-aarch64.so.1",
        ]
        for p in glibc_paths:
            if Path(p).exists():
                profile.libc_type = "glibc"
                break
        if profile.libc_type == "unknown":
            for p in musl_paths:
                if Path(p).exists():
                    profile.libc_type = "musl"
                    break


def _analyze_ollama_binary(profile: ToolchainProfile, ollama_path: str) -> None:
    """通过 ldd 分析 ollama 二进制的动态链接库，再扫描 Ollama 库目录。
    对静态编译的 ollama（如 SpacemiT K1 定制版），直接扫描二进制符号表。
    """
    try:
        result = subprocess.run(
            ["ldd", ollama_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _parse_ldd_output(profile, result.stdout + result.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    _scan_ollama_lib_dir(profile)

    # 静态编译兜底：直接扫描 ollama 主二进制的全符号表（nm 不加 -D）
    # 适用于 SpacemiT K1 等将 GGML/RVV 内联编译进单一 ELF 的平台
    _analyze_static_binary(profile, ollama_path)


def _parse_ldd_output(profile: ToolchainProfile, ldd_output: str) -> None:
    """从 ldd 输出解析依赖库，识别 GPU/BLAS/OpenMP。"""
    for line in ldd_output.splitlines():
        m = re.search(r"=> (/[^\s]+)", line)
        lib_path = m.group(1) if m else ""
        if not lib_path:
            m2 = re.match(r"\s+(/[^\s]+\.so[^\s]*)", line)
            lib_path = m2.group(1) if m2 else ""
        if not lib_path:
            continue

        lib_name = Path(lib_path).name.lower()

        if "cublas" in lib_name or ("cuda" in lib_name and "curand" not in lib_name):
            profile.ggml_backend = "cuda"
        elif "rocblas" in lib_name or "rocm" in lib_name or "hipblas" in lib_name:
            profile.ggml_backend = "rocm"
        elif "metal" in lib_name:
            profile.ggml_backend = "metal"

        if "openblas" in lib_name:
            profile.blas_backend = "openblas"
            profile.ggml_blas_enabled = True
        elif "mkl_rt" in lib_name or ("mkl" in lib_name and "libmkl" in lib_name):
            profile.blas_backend = "mkl"
            profile.ggml_blas_enabled = True
        elif "blis" in lib_name:
            profile.blas_backend = "blis"
            profile.ggml_blas_enabled = True

        if "libstdc++" in lib_name:
            profile.libstdcpp_version = lib_path
        if "libgomp" in lib_name or "libomp" in lib_name or "libiomp" in lib_name:
            profile.openmp_available = True


def _scan_ollama_lib_dir(profile: ToolchainProfile) -> None:
    """扫描 Ollama 库目录，分析 ggml 共享库符号。"""
    search_dirs = [
        Path.home() / ".ollama" / "lib",
        Path("/usr/lib/ollama"),
        Path("/usr/local/lib/ollama"),
        Path("/opt/ollama/lib"),
    ]
    for lib_dir in search_dirs:
        if not lib_dir.exists():
            continue
        for so_file in lib_dir.rglob("*.so*"):
            name = so_file.name.lower()
            if "ggml" in name or ("llama" in name and "llama_bench" not in name):
                _analyze_ggml_so(profile, str(so_file))
            if "openblas" in name:
                profile.blas_backend = "openblas"
                profile.ggml_blas_enabled = True
            if "cublas" in name or ("cuda" in name and "libcuda" not in name):
                profile.ggml_backend = "cuda"
            if "libgomp" in name or "libomp" in name:
                profile.openmp_available = True


def _analyze_static_binary(profile: ToolchainProfile, binary_path: str) -> None:
    """扫描静态编译的二进制全符号表（nm 无 -D），提取 GGML/MLAS ISA 符号。
    仅在 ldd + 库目录扫描未检测到 RVV/AVX 时触发，避免重复。
    """
    if profile.ggml_rvv_enabled and profile.ggml_avx2_enabled:
        return  # 已由动态库检测覆盖
    try:
        result = subprocess.run(
            ["nm", "--defined-only", binary_path],
            capture_output=True,
            text=True,
            timeout=30,  # 大型静态二进制扫描较慢
        )
        symbols = result.stdout
        if not symbols.strip():
            return

        arch = platform.machine().lower()
        is_x86 = arch in ("x86_64", "i386", "i686", "amd64")
        isa_patterns = {
            "avx2": r"avx2|_avx2|HAVE_AVX2",
            "avx512": r"avx512|_avx512|vnni",
            "rvv": r"_rvv|rvv_|ggml.*rvv|rvv.*ggml|Mlas\w+Kernel_RVV|riscv64_spacemit",
        }
        for isa_name, pattern in isa_patterns.items():
            # x86 专属扩展不适用于其他架构，避免误判
            if isa_name in ("avx2", "avx512") and not is_x86:
                continue
            if re.search(pattern, symbols, re.IGNORECASE):
                if isa_name == "avx2":
                    profile.ggml_avx2_enabled = True
                elif isa_name == "avx512":
                    profile.ggml_avx512_enabled = True
                elif isa_name == "rvv":
                    profile.ggml_rvv_enabled = True
                    sym_names = re.findall(
                        r"\b\w*(?:_rvv|rvv_|Kernel_RVV|spacemit)\w*\b",
                        symbols,
                        re.IGNORECASE,
                    )
                    profile.ggml_symbols_found.extend(list(set(sym_names))[:5])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _analyze_ggml_libs_directly(profile: ToolchainProfile) -> None:
    """在常见系统路径查找 ggml 库。"""
    arch = platform.machine().lower()
    search_dirs = [
        "/usr/lib",
        "/usr/local/lib",
        f"/usr/lib/{arch}-linux-gnu",
        "/usr/lib64",
    ]
    for search_dir in search_dirs:
        try:
            for so_file in Path(search_dir).glob("libggml*.so*"):
                _analyze_ggml_so(profile, str(so_file))
        except OSError:
            continue


def _analyze_ggml_so(profile: ToolchainProfile, so_path: str) -> None:
    """通过 nm -D 分析 ggml 共享库的 ISA 相关符号。"""
    try:
        result = subprocess.run(
            ["nm", "-D", "--defined-only", so_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        symbols = result.stdout
        if not symbols.strip():
            return

        isa_patterns = {
            "avx2": r"avx2|_avx2|HAVE_AVX2",
            "avx512": r"avx512|_avx512|vnni",
            "rvv": r"_rvv|rvv_|riscv.*vec|vec.*riscv|ggml.*rvv|rvv.*ggml",
        }
        for isa_name, pattern in isa_patterns.items():
            if re.search(pattern, symbols, re.IGNORECASE):
                if isa_name == "avx2":
                    profile.ggml_avx2_enabled = True
                elif isa_name == "avx512":
                    profile.ggml_avx512_enabled = True
                elif isa_name == "rvv":
                    profile.ggml_rvv_enabled = True
                    sym_names = re.findall(r"\b\w*(?:rvv|riscv_v)\w*\b", symbols, re.IGNORECASE)
                    profile.ggml_symbols_found.extend(list(set(sym_names))[:3])

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _analyze_openblas(profile: ToolchainProfile) -> None:
    """分析 OpenBLAS 是否针对当前 ISA 编译（查找 ISA 专用 kernel 字符串）。"""
    openblas_paths: List[str] = []

    # ldconfig 查找
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "openblas" in line.lower():
                m = re.search(r"=> (/[^\s]+)", line)
                if m:
                    openblas_paths.append(m.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 常见路径
    arch = platform.machine().lower()
    for candidate in [
        f"/usr/lib/{arch}-linux-gnu/libopenblas.so",
        "/usr/lib/libopenblas.so",
        "/usr/local/lib/libopenblas.so",
    ]:
        if Path(candidate).exists() and candidate not in openblas_paths:
            openblas_paths.append(candidate)

    for lib_path in openblas_paths:
        try:
            result = subprocess.run(
                ["strings", lib_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            content = result.stdout

            # 查版本
            ver_m = re.search(r"OpenBLAS\s+([\d.]+)", content)
            if ver_m:
                profile.blas_version = ver_m.group(1)

            # 检测 ISA kernel 字符串
            isa_kernel_patterns = {
                "RISCV64_RVV": r"RISCV64_ZVL128B|RISCV64.*RVV|VLEN",
                "AVX2_HASWELL": r"HASWELL|AVX2HASWELL|SKYLAKEX|ZEN",
                "AVX512": r"AVX512|SKYLAKEX|ZEN3|ZEN4",
                "NEON_CORTEXA": r"CORTEXA|NEOVERSEN|ARMV8",
            }
            for kernel_name, pattern in isa_kernel_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    profile.blas_isa_kernels.append(kernel_name)

            if profile.blas_isa_kernels:
                profile.blas_isa_match = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        break  # 只检查第一个匹配


def _detect_llama_cpp_python(profile: ToolchainProfile) -> None:
    """检测 llama-cpp-python 版本和编译标志。"""
    try:
        result = subprocess.run(
            ["python3", "-c", "import llama_cpp; print(llama_cpp.__version__)"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            profile.llama_cpp_python_version = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _generate_warnings(profile: ToolchainProfile) -> None:
    """根据分析结果生成诊断警告。"""
    arch = platform.machine().lower()

    if "riscv" in arch:
        if not profile.ggml_rvv_enabled:
            profile.warnings.append(
                "RISC-V 平台：llama.cpp 未检测到 RVV 编译符号。"
                "如硬件支持 RVV，建议重新编译: "
                'CMAKE_ARGS="-DGGML_RVV=on" pip install llama-cpp-python --no-cache-dir'
            )
        if profile.blas_backend == "none":
            profile.warnings.append(
                "RISC-V 平台：未检测到 BLAS 库，prefill 性能可能受限。"
                "建议安装针对 RVV 编译的 OpenBLAS: "
                "apt install libopenblas-dev 或从源码编译 --enable-rvv"
            )
        if profile.libc_type == "musl":
            profile.warnings.append(
                "检测到 musl libc。RISC-V 上 musl 的 RVV 支持可能不完整，"
                "glibc 通常有更好的向量化支持。"
            )

    if profile.blas_backend != "none" and not profile.ggml_blas_enabled:
        profile.warnings.append(
            f"检测到 {profile.blas_backend} 已安装，但 GGML_BLAS 可能未启用。"
            '建议: CMAKE_ARGS="-DGGML_BLAS=on" 重新编译 llama-cpp-python'
        )
