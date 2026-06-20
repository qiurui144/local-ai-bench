"""远程设备管理：SSH 隧道建立、远端 ISA/工具链采集、生命周期管理。

用法：
    cfg = RemoteDeviceConfig(host=os.environ["K1_HOST"], user=os.environ["K1_USER"], password=os.environ.get("K1_PASS"))
    with RemoteDeviceSession(cfg) as sess:
        system_info = sess.collect_system_info()   # 在 K1 上运行检测，返回 dict
        ollama_url  = sess.ollama_base_url          # "http://localhost:11435"
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RemoteDeviceConfig:
    """远程设备 SSH 连接与 Ollama 转发配置。"""

    host: str                               # IP 或主机名
    user: str                               # SSH 用户名
    password: Optional[str] = None         # SSH 密码（与 key_file 二选一）
    key_file: Optional[str] = None         # SSH 私钥路径
    ssh_port: int = 22
    ollama_remote_port: int = 11434        # 远端 Ollama 监听端口
    local_tunnel_port: int = 11435         # 本机转发端口（避免与本地 11434 冲突）
    name: str = ""                          # 设备显示名称，e.g. "k1-spacemit"
    arch_hint: str = ""                     # "riscv64" | "aarch64" | "" = 自动检测
    tunnel_timeout: float = 15.0           # 等待隧道就绪的最长时间（秒）
    ssh_connect_timeout: int = 10          # SSH 连接超时（ConnectTimeout，秒）
    ssh_command_timeout: int = 60          # 默认 SSH 命令执行超时（秒）


class RemoteDeviceSession:
    """SSH 隧道 + 远端 ISA/工具链采集的上下文管理器。

    所有 SSH 操作均依赖宿主机已安装 sshpass（密码登录）或配置 SSH 密钥。
    """

    def __init__(self, config: RemoteDeviceConfig) -> None:
        self.config = config
        self._tunnel_proc: Optional[subprocess.Popen] = None
        self._system_info_cache: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "RemoteDeviceSession":
        self._open_tunnel()
        return self

    def __exit__(self, *_: Any) -> None:
        self._close_tunnel()

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    @property
    def ollama_base_url(self) -> str:
        """返回本机 SSH 转发后可直接访问的 Ollama URL。"""
        return f"http://localhost:{self.config.local_tunnel_port}"

    def run_remote(self, command: str, timeout: Optional[int] = None) -> str:
        """在远端设备上执行 shell 命令，返回 stdout（失败返回空字符串）。

        timeout 默认使用 config.ssh_command_timeout（秒）。
        命令执行超时或连接失败均返回空字符串，并记录警告日志。
        """
        effective_timeout = timeout if timeout is not None else self.config.ssh_command_timeout
        cmd = self._ssh_prefix() + [f"{self.config.user}@{self.config.host}", command]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=effective_timeout, check=False,
            )
            if result.returncode != 0 and result.stderr:
                logger.debug(
                    "远程命令退出码 %d，stderr: %s",
                    result.returncode, result.stderr.strip()[:200],
                )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning("远程命令超时（%ds）：%s", effective_timeout, command[:100])
            return ""

    def collect_system_info(self) -> Dict[str, Any]:
        """SSH 远端采集完整系统信息（基础 + ISA + 工具链）。

        结果与本地 system_info.get_system_info() 格式兼容，
        包含 platform / cpu / memory / isa / toolchain / hw_label 字段。
        """
        if self._system_info_cache is not None:
            return self._system_info_cache

        info = self._collect_basic_info()
        isa_data = self._collect_remote_isa()
        if isa_data:
            info["isa"] = isa_data
        toolchain_data = self._collect_remote_toolchain()
        if toolchain_data:
            info["toolchain"] = toolchain_data

        # 生成简洁硬件标签
        arch = info.get("cpu", {}).get("arch", self.config.arch_hint or "unknown")
        cores = info.get("cpu", {}).get("logical_cores", "?")
        device_name = self.config.name or self.config.host
        info["hw_label"] = f"{device_name}-{arch}-{cores}C-CPU-only"

        self._system_info_cache = info
        return info

    # ── SSH 命令构造 ──────────────────────────────────────────────────────────

    def _scp_prefix(self) -> List[str]:
        """构建 SCP 命令前缀（与 _ssh_prefix 保持一致，但使用 -P 端口参数）。"""
        cmd: List[str] = []
        if self.config.password:
            cmd = ["sshpass", "-p", self.config.password]
        cmd += [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.config.ssh_connect_timeout}",
            "-P", str(self.config.ssh_port),  # scp 用 -P（大写），ssh 用 -p
        ]
        if self.config.key_file:
            cmd += ["-i", self.config.key_file]
        return cmd

    def _ssh_prefix(self) -> List[str]:
        """构建基础 SSH 命令前缀（含 sshpass、StrictHostKeyChecking 及连接超时）。"""
        cmd: List[str] = []
        if self.config.password:
            cmd = ["sshpass", "-p", self.config.password]
        cmd += [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=no",
            "-o", f"ConnectTimeout={self.config.ssh_connect_timeout}",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-p", str(self.config.ssh_port),
        ]
        if self.config.key_file:
            cmd += ["-i", self.config.key_file]
        return cmd

    # ── 隧道管理 ──────────────────────────────────────────────────────────────

    def _open_tunnel(self) -> None:
        """在后台建立 SSH 本地端口转发隧道，通过主动 TCP 探测等待就绪。

        替代 time.sleep(1.5)：每 0.2 秒尝试连接本地转发端口，
        最多等待 tunnel_timeout 秒；若进程已退出则提前报错。
        """
        cmd = self._ssh_prefix() + [
            "-N",
            "-L", (
                f"{self.config.local_tunnel_port}:localhost:"
                f"{self.config.ollama_remote_port}"
            ),
            f"{self.config.user}@{self.config.host}",
        ]
        self._tunnel_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.debug(
            "SSH 隧道进程已启动（PID=%d），等待本地端口 %d 就绪…",
            self._tunnel_proc.pid,
            self.config.local_tunnel_port,
        )
        self._wait_for_tunnel()

    def _wait_for_tunnel(self) -> None:
        """轮询 localhost:local_tunnel_port 直到可连接或超时。"""
        deadline = time.monotonic() + self.config.tunnel_timeout
        poll_interval = 0.2
        port = self.config.local_tunnel_port

        while time.monotonic() < deadline:
            # 若隧道进程已提前退出，立即报错
            if self._tunnel_proc and self._tunnel_proc.poll() is not None:
                raise RuntimeError(
                    f"SSH 隧道进程意外退出（returncode={self._tunnel_proc.returncode}），"
                    f"请检查 SSH 连通性和密钥/密码配置。"
                )
            # 主动 TCP 探测
            try:
                with socket.create_connection(("localhost", port), timeout=0.5):
                    logger.debug("SSH 隧道端口 %d 已就绪", port)
                    return
            except OSError:
                time.sleep(poll_interval)

        # 超时：关闭进程并报错
        self._close_tunnel()
        raise TimeoutError(
            f"SSH 隧道在 {self.config.tunnel_timeout}s 内未就绪（本地端口 {port} 不可达），"
            f"请检查远端 Ollama 是否运行在端口 {self.config.ollama_remote_port}。"
        )

    def _close_tunnel(self) -> None:
        """终止 SSH 隧道进程。"""
        if self._tunnel_proc and self._tunnel_proc.poll() is None:
            self._tunnel_proc.terminate()
            try:
                self._tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._tunnel_proc.kill()
        self._tunnel_proc = None

    # ── 远端信息采集 ──────────────────────────────────────────────────────────

    def _collect_basic_info(self) -> Dict[str, Any]:
        """通过简单 shell 命令采集基础系统信息（无 Python 依赖）。"""
        uname = self.run_remote("uname -a").strip()
        arch = self.run_remote("uname -m").strip()
        cpu_count = self.run_remote("nproc").strip()
        mem_kb_str = self.run_remote(
            "grep MemTotal /proc/meminfo | awk '{print $2}'"
        ).strip()
        os_release = self.run_remote(
            "cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"'"
        ).strip()

        info: Dict[str, Any] = {
            "platform": uname,
            "python_version": self.run_remote("python3 --version 2>&1").strip(),
            "cpu": {
                "arch": arch or self.config.arch_hint,
                "logical_cores": int(cpu_count) if cpu_count.isdigit() else None,
                "brand": self.run_remote(
                    "grep 'model name\\|uarch' /proc/cpuinfo | head -1 | cut -d: -f2"
                ).strip(),
            },
            "os_release": os_release,
        }
        if mem_kb_str.isdigit():
            info["memory"] = {"total_gb": round(int(mem_kb_str) / 1024 / 1024, 2)}
        return info

    def _run_remote_python(self, script_src: str, suffix: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
        """通过 SCP 上传脚本到远端 /tmp/ 再 SSH 执行，解析 JSON 输出。

        遵循 CLAUDE.md 规范：禁止 stdin 管道传脚本（避免 sudo/密码与 stdin 冲突）。
        suffix 追加到 script_src 末尾，作为执行入口（输出一行 JSON）。
        """
        full_script = script_src + "\n" + suffix
        remote_path = f"/tmp/llm_bench_{uuid.uuid4().hex[:8]}.py"

        # 写入本地临时文件
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="llm_bench_")
        try:
            try:
                os.write(tmp_fd, full_script.encode("utf-8"))
            finally:
                os.close(tmp_fd)

            # SCP 上传
            scp_cmd = self._scp_prefix() + [
                tmp_path,
                f"{self.config.user}@{self.config.host}:{remote_path}",
            ]
            scp_result = subprocess.run(
                scp_cmd, capture_output=True, text=True, timeout=30, check=False
            )
            if scp_result.returncode != 0:
                logger.debug("SCP 上传失败（%s）: %s", remote_path, scp_result.stderr.strip()[:200])
                return None

            # SSH 执行（执行完毕后清理远端文件）
            stdout = self.run_remote(
                f"python3 {remote_path}; rm -f {remote_path}", timeout=timeout
            )
            # 以防执行异常未清理
            self.run_remote(f"rm -f {remote_path}", timeout=10)

            stdout = stdout.strip()
            if not stdout:
                return None
            # 取最后一行（屏蔽 import warning 等干扰输出）
            last_line = stdout.splitlines()[-1]
            return json.loads(last_line)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _collect_remote_isa(self) -> Optional[Dict[str, Any]]:
        """在远端执行 isa_detector，返回 ISAProfile.to_dict()。"""
        isa_py = Path(__file__).parent / "isa_detector.py"
        if not isa_py.exists():
            return None
        src = isa_py.read_text(encoding="utf-8")
        suffix = "import json as _j; print(_j.dumps(detect_isa().to_dict()))"
        return self._run_remote_python(src, suffix, timeout=30)

    def _collect_remote_toolchain(self) -> Optional[Dict[str, Any]]:
        """在远端执行 toolchain_inspector，返回 ToolchainProfile.to_dict()。"""
        tc_py = Path(__file__).parent / "toolchain_inspector.py"
        if not tc_py.exists():
            return None
        src = tc_py.read_text(encoding="utf-8")
        suffix = "import json as _j; print(_j.dumps(inspect_toolchain().to_dict()))"
        return self._run_remote_python(src, suffix, timeout=90)
