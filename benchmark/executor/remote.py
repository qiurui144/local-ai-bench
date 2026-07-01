r"""
RemoteExecutor — SSH → rsync/scp → run → scp 三步远程执行。

流程：
1. sync_code(): rsync (Linux target) 或 scp (Windows target) 推代码到目标机
2. run_remote(): SSH 执行 `python run_benchmark.py --model X --local-only ...`
3. collect_reports(): scp 将报告拉回本机

Windows 特殊处理：
- rsync 目标路径需转为 MSYS 格式（C:\ → /c/）
- scp 目标路径用正斜杠（Windows OpenSSH 接受）
- cmd /c 封装 SSH 命令，支持 && 链式
- 通过 env_overrides 注入目标机环境变量（如 OLLAMA_AMD_BASE_URL=http://localhost:11434/v1）
"""
from __future__ import annotations
import base64
import subprocess
from pathlib import Path
import yaml
from common import TargetConfig

ROOT = Path(__file__).parent.parent.parent


def _win_to_msys(win_path: str) -> str:
    """C:\\Users\\x\\y  →  /c/Users/x/y  (rsync 到 Windows 需要 MSYS 格式)"""
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = "/" + p[0].lower() + p[2:]
    return p


def _win_to_scp(win_path: str) -> str:
    """C:\\Users\\x\\y  →  C:/Users/x/y  (scp 到 Windows OpenSSH 接受正斜杠)"""
    return win_path.replace("\\", "/")


class RemoteExecutor:
    def __init__(self, target: TargetConfig):
        self.target = target
        self.last_error: Exception | None = None
        self._check_env()

    def _check_env(self):
        if not self.target.ip:
            raise ValueError(
                f"Target '{self.target.name}' requires env var {self.target.ip_env} to be set"
            )

    def _ssh_base(self) -> list[str]:
        t = self.target
        return [
            "sshpass", "-p", t.ssh_pass, "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            f"{t.ssh_user}@{t.ip}",
        ]

    def _ssh_cmd(self, cmd: str) -> list[str]:
        """构建 SSH 命令列表；Windows 用 cmd /c 以支持 && 链式。"""
        base = self._ssh_base()
        if self.target.platform == "windows":
            return base + ["cmd", "/S", "/C", f'"{cmd}"']
        return base + [cmd]

    def _run_checked(
        self,
        cmd: list[str],
        timeout: int,
        *,
        allowed_exit_codes: set[int] | None = None,
    ) -> int:
        allowed = allowed_exit_codes or {0}
        rc = subprocess.call(cmd, timeout=timeout)
        if rc not in allowed:
            raise RuntimeError(f"remote command failed with exit code {rc}")
        return rc

    def sync_code(self):
        """推送项目代码到目标机（排除 output/、.git/、__pycache__/）。

        Linux/macOS target: rsync -az
        Windows target: 先在目标机建目录，再 scp -r 推送（避免 rsync Windows 路径问题）
        """
        t = self.target
        excl = [
            "output/", ".git/", "__pycache__/", "*.pyc", ".pytest_cache/", ".venv/",
            # Do not push checked-in/generated model artifacts to small edge
            # filesystems; targets keep these under their own model hubs.
            "datasets/asr/models/*.tar.bz2",
            "drivers/*/ov_models/",
            "drivers/*/ort_models/",
            "drivers/rk182x-linux/RKNN3_SDK/",
        ]

        if t.platform == "windows":
            # Step 1: 确保目标目录存在
            win_dir = t.remote_workdir
            mkdir_cmd = f'if not exist "{win_dir}" mkdir "{win_dir}"'
            subprocess.check_call(self._ssh_cmd(mkdir_cmd), timeout=30)
            # Step 2: tar-over-ssh 推运行所需文件。Windows OpenSSH 自带
            # bsdtar/tar.exe；比逐文件 scp -r 快很多，尤其 fixtures/datasets
            # 小文件多时。
            include = [
                "benchmark", "docs", "fixtures", "golden", "scripts",
                "vllm_configs", "common.py", "run_benchmark.py", "models.yaml",
                "targets.yaml", "requirements.txt", "requirements-windows.txt",
                "README.md", "DEVELOP.md", "RELEASE.md", "LICENSE",
                "datasets/conditioned", "datasets/conversation_drift",
                "datasets/asr", "datasets/ocr", "datasets/retrieval",
                "datasets/scenarios", "datasets/translation",
            ]
            include = [name for name in include if (ROOT / name).exists()]
            tar_cmd = [
                "tar", "-C", str(ROOT),
                "--exclude=*/__pycache__",
                "--exclude=*.pyc",
                "--exclude=datasets/asr/models/*.tar.bz2",
                "-cf", "-", *include,
            ]
            extract_cmd = self._ssh_cmd(f'tar -xf - -C "{win_dir}"')
            tar_proc = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
            try:
                ssh_proc = subprocess.run(
                    extract_cmd,
                    stdin=tar_proc.stdout,
                    timeout=1800,
                    check=False,
                )
            finally:
                if tar_proc.stdout:
                    tar_proc.stdout.close()
            tar_rc = tar_proc.wait(timeout=30)
            if tar_rc != 0:
                raise subprocess.CalledProcessError(tar_rc, tar_cmd)
            if ssh_proc.returncode != 0:
                raise subprocess.CalledProcessError(ssh_proc.returncode, extract_cmd)
        else:
            dest = f"{t.ssh_user}@{t.ip}:{t.remote_workdir}"
            excl_args = [f"--exclude={e}" for e in excl]
            cmd = [
                "rsync", "-az", "--delete",
                *excl_args,
                "-e", f"sshpass -p {t.ssh_pass} ssh -o StrictHostKeyChecking=no",
                str(ROOT) + "/",
                dest,
            ]
            subprocess.check_call(cmd, timeout=120)

    def run_remote(self, model_name: str, extra_args: list[str] = ()) -> None:
        """SSH 执行 benchmark；--local-only 防止目标机再次远程分发。

        env_overrides（targets.yaml 可选字段）中的变量会注入到目标机 SSH 环境，
        例如 OLLAMA_AMD_BASE_URL=http://localhost:11434/v1（让目标机用本地 Ollama）。
        """
        t = self.target
        args_str = " ".join(["--model", model_name, "--local-only", *extra_args])
        py = t.python_cmd
        workdir = t.remote_workdir

        # 注入 env_overrides（优先让目标机用 localhost 端点，避免回环到 controller IP）
        env_overrides = getattr(t, "env_overrides", None) or {}
        # 目标机自身 Ollama 默认 localhost（如果 models.yaml base_url_override 也写了 IP，也能自访问）

        def _provider_for_model(name: str) -> str | None:
            try:
                data = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
            except Exception:
                return None
            for item in data.get("models", []):
                if item.get("name") == name:
                    return item.get("provider")
            return None

        if t.platform == "windows":
            # Windows: set ENV=VAL && ... 链式
            env_prefix = ""
            for k, v in env_overrides.items():
                env_prefix += f'set "{k}={v}" && '
            if t.runtime == "ollama" and _provider_for_model(model_name) != "local_onnx":
                port = t.runtime_port or 11434
                ps_script = "\n".join([
                    "$env:OLLAMA_HOST='0.0.0.0'",
                    "$env:OLLAMA_IGPU_ENABLE='1'",
                    "$exe=Join-Path $env:LOCALAPPDATA 'Programs\\Ollama\\ollama.exe'",
                    "$out=Join-Path $env:USERPROFILE 'ollama-benchmark-serve.log'",
                    "$err=Join-Path $env:USERPROFILE 'ollama-benchmark-serve.err.log'",
                    "Start-Process -FilePath $exe -ArgumentList 'serve' "
                    "-WindowStyle Hidden -RedirectStandardOutput $out "
                    "-RedirectStandardError $err",
                    "Start-Sleep -Seconds 8",
                ])
                encoded = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
                env_prefix += (
                    f'(curl -s http://localhost:{port}/api/version >NUL 2>NUL || '
                    f'powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}) && '
                )
            run_cmd = f'{env_prefix}cd /d {workdir} && {py} run_benchmark.py {args_str}'
        else:
            env_prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
            run_cmd = f'cd "{workdir}" && {env_prefix} "{py}" run_benchmark.py {args_str}'

        rc = self._run_checked(
            self._ssh_cmd(run_cmd),
            timeout=7200,
            allowed_exit_codes={0, 1, 2},
        )
        self.last_error = (
            RuntimeError(f"remote benchmark exited with code {rc}")
            if rc != 0 else None
        )

    def collect_reports(self, local_out: Path) -> None:
        """scp 拉回目标机 output/reports/ 到本机 local_out/。"""
        t = self.target
        if t.platform == "windows":
            remote_path = _win_to_scp(t.remote_workdir) + "/output/reports/"
        else:
            remote_path = f"{t.remote_workdir}/output/reports/"
        src = f"{t.ssh_user}@{t.ip}:{remote_path}"
        local_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            "sshpass", "-p", t.ssh_pass,
            "scp", "-r", "-o", "StrictHostKeyChecking=no",
            src, str(local_out),
        ]
        subprocess.check_call(cmd, timeout=300)

    def install_deps(self, requirements_file: str = "requirements.txt") -> None:
        """在目标机安装 Python 依赖（首次部署或依赖更新时调用）。

        Windows: py -m pip install ...
        Linux: python3 -m pip install ...
        """
        t = self.target
        py = t.python_cmd
        if t.platform == "windows":
            requirements_file = (
                "requirements-windows.txt"
                if requirements_file == "requirements.txt" else requirements_file
            )
            install_cmd = (
                f'set "PYTHONUTF8=1" && cd /d {t.remote_workdir} '
                f'&& {py} -m pip install -r {requirements_file} '
                f'&& {py} -m pip uninstall -y onnxruntime onnxruntime-directml '
                f'&& {py} -m pip install openvino==2025.4.1 openvino-telemetry==2025.2.0 '
                f'onnxruntime-directml rapidocr-onnxruntime rapidocr-openvino --no-deps'
            )
        else:
            install_cmd = f'cd "{t.remote_workdir}" && "{py}" -m pip install -r {requirements_file}'
        subprocess.check_call(self._ssh_cmd(install_cmd), timeout=600)

    def run_benchmark(self, model_name: str, extra_args: list[str] = (),
                      install_first: bool = False, raise_on_error: bool = True) -> Path:
        """完整三步（可选四步）：[install_deps →] sync → run → collect。"""
        out_dir = ROOT / "output" / "reports" / self.target.name
        self.sync_code()
        if install_first:
            self.install_deps()
        error: Exception | None = None
        self.last_error = None
        try:
            self.run_remote(model_name, extra_args)
        except Exception as exc:
            error = exc
            self.last_error = exc
        finally:
            self.collect_reports(out_dir)
        if error and raise_on_error:
            raise error
        return out_dir
