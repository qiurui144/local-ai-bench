"""LLMBenchmarkRunner：协调 LLM 相关所有 benchmark 任务。"""

from __future__ import annotations

from typing import List, Optional

from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
from benchmark.llama_benchmark.core.config import AppConfig, ModelConfig, ModelType
from benchmark.llama_benchmark.core.registry import create_backend, register_runner
from benchmark.llama_benchmark.core.result import TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger
from benchmark.llama_benchmark.utils.system_info import get_system_info

logger = get_logger(__name__)


@register_runner(ModelType.LLM.value)
class LLMBenchmarkRunner(AbstractBenchmarkRunner):
    """LLM 模型 benchmark runner。

    按顺序执行：MMLU → GSM8K → HellaSwag → Performance（均可通过配置开关）。
    支持远程设备：ollama.device 配置后自动建立 SSH 隧道并采集远端 ISA/工具链信息。
    """

    supported_model_types = [ModelType.LLM.value]

    def setup(self) -> None:
        self._remote_session = None

        device_cfg = self.app_config.ollama.device
        if device_cfg is not None:
            from benchmark.llama_benchmark.utils.remote_device import RemoteDeviceConfig, RemoteDeviceSession
            remote_cfg = RemoteDeviceConfig(
                host=device_cfg.host,
                user=device_cfg.user,
                password=device_cfg.password,
                key_file=device_cfg.key_file,
                ssh_port=device_cfg.ssh_port,
                ollama_remote_port=device_cfg.ollama_remote_port,
                local_tunnel_port=device_cfg.local_tunnel_port,
                name=device_cfg.name,
                arch_hint=device_cfg.arch_hint,
            )
            self._remote_session = RemoteDeviceSession(remote_cfg)
            self._remote_session._open_tunnel()
            # 将 Ollama 地址替换为本地隧道端口
            ollama_url = self._remote_session.ollama_base_url
            logger.info(
                f"[{self.model_config.name}] 远程设备 {device_cfg.host} "
                f"SSH 隧道已建立 → {ollama_url}"
            )
        else:
            ollama_url = self.app_config.ollama.base_url

        self._backend = create_backend(self.model_config)
        if hasattr(self._backend, "configure"):
            self._backend.configure(ollama_url)
        self._backend.load()
        logger.info(f"[{self.model_config.name}] LLM 后端初始化完成")

    def run(self) -> List[TaskResult]:
        from benchmark.llama_benchmark.benchmarks.llm.mmlu import run_mmlu
        from benchmark.llama_benchmark.benchmarks.llm.gsm8k import run_gsm8k
        from benchmark.llama_benchmark.benchmarks.llm.hellaswag import run_hellaswag
        from benchmark.llama_benchmark.benchmarks.llm.performance import run_performance

        llm_cfg = self.app_config.benchmarks.llm
        results: List[TaskResult] = []

        if llm_cfg.mmlu.enabled:
            logger.info(f"[{self.model_config.name}] 开始 MMLU...")
            results.append(run_mmlu(self._backend, llm_cfg.mmlu, self.model_config.name))

        if llm_cfg.gsm8k.enabled:
            logger.info(f"[{self.model_config.name}] 开始 GSM8K...")
            results.append(run_gsm8k(self._backend, llm_cfg.gsm8k, self.model_config.name))

        if llm_cfg.hellaswag.enabled:
            logger.info(f"[{self.model_config.name}] 开始 HellaSwag...")
            results.append(run_hellaswag(self._backend, llm_cfg.hellaswag, self.model_config.name))

        if llm_cfg.performance.enabled:
            logger.info(f"[{self.model_config.name}] 开始性能测试...")
            results.append(
                run_performance(self._backend, llm_cfg.performance, self.model_config.name)
            )

        profiling_cfg = self.app_config.benchmarks.profiling
        if not profiling_cfg.enabled:
            return results

        # ── 全链路 profiling（按 L6→L7 顺序执行）──────────────────────────────

        # 并发压测
        from benchmark.llama_benchmark.benchmarks.llm.performance import run_concurrency_stress
        logger.info(f"[{self.model_config.name}] 开始并发压测...")
        results.append(
            run_concurrency_stress(
                self._backend,
                concurrency_levels=profiling_cfg.concurrency_levels,
                model_name=self.model_config.name,
                requests_per_level=profiling_cfg.stress_requests_per_level,
            )
        )

        # Context Length 扩展曲线
        from benchmark.llama_benchmark.benchmarks.llm.context_scaling import run_context_scaling
        logger.info(f"[{self.model_config.name}] 开始 Context Scaling 测试...")
        results.append(
            run_context_scaling(
                self._backend,
                context_lengths=profiling_cfg.context_lengths,
                model_name=self.model_config.name,
                output_tokens=profiling_cfg.context_scaling_output_tokens,
            )
        )

        # 持续负载测试（热降频检测）
        from benchmark.llama_benchmark.benchmarks.llm.performance import run_sustained_load
        logger.info(f"[{self.model_config.name}] 开始持续负载测试...")
        results.append(
            run_sustained_load(
                self._backend,
                model_name=self.model_config.name,
                duration_s=profiling_cfg.sustained_load_duration_s,
                window_s=profiling_cfg.sustained_load_window_s,
            )
        )

        return results

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        if self._remote_session is not None:
            self._remote_session._close_tunnel()
            self._remote_session = None
