"""AbstractBenchmarkRunner：所有 benchmark runner 的基类。"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.config import AppConfig, ModelConfig
    from benchmark.llama_benchmark.core.result import ModelBenchmarkResult, TaskResult


class AbstractBenchmarkRunner(ABC):
    """所有 Benchmark Runner 的抽象基类。

    子类必须实现 setup()、run()、teardown()。
    使用 run_safe() 执行完整流程，保证 teardown 一定被调用。
    """

    supported_model_types: List[str] = []

    def __init__(self, model_config: "ModelConfig", app_config: "AppConfig") -> None:
        self.model_config = model_config
        self.app_config = app_config
        self._backend = None

    @abstractmethod
    def setup(self) -> None:
        """初始化后端、加载模型。"""

    @abstractmethod
    def run(self) -> List["TaskResult"]:
        """执行所有 benchmark 任务，返回结果列表。"""

    @abstractmethod
    def teardown(self) -> None:
        """释放资源（模型内存、连接等）。"""

    def validate_config(self) -> None:
        """验证配置合法性，子类可覆盖以添加额外检查。"""
        if self.model_config.type.value not in self.supported_model_types:
            raise ValueError(
                f"{self.__class__.__name__} 不支持模型类型: {self.model_config.type.value}，"
                f"支持的类型: {self.supported_model_types}"
            )

    def run_safe(self) -> List["TaskResult"]:
        """带错误捕获的完整执行流程，确保 teardown 一定执行。"""
        from benchmark.llama_benchmark.utils.logging import get_logger
        logger = get_logger(__name__)

        self.validate_config()
        try:
            logger.info(f"[{self.model_config.name}] 开始初始化...")
            self.setup()
            logger.info(f"[{self.model_config.name}] 开始执行 benchmark...")
            return self.run()
        except Exception as e:
            logger.error(
                f"[{self.model_config.name}] 执行失败: {e}\n{traceback.format_exc()}"
            )
            return self._make_error_results(e)
        finally:
            try:
                self.teardown()
            except Exception as e:
                logger.warning(f"[{self.model_config.name}] teardown 失败: {e}")

    def _make_error_results(self, error: Exception) -> List["TaskResult"]:
        from benchmark.llama_benchmark.core.result import BenchmarkStatus, MetricResult, TaskResult

        return [
            TaskResult(
                task_name="unknown",
                model_name=self.model_config.name,
                metrics=[],
                num_samples=0,
                duration_seconds=0.0,
                status=BenchmarkStatus.ERROR,
                error_message=str(error),
            )
        ]
