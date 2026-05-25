"""结果数据类：不可变结构，用于存储 benchmark 运行结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class BenchmarkStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class MetricResult:
    """单个指标的计算结果。"""

    name: str
    value: Optional[float]
    unit: str = ""
    higher_is_better: bool = True
    threshold: Optional[float] = None
    status: BenchmarkStatus = BenchmarkStatus.PASS
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class SampleResult:
    """单个样本的推理结果，用于错误分析。"""

    sample_id: str
    input: str
    expected: str
    predicted: str
    correct: bool
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """单个 benchmark 任务（如 MMLU）的完整结果。"""

    task_name: str
    model_name: str
    metrics: List[MetricResult]
    num_samples: int
    duration_seconds: float
    status: BenchmarkStatus
    error_message: Optional[str] = None
    sample_results: List[SampleResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_metric(self, name: str) -> Optional[MetricResult]:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "status": self.status.value,
            "num_samples": self.num_samples,
            "duration_seconds": round(self.duration_seconds, 3),
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "metrics": [m.to_dict() for m in self.metrics],
            "metadata": self.metadata,
        }


@dataclass
class ModelBenchmarkResult:
    """单个模型的全部 benchmark 任务结果汇总。"""

    model_name: str
    model_type: str
    backend: str
    task_results: List[TaskResult]
    start_time: datetime
    end_time: datetime
    system_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def pass_count(self) -> int:
        return sum(1 for t in self.task_results if t.status == BenchmarkStatus.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for t in self.task_results if t.status == BenchmarkStatus.FAIL)

    @property
    def error_count(self) -> int:
        return sum(1 for t in self.task_results if t.status == BenchmarkStatus.ERROR)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "backend": self.backend,
            "duration_seconds": round(self.duration_seconds, 3),
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "error_count": self.error_count,
            "system_info": self.system_info,
            "task_results": [t.to_dict() for t in self.task_results],
        }


@dataclass
class BenchmarkSuiteResult:
    """整个测试套件的顶层结果，包含所有模型的结果。"""

    suite_name: str
    model_results: List[ModelBenchmarkResult]
    start_time: datetime
    end_time: datetime
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    def compute_summary(self) -> None:
        """计算汇总统计信息。"""
        total_tasks = sum(len(mr.task_results) for mr in self.model_results)
        total_pass = sum(mr.pass_count for mr in self.model_results)
        total_fail = sum(mr.fail_count for mr in self.model_results)
        total_error = sum(mr.error_count for mr in self.model_results)

        self.summary = {
            "total_models": len(self.model_results),
            "total_tasks": total_tasks,
            "passed": total_pass,
            "failed": total_fail,
            "errored": total_error,
            "pass_rate": round(total_pass / total_tasks, 4) if total_tasks > 0 else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.compute_summary()
        return {
            "run_id": self.run_id,
            "suite_name": self.suite_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": round(self.duration_seconds, 3),
            "summary": self.summary,
            "config_snapshot": self.config_snapshot,
            "model_results": [mr.to_dict() for mr in self.model_results],
        }
