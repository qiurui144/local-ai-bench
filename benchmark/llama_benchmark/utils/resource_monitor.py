"""后台资源监控守护线程：采样 GPU / CPU / 内存使用率。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResourceSample:
    timestamp: float        # time.monotonic()
    cpu_percent: float      # 0-100
    memory_used_gb: float
    gpu_util_percent: Optional[float] = None  # None if no GPU
    gpu_vram_used_gb: Optional[float] = None
    gpu_temperature_c: Optional[float] = None


@dataclass
class ResourceSummary:
    """采样周期内的资源统计摘要。"""
    cpu_avg_percent: float = 0.0
    cpu_max_percent: float = 0.0
    memory_avg_gb: float = 0.0
    memory_max_gb: float = 0.0
    gpu_util_avg_percent: Optional[float] = None
    gpu_util_max_percent: Optional[float] = None
    gpu_util_p95_percent: Optional[float] = None
    gpu_vram_peak_gb: Optional[float] = None
    gpu_temp_max_c: Optional[float] = None
    num_samples: int = 0
    duration_seconds: float = 0.0
    # 相位区间峰值（由 record_event 标注）
    peak_gpu_util_by_phase: Optional[Dict[str, float]] = None
    peak_cpu_pct_by_phase: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ResourceMonitor:
    """100ms 采样间隔的后台守护线程。

    用法：
        monitor = ResourceMonitor(interval_ms=100)
        monitor.start()
        # ... 执行推理 ...
        monitor.stop()
        summary = monitor.get_summary()
    """

    def __init__(self, interval_ms: int = 100) -> None:
        self._interval = interval_ms / 1000.0
        self._samples: List[ResourceSample] = []
        self._events: List[tuple] = []  # (timestamp_ns, label)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._gpu_handle = None
        self._has_gpu = self._init_gpu()

    def _init_gpu(self) -> bool:
        """尝试初始化 pynvml，不可用时静默 fallback 到 CPU-only。"""
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return True
        except Exception:
            pass
        return False

    def start(self) -> None:
        """启动后台采样线程。"""
        self._samples.clear()
        self._events.clear()
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def record_event(self, label: str) -> None:
        """记录一个带时间戳的阶段事件，用于后续分析各相位的资源峰值。

        使用 time.monotonic() 与采样时间戳保持一致（避免 perf_counter 与 monotonic 的时钟偏移）。

        典型用法：
            monitor.record_event("prefill_start")
            # ... prefill 推理 ...
            monitor.record_event("decode_start")
            # ... decode 推理 ...
            monitor.record_event("decode_end")
        """
        self._events.append((time.monotonic(), label))

    def stop(self) -> None:
        """停止采样线程（立即返回，不等待最后一次采样完成）。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_summary(self) -> ResourceSummary:
        """返回整个采样周期的统计摘要。"""
        if not self._samples:
            return ResourceSummary()

        import statistics

        cpu_vals = [s.cpu_percent for s in self._samples]
        mem_vals = [s.memory_used_gb for s in self._samples]
        gpu_util_vals = [s.gpu_util_percent for s in self._samples if s.gpu_util_percent is not None]
        gpu_vram_vals = [s.gpu_vram_used_gb for s in self._samples if s.gpu_vram_used_gb is not None]
        gpu_temp_vals = [s.gpu_temperature_c for s in self._samples if s.gpu_temperature_c is not None]

        duration = time.monotonic() - self._start_time

        summary = ResourceSummary(
            cpu_avg_percent=round(statistics.mean(cpu_vals), 1),
            cpu_max_percent=round(max(cpu_vals), 1),
            memory_avg_gb=round(statistics.mean(mem_vals), 2),
            memory_max_gb=round(max(mem_vals), 2),
            num_samples=len(self._samples),
            duration_seconds=round(duration, 2),
        )

        if gpu_util_vals:
            sorted_util = sorted(gpu_util_vals)
            p95_idx = int(len(sorted_util) * 0.95)
            summary.gpu_util_avg_percent = round(statistics.mean(gpu_util_vals), 1)
            summary.gpu_util_max_percent = round(max(gpu_util_vals), 1)
            summary.gpu_util_p95_percent = round(sorted_util[min(p95_idx, len(sorted_util) - 1)], 1)

        if gpu_vram_vals:
            summary.gpu_vram_peak_gb = round(max(gpu_vram_vals), 2)

        if gpu_temp_vals:
            summary.gpu_temp_max_c = round(max(gpu_temp_vals), 1)

        # 计算各相位区间的资源峰值
        if self._events and len(self._events) >= 2:
            summary.peak_gpu_util_by_phase = self._compute_phase_peaks("gpu_util_percent")
            summary.peak_cpu_pct_by_phase = self._compute_phase_peaks("cpu_percent")

        return summary

    def _compute_phase_peaks(self, field_name: str) -> Dict[str, float]:
        """按事件区间计算指定字段的峰值。

        事件时间戳和采样时间戳均使用 time.monotonic()，直接比较无需转换。
        """
        phase_peaks: Dict[str, float] = {}
        if not self._events or not self._samples:
            return phase_peaks

        # 事件按时间排序
        events = sorted(self._events, key=lambda e: e[0])

        for i in range(len(events) - 1):
            ts_start, label_start = events[i]
            ts_end, _ = events[i + 1]

            phase_samples = [
                getattr(s, field_name)
                for s in self._samples
                if getattr(s, field_name) is not None
                and ts_start <= s.timestamp < ts_end
            ]
            if phase_samples:
                phase_peaks[label_start] = round(max(phase_samples), 1)

        return phase_peaks

    def _sample_loop(self) -> None:
        """后台采样循环，每 interval_ms 采集一次。"""
        import psutil

        while not self._stop_event.wait(timeout=self._interval):
            sample = self._collect_sample(psutil)
            self._samples.append(sample)

    def _collect_sample(self, psutil_module) -> ResourceSample:
        import psutil

        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_gb = round(mem.used / (1024 ** 3), 2)

        gpu_util = None
        gpu_vram = None
        gpu_temp = None

        if self._has_gpu and self._gpu_handle is not None:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                gpu_util = float(util.gpu)
                gpu_vram = round(mem_info.used / (1024 ** 3), 2)
                gpu_temp = float(temp)
            except Exception:
                pass

        return ResourceSample(
            timestamp=time.monotonic(),
            cpu_percent=cpu_pct,
            memory_used_gb=mem_gb,
            gpu_util_percent=gpu_util,
            gpu_vram_used_gb=gpu_vram,
            gpu_temperature_c=gpu_temp,
        )
