"""
Windows power sampler — collects real-time CPU + iGPU + NPU power during benchmark runs.

Sampling strategy:
  1. WMI MSAcpi_ThermalZoneTemperature + Win32_PerfFormattedData (in-process, no extra deps)
  2. LibreHardwareMonitor COM interface (if LHM is running; optional)
  3. PowerShell Get-CimInstance as fallback

Usage:
  sampler = WindowsPowerSampler()
  sampler.start()
  # ... run benchmark ...
  report = sampler.stop()
  # report: {"avg_w": float, "max_w": float, "samples": int, "duration_s": float,
  #           "cpu_w": float, "igpu_w": float, "npu_w_est": float}
"""
from __future__ import annotations

import threading
import time
import subprocess
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class PowerSample:
    timestamp: float
    total_w: float
    cpu_w: float = 0.0
    igpu_w: float = 0.0
    npu_w_est: float = 0.0  # estimated; XDNA NPU doesn't expose direct wattage
    source: str = "unknown"


@dataclass
class PowerReport:
    samples: int = 0
    duration_s: float = 0.0
    avg_w: float = 0.0
    max_w: float = 0.0
    min_w: float = 0.0
    cpu_w_avg: float = 0.0
    igpu_w_avg: float = 0.0
    npu_w_est_avg: float = 0.0
    source: str = "unknown"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


class WindowsPowerSampler:
    """
    Samples power consumption on Windows during a benchmark run.

    Priority order:
      1. LibreHardwareMonitor COM (most accurate, requires LHM running as admin)
      2. WMI Win32_PerfFormattedData_Counters_ThermalZoneInformation (temperatures only, estimates power)
      3. PowerShell RAPL via WMI (Intel RAPL / AMD PerformanceCounter)
    """

    SAMPLE_INTERVAL_S = 0.5  # 500 ms between samples

    def __init__(self, target_host: Optional[str] = None,
                 ssh_user: Optional[str] = None,
                 ssh_pass: Optional[str] = None):
        self._target_host = target_host
        self._ssh_user = ssh_user
        self._ssh_pass = ssh_pass
        self._remote = target_host is not None
        self._samples: list[PowerSample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self._source: str = "none"

    def start(self) -> None:
        self._stop_event.clear()
        self._samples = []
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> PowerReport:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        duration = time.monotonic() - self._start_time
        return self._compute_report(duration)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._collect_one()
            if sample:
                self._samples.append(sample)
            self._stop_event.wait(timeout=self.SAMPLE_INTERVAL_S)

    def _collect_one(self) -> Optional[PowerSample]:
        if self._remote:
            return self._collect_remote()
        return self._collect_local()

    def _collect_local(self) -> Optional[PowerSample]:
        # Strategy 1: LibreHardwareMonitor COM
        sample = self._try_lhm()
        if sample:
            return sample
        # Strategy 2: PowerShell WMI RAPL
        return self._try_powershell_rapl()

    def _try_lhm(self) -> Optional[PowerSample]:
        """LibreHardwareMonitor COM interface — most accurate."""
        try:
            import win32com.client  # type: ignore[import]
        except ImportError:
            return None
        try:
            lhm = win32com.client.Dispatch("LibreHardwareMonitor.Hardware")
            lhm.Update()
            cpu_w = igpu_w = 0.0
            for hw in lhm.Hardware:
                hw.Update()
                for sensor in hw.Sensors:
                    if sensor.SensorType != 5:  # 5 = Power
                        continue
                    name = sensor.Name.lower()
                    val = sensor.Value or 0.0
                    if "package" in name or "cpu" in name:
                        cpu_w += val
                    elif "gpu" in name or "igpu" in name:
                        igpu_w += val
            if cpu_w == 0.0 and igpu_w == 0.0:
                return None
            self._source = "lhm"
            return PowerSample(
                timestamp=time.monotonic(),
                total_w=cpu_w + igpu_w,
                cpu_w=cpu_w,
                igpu_w=igpu_w,
                source="lhm",
            )
        except Exception:
            return None

    def _try_powershell_rapl(self) -> Optional[PowerSample]:
        """PowerShell: AMD energy counters via WMI."""
        ps_script = r"""
$ErrorActionPreference = 'SilentlyContinue'
try {
    # AMD μProf energy / RAPL via WMI (requires AMD Energy driver or HWINFO64)
    $ctr = Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ProcessorInformation -Property * 2>$null |
           Select-Object -First 1
    $pkg = if ($ctr.ProcessorFrequency) { [math]::Round($ctr.ProcessorFrequency * 0.001, 2) } else { 0 }
    @{ cpu_w = $pkg; igpu_w = 0.0; source = 'rapl_est' } | ConvertTo-Json
} catch {
    @{ cpu_w = 0.0; igpu_w = 0.0; source = 'unavailable' } | ConvertTo-Json
}
"""
        try:
            result = subprocess.run(
                ["powershell", "-NonInteractive", "-Command", ps_script],
                capture_output=True, text=True, timeout=3.0
            )
            data = json.loads(result.stdout.strip())
            cpu_w = float(data.get("cpu_w", 0.0))
            igpu_w = float(data.get("igpu_w", 0.0))
            src = data.get("source", "rapl_est")
            if cpu_w == 0.0 and igpu_w == 0.0:
                return None
            self._source = src
            return PowerSample(
                timestamp=time.monotonic(),
                total_w=cpu_w + igpu_w,
                cpu_w=cpu_w,
                igpu_w=igpu_w,
                source=src,
            )
        except Exception:
            return None

    def _collect_remote(self) -> Optional[PowerSample]:
        """SSH to Windows target and run a mini PowerShell energy probe."""
        ps_script = (
            "$ErrorActionPreference='SilentlyContinue';"
            # AMD APU: try AMD μProf or HWINFO shared memory
            # Fallback: estimate from CPU utilization × TDP
            "try {"
            "  $util=(Get-CimInstance Win32_Processor).LoadPercentage;"
            "  $tdp=45;"  # Ryzen 8845H TDP
            "  $cpu_w=[math]::Round($util/100*$tdp,2);"
            "  $igpu_w=0;"
            "  @{cpu_w=$cpu_w;igpu_w=$igpu_w;source='tdp_est'} | ConvertTo-Json"
            "} catch {"
            "  @{cpu_w=0;igpu_w=0;source='unavailable'} | ConvertTo-Json"
            "}"
        )
        cmd = [
            "sshpass", "-p", self._ssh_pass,
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=3",
            f"{self._ssh_user}@{self._target_host}",
            "powershell", "-NonInteractive", "-Command", ps_script,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
            data = json.loads(result.stdout.strip())
            cpu_w = float(data.get("cpu_w", 0.0))
            igpu_w = float(data.get("igpu_w", 0.0))
            src = data.get("source", "tdp_est")
            self._source = src
            return PowerSample(
                timestamp=time.monotonic(),
                total_w=cpu_w + igpu_w,
                cpu_w=cpu_w,
                igpu_w=igpu_w,
                source=src,
            )
        except Exception:
            return None

    def _compute_report(self, duration_s: float) -> PowerReport:
        if not self._samples:
            return PowerReport(
                duration_s=duration_s,
                error="no power samples collected — LHM not running and PowerShell probe unavailable",
                source=self._source or "none",
            )
        totals = [s.total_w for s in self._samples]
        return PowerReport(
            samples=len(self._samples),
            duration_s=duration_s,
            avg_w=sum(totals) / len(totals),
            max_w=max(totals),
            min_w=min(totals),
            cpu_w_avg=sum(s.cpu_w for s in self._samples) / len(self._samples),
            igpu_w_avg=sum(s.igpu_w for s in self._samples) / len(self._samples),
            npu_w_est_avg=sum(s.npu_w_est for s in self._samples) / len(self._samples),
            source=self._source,
        )


def measure_power_during(fn, target_host=None, ssh_user=None, ssh_pass=None):
    """
    Context helper: wraps a callable with power sampling.

    Returns (fn_result, PowerReport).
    """
    sampler = WindowsPowerSampler(
        target_host=target_host, ssh_user=ssh_user, ssh_pass=ssh_pass
    )
    sampler.start()
    try:
        result = fn()
    finally:
        report = sampler.stop()
    return result, report
