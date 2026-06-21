"""Offline unit tests for the Windows power sampler — no actual Windows/SSH needed."""
import time
import pytest
from benchmark.power.windows_sampler import (
    WindowsPowerSampler,
    PowerReport,
    PowerSample,
    measure_power_during,
)


def test_power_report_to_dict():
    r = PowerReport(samples=10, duration_s=5.0, avg_w=35.0, max_w=45.0, min_w=25.0)
    d = r.to_dict()
    assert d["samples"] == 10
    assert d["avg_w"] == pytest.approx(35.0)
    assert "source" in d


def test_sampler_stop_without_samples_returns_error():
    sampler = WindowsPowerSampler()
    sampler.start()
    # Immediately stop before any sample can be collected from unavailable WMI
    time.sleep(0.1)
    report = sampler.stop()
    # On Linux (test environment) all probes fail → error field set
    if report.samples == 0:
        assert report.error is not None
    else:
        # If somehow a sample was collected, check structure
        assert report.avg_w >= 0.0


def test_sampler_no_crash_on_missing_wmi():
    """Sampler must not crash even when WMI and PowerShell are unavailable."""
    sampler = WindowsPowerSampler()
    sampler.start()
    time.sleep(0.2)
    report = sampler.stop()
    assert isinstance(report, PowerReport)
    assert isinstance(report.to_dict(), dict)


def test_measure_power_during_wraps_callable():
    counter = {"n": 0}

    def work():
        counter["n"] += 1
        return 42

    result, report = measure_power_during(work)
    assert result == 42
    assert counter["n"] == 1
    assert isinstance(report, PowerReport)


def test_sampler_compute_report_with_injected_samples():
    """Inject fake samples and verify arithmetic."""
    sampler = WindowsPowerSampler()
    sampler._samples = [
        PowerSample(timestamp=0.0, total_w=30.0, cpu_w=25.0, igpu_w=5.0, source="test"),
        PowerSample(timestamp=0.5, total_w=40.0, cpu_w=30.0, igpu_w=10.0, source="test"),
        PowerSample(timestamp=1.0, total_w=50.0, cpu_w=35.0, igpu_w=15.0, source="test"),
    ]
    sampler._source = "test"
    report = sampler._compute_report(1.0)
    assert report.samples == 3
    assert report.avg_w == pytest.approx(40.0)
    assert report.max_w == pytest.approx(50.0)
    assert report.min_w == pytest.approx(30.0)
    assert report.cpu_w_avg == pytest.approx(30.0)
    assert report.igpu_w_avg == pytest.approx(10.0)


def test_sampler_remote_flag_set():
    sampler = WindowsPowerSampler(
        target_host="192.168.100.201",
        ssh_user="user",
        ssh_pass="pass",
    )
    assert sampler._remote is True
    assert sampler._target_host == "192.168.100.201"
