"""run_diarization must return a contract-compliant TaskResult (pre-existing bug,
registered in commit 4c85568's message).

The TaskResult dataclass (core/result.py) requires:
- ``metrics``: List[MetricResult] (diarization passed a plain dict)
- ``num_samples``: int and ``duration_seconds``: float (diarization omitted both)
- ``status``: BenchmarkStatus enum (diarization passed a raw str, breaking
  ``to_dict()`` which calls ``self.status.value``)

Also: ``ds.load(dataset_path=..., num_samples=...)`` mismatched the base
``AbstractDataset.load()`` signature (no args), so dataset-track loads always
TypeError'd into the local fallback. ``num_samples`` must instead go through
the dataset constructor, like gsm8k/mmlu do.

No audio deps, no network: backend and datasets are stubbed.
"""

import json

from benchmark.llama_benchmark.benchmarks.speaker import diarization
from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig, ThresholdConfig
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    TaskResult,
)
from benchmark.llama_benchmark.datasets import ami_dataset
from benchmark.llama_benchmark.datasets.base_dataset import SYNTHETIC_FALLBACK_WARNING

REF_SEGMENTS = [(0.0, 2.0, "SPK_A"), (2.5, 5.0, "SPK_B")]


class _StubBackend:
    """Returns the reference segments verbatim → DER == 0."""

    def diarize(self, audio_path):
        return REF_SEGMENTS, 50.0  # (hyp_segments, latency_ms)


class _FailingBackend:
    def diarize(self, audio_path):
        raise RuntimeError("inference exploded (forced)")


class _StubDataset:
    """Stands in for AMIDataset: real-data flavor, no audio deps."""

    items = [
        {"audio_path": "stub-1.wav", "segments": REF_SEGMENTS, "duration": 5.0},
        {"audio_path": "stub-2.wav", "segments": REF_SEGMENTS, "duration": 5.0},
    ]
    # 记录最近一次实例(测试断言构造参数用;每次 __init__ 覆盖,避免类变量
    # dict 跨测试残留)
    last_instance = None

    def __init__(self, *args, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        type(self).last_instance = self
        self.synthetic_fallback = False

    def load(self):  # base AbstractDataset.load() signature: no args
        return self.items

    def __iter__(self):
        return iter(self.items)


class _StubDatasetFallback(_StubDataset):
    def load(self):
        self.synthetic_fallback = True
        return self.items


def _run(backend, monkeypatch, dataset_cls=_StubDataset, **cfg_kwargs):
    monkeypatch.setattr(ami_dataset, "AMIDataset", dataset_cls)
    return diarization.run_diarization(
        backend=backend,
        task_cfg=BenchmarkTaskConfig(**cfg_kwargs),
        model_name="stub-model",
        datasets=["ami"],
    )


# --------------------------------------------------------------------------- #
# 1. TaskResult dataclass contract
# --------------------------------------------------------------------------- #
def test_success_result_satisfies_dataclass_contract(monkeypatch):
    result = _run(_StubBackend(), monkeypatch)

    assert isinstance(result, TaskResult)
    # metrics: List[MetricResult], not a plain dict
    assert isinstance(result.metrics, list)
    assert result.metrics, "metrics must not be empty on success"
    assert all(isinstance(m, MetricResult) for m in result.metrics)
    # required fields, correct types
    assert isinstance(result.num_samples, int)
    assert result.num_samples == 2
    assert isinstance(result.duration_seconds, float)
    assert result.duration_seconds >= 0.0
    # status: enum, not str
    assert isinstance(result.status, BenchmarkStatus)
    assert result.status == BenchmarkStatus.PASS
    # get_metric works against the list shape
    der = result.get_metric("der")
    assert der is not None and der.value == 0.0


def test_success_result_roundtrips_to_dict(monkeypatch):
    result = _run(_StubBackend(), monkeypatch)
    d = result.to_dict()  # str status would raise: str has no .value
    assert d["status"] == "pass"
    assert d["num_samples"] == 2
    assert isinstance(d["metrics"], list)
    json.dumps(d, default=str)  # fully serializable


def test_threshold_failure_yields_fail_enum(monkeypatch):
    result = _run(
        _StubBackend(),
        monkeypatch,
        thresholds={"num_evaluated": ThresholdConfig(min_value=10)},
    )
    assert result.status is BenchmarkStatus.FAIL
    assert result.to_dict()["status"] == "fail"


def test_no_samples_error_result_is_contract_compliant(monkeypatch):
    monkeypatch.setattr(diarization, "_load_samples", lambda *a, **k: ([], False))
    result = diarization.run_diarization(
        backend=_StubBackend(),
        task_cfg=BenchmarkTaskConfig(),
        model_name="stub-model",
    )
    assert result.status is BenchmarkStatus.ERROR
    assert result.metrics == []
    assert result.num_samples == 0
    assert isinstance(result.duration_seconds, float)
    assert result.error_message
    result.to_dict()


def test_all_inference_failures_error_result_is_contract_compliant(monkeypatch):
    result = _run(_FailingBackend(), monkeypatch)
    assert result.status is BenchmarkStatus.ERROR
    assert result.metrics == []
    assert result.num_samples == 0
    assert isinstance(result.duration_seconds, float)
    assert "inference exploded" in result.error_message
    result.to_dict()


# --------------------------------------------------------------------------- #
# 2. Dataset-track load: base load() takes no args; num_samples via constructor
# --------------------------------------------------------------------------- #
def test_dataset_track_passes_num_samples_through_constructor(monkeypatch):
    result = _run(_StubBackend(), monkeypatch, num_samples=1)
    assert _StubDataset.last_instance.init_kwargs.get("num_samples") == 1
    # load() succeeded (no TypeError → no fallback flag)
    assert "synthetic_fallback" not in result.metadata
    assert result.num_samples == 1  # task_cfg.num_samples truncation still applies


def test_dataset_track_loader_failure_surfaces_flagged_fallback(monkeypatch):
    """AMIDataset 加载失败时 base load() 回退 builtin 且 flag 必须传播 —
    回归点是「不再 TypeError」。显式注入 RuntimeError,不依赖真 loader
    当前恰好是死仓的状态(那是供应链事实,不是本测试的契约)。"""
    from benchmark.llama_benchmark.datasets import ami_dataset

    def boom(self):
        raise RuntimeError("injected: upstream unavailable")

    monkeypatch.setattr(ami_dataset.AMIDataset, "_load_hf", boom)
    samples, synthetic_fallback = diarization._load_samples(
        BenchmarkTaskConfig(), ["ami"]
    )
    assert synthetic_fallback is True
    assert samples  # AMI builtin samples reached the diarization track


# --------------------------------------------------------------------------- #
# 3. synthetic_fallback metadata behavior stays intact end-to-end
# --------------------------------------------------------------------------- #
def test_synthetic_fallback_metadata_intact_in_taskresult(monkeypatch):
    result = _run(_StubBackend(), monkeypatch, dataset_cls=_StubDatasetFallback)
    assert result.metadata["synthetic_fallback"] is True
    assert result.metadata["warning"] == SYNTHETIC_FALLBACK_WARNING
    d = result.to_dict()
    assert d["metadata"]["synthetic_fallback"] is True


def test_real_data_metadata_has_no_fallback_keys(monkeypatch):
    result = _run(_StubBackend(), monkeypatch)
    assert "synthetic_fallback" not in result.metadata
    assert "warning" not in result.metadata
    assert result.metadata["datasets"] == ["ami"]
