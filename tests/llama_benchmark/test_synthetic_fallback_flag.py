"""Synthetic-fallback must be machine-detectable (review finding B-I4).

The earlier "fail loud" remediation only logged a WARNING when dataset loaders
fell back to builtin synthetic samples — callers swallowed the failure and the
resulting scores were indistinguishable from real benchmark scores. Contracts:

1. ``AbstractDataset.load()`` sets ``synthetic_fallback=True`` on fallback,
   stays ``False`` on success (and on local ``dataset_path`` loads).
2. Diarization's local fallback (``_load_samples``) reports the flag too,
   both for its own ``except → _builtin_samples()`` path and by propagating
   a dataset-level ``synthetic_fallback`` attribute.
3. ``synthetic_fallback_metadata()`` / ``_result_metadata()`` inject
   ``metadata["synthetic_fallback"]`` + ``metadata["warning"]`` into the
   TaskResult metadata so reporters/consumers can detect the masquerade.

No network, no real datasets.
"""

import sys
import types
from typing import Any, Dict, List

from benchmark.llama_benchmark.benchmarks.speaker import diarization
from benchmark.llama_benchmark.core.config import BenchmarkTaskConfig
from benchmark.llama_benchmark.datasets import ami_dataset, gsm8k_dataset
from benchmark.llama_benchmark.datasets.base_dataset import (
    SYNTHETIC_FALLBACK_WARNING,
    AbstractDataset,
    synthetic_fallback_metadata,
)


class _ToyDataset(AbstractDataset):
    """Minimal concrete dataset: HF load behavior is injectable."""

    def __init__(self, hf_raises: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hf_raises = hf_raises

    def _load_hf(self) -> List[Dict[str, Any]]:
        if self._hf_raises:
            raise RuntimeError("HF source unavailable (forced)")
        return [{"source": "hf"}]

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return [{"source": "builtin"}]


# --------------------------------------------------------------------------- #
# 1. Base dataset flag
# --------------------------------------------------------------------------- #
def test_base_dataset_flag_default_false_before_load():
    assert _ToyDataset().synthetic_fallback is False


def test_base_dataset_flag_true_after_forced_load_failure(caplog):
    ds = _ToyDataset(hf_raises=True)
    with caplog.at_level("WARNING"):
        samples = ds.load()
    assert samples == [{"source": "builtin"}]
    assert ds.synthetic_fallback is True
    # The loud WARN from the earlier remediation must still be there.
    assert any("内置合成样本" in r.getMessage() for r in caplog.records)


def test_base_dataset_flag_false_on_successful_load():
    ds = _ToyDataset(hf_raises=False)
    assert ds.load() == [{"source": "hf"}]
    assert ds.synthetic_fallback is False


def test_real_loader_sets_flag_when_hf_module_raises(monkeypatch):
    """GSM8K loader (real subclass) through the base load() fallback path."""
    fake_mod = types.ModuleType("datasets")

    def _boom(*args, **kwargs):
        raise ConnectionError("offline (forced)")

    fake_mod.load_dataset = _boom
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    ds = gsm8k_dataset.GSM8KDataset(num_samples=2)
    samples = ds.load()
    assert len(samples) == 2  # builtin samples, truncated
    assert ds.synthetic_fallback is True


# --------------------------------------------------------------------------- #
# 2. Metadata injection helper (wired into gsm8k/mmlu TaskResult metadata)
# --------------------------------------------------------------------------- #
def test_synthetic_fallback_metadata_injects_flag_and_warning():
    ds = _ToyDataset(hf_raises=True)
    ds.load()
    meta = synthetic_fallback_metadata(ds)
    assert meta["synthetic_fallback"] is True
    assert meta["warning"] == SYNTHETIC_FALLBACK_WARNING


def test_synthetic_fallback_metadata_empty_on_real_data():
    ds = _ToyDataset(hf_raises=False)
    ds.load()
    assert synthetic_fallback_metadata(ds) == {}


def test_synthetic_fallback_metadata_tolerates_flagless_objects():
    assert synthetic_fallback_metadata(object()) == {}


# --------------------------------------------------------------------------- #
# 3. Diarization local fallback + TaskResult metadata
# --------------------------------------------------------------------------- #
class _StubDiarDataset:
    """Stands in for AMIDataset inside diarization._load_samples."""

    items = [{"audio_path": None, "segments": [(0.0, 1.0, "SPK_A")], "duration": 1.0}]

    def __init__(self, *args, **kwargs) -> None:
        self.synthetic_fallback = False

    def load(self, dataset_path=None, num_samples=None):
        return self.items

    def __iter__(self):
        return iter(self.items)


class _StubDiarDatasetFallback(_StubDiarDataset):
    def load(self, dataset_path=None, num_samples=None):
        self.synthetic_fallback = True
        return self.items


class _StubDiarDatasetBroken(_StubDiarDataset):
    def load(self, dataset_path=None, num_samples=None):
        raise RuntimeError("loader exploded (forced)")


def test_diarization_load_samples_real_data_no_flag(monkeypatch):
    monkeypatch.setattr(ami_dataset, "AMIDataset", _StubDiarDataset)
    samples, synthetic_fallback = diarization._load_samples(
        BenchmarkTaskConfig(), ["ami"]
    )
    assert samples
    assert synthetic_fallback is False


def test_diarization_load_samples_propagates_dataset_level_flag(monkeypatch):
    monkeypatch.setattr(ami_dataset, "AMIDataset", _StubDiarDatasetFallback)
    _, synthetic_fallback = diarization._load_samples(BenchmarkTaskConfig(), ["ami"])
    assert synthetic_fallback is True


def test_diarization_load_samples_local_except_path_sets_flag(monkeypatch):
    monkeypatch.setattr(ami_dataset, "AMIDataset", _StubDiarDatasetBroken)
    samples, synthetic_fallback = diarization._load_samples(
        BenchmarkTaskConfig(), ["ami"]
    )
    assert samples == diarization._builtin_samples()
    assert synthetic_fallback is True


def test_diarization_unknown_dataset_falls_back_with_flag():
    samples, synthetic_fallback = diarization._load_samples(
        BenchmarkTaskConfig(), ["no-such-dataset"]
    )
    assert samples == diarization._builtin_samples()
    assert synthetic_fallback is True


def test_diarization_result_metadata_carries_flag():
    meta = diarization._result_metadata(
        datasets=["ami"],
        collar=0.25,
        skip_overlap=False,
        num_samples=1,
        errors=[],
        synthetic_fallback=True,
    )
    assert meta["synthetic_fallback"] is True
    assert meta["warning"] == SYNTHETIC_FALLBACK_WARNING
    # Pre-existing keys untouched.
    assert meta["datasets"] == ["ami"]
    assert meta["num_samples"] == 1


def test_diarization_result_metadata_clean_on_real_data():
    meta = diarization._result_metadata(
        datasets=["ami"],
        collar=0.25,
        skip_overlap=False,
        num_samples=3,
        errors=["e1"],
        synthetic_fallback=False,
    )
    assert "synthetic_fallback" not in meta
    assert "warning" not in meta
