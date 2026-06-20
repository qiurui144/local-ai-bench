"""llama_benchmark dataset supply-chain contracts (idiom: tests/translation/test_flores_loading.py).

Three contracts, no network:

1. HF call shape: every ``load_dataset`` call carries NO ``trust_remote_code``
   (remote-code-execution surface removed repo-wide) and pins ``revision`` to a
   full commit SHA, env-overridable per dataset.
2. Revision helpers: env override works; set-but-empty env falls back to the
   default SHA (never passes ``revision=""``).
3. Unloadable sources (AMI ``Edinburgh/ami`` / AISHELL-4 ``speechio/aishell4``,
   both dead on the Hub) fail LOUDLY: ``_load_hf`` raises an actionable
   ``RuntimeError``, and the base-class fallback to builtin synthetic samples
   emits a WARNING — it can never silently pose as real benchmark data.
"""

import logging
import sys
import types

import pytest

from benchmark.llama_benchmark.datasets import (
    aishell4_dataset,
    ami_dataset,
    callhome_dataset,
    gsm8k_dataset,
    hellaswag_dataset,
    librispeech_dataset,
    mmlu_dataset,
)

REVISION_ENVS = [
    ("GSM8K_REVISION", gsm8k_dataset.gsm8k_revision, gsm8k_dataset.GSM8K_DEFAULT_REVISION),
    ("MMLU_REVISION", mmlu_dataset.mmlu_revision, mmlu_dataset.MMLU_DEFAULT_REVISION),
    (
        "HELLASWAG_REVISION",
        hellaswag_dataset.hellaswag_revision,
        hellaswag_dataset.HELLASWAG_DEFAULT_REVISION,
    ),
    (
        "LIBRISPEECH_REVISION",
        librispeech_dataset.librispeech_revision,
        librispeech_dataset.LIBRISPEECH_DEFAULT_REVISION,
    ),
    (
        "CALLHOME_REVISION",
        callhome_dataset.callhome_revision,
        callhome_dataset.CALLHOME_DEFAULT_REVISION,
    ),
]


def _install_fake_datasets(monkeypatch, rows):
    """Install a fake ``datasets`` module capturing the load_dataset call."""
    captured = {}

    def fake_load_dataset(path, *args, **kwargs):
        captured["path"] = path
        captured["args"] = args
        captured.update(kwargs)
        return rows

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)
    return captured


# --------------------------------------------------------------------------- #
# 1. HF call shape: no trust_remote_code, revision pinned to a full SHA
# --------------------------------------------------------------------------- #
def test_gsm8k_no_trust_remote_code_and_pinned(monkeypatch):
    monkeypatch.delenv("GSM8K_REVISION", raising=False)
    captured = _install_fake_datasets(
        monkeypatch, [{"question": "1+1?", "answer": "2 #### 2"}]
    )

    samples = gsm8k_dataset.GSM8KDataset()._load_hf()

    assert "trust_remote_code" not in captured, "remote code execution surface must be gone"
    assert captured["path"] == "openai/gsm8k"
    assert captured["args"] == ("main",)
    assert captured.get("split") == "test"
    assert captured.get("revision") == gsm8k_dataset.GSM8K_DEFAULT_REVISION
    assert samples == [{"question": "1+1?", "answer": "2 #### 2"}]


def test_mmlu_no_trust_remote_code_and_pinned(monkeypatch):
    monkeypatch.delenv("MMLU_REVISION", raising=False)
    captured = _install_fake_datasets(
        monkeypatch,
        [{"question": "q", "choices": ["a", "b", "c", "d"], "answer": 1, "subject": "math"}],
    )

    samples = mmlu_dataset.MMLUDataset()._load_hf()

    assert "trust_remote_code" not in captured, "remote code execution surface must be gone"
    assert captured["path"] == "cais/mmlu"
    assert captured["args"] == ("all",)
    assert captured.get("split") == "test"
    assert captured.get("revision") == mmlu_dataset.MMLU_DEFAULT_REVISION
    assert samples[0]["answer"] == "B"


def test_hellaswag_no_trust_remote_code_and_pinned(monkeypatch):
    monkeypatch.delenv("HELLASWAG_REVISION", raising=False)
    captured = _install_fake_datasets(
        monkeypatch,
        [{"activity_label": "x", "ctx": "c", "endings": ["1", "2", "3", "4"], "label": "0"}],
    )

    samples = hellaswag_dataset.HellaSwagDataset()._load_hf()

    assert "trust_remote_code" not in captured, "remote code execution surface must be gone"
    assert captured["path"] == "Rowan/hellaswag"
    assert captured["args"] == ()
    assert captured.get("split") == "validation"
    assert captured.get("revision") == hellaswag_dataset.HELLASWAG_DEFAULT_REVISION
    assert samples[0]["label"] == 0


def test_librispeech_no_trust_remote_code_and_pinned(monkeypatch):
    monkeypatch.delenv("LIBRISPEECH_REVISION", raising=False)
    captured = _install_fake_datasets(
        monkeypatch,
        [{"audio": {"path": "/a.flac", "array": None, "sampling_rate": 16000},
          "text": "hello", "speaker_id": 7}],
    )

    samples = librispeech_dataset.LibriSpeechDataset()._load_hf()

    assert "trust_remote_code" not in captured, "remote code execution surface must be gone"
    assert captured["path"] == "openslr/librispeech_asr"
    # legacy "test.clean" must map to parquet-era ("clean", split="test")
    assert captured["args"] == ("clean",)
    assert captured.get("split") == "test"
    assert captured.get("revision") == librispeech_dataset.LIBRISPEECH_DEFAULT_REVISION
    assert samples[0]["transcription"] == "HELLO"


def test_callhome_no_trust_remote_code_and_pinned(monkeypatch):
    monkeypatch.delenv("CALLHOME_REVISION", raising=False)
    captured = _install_fake_datasets(
        monkeypatch,
        [{"audio": {"path": "/c.wav"}, "duration": 1.0, "timestamps_start": [], "id": "c1"}],
    )

    samples = callhome_dataset.CallhomeDataset()._load_hf()

    assert "trust_remote_code" not in captured, "remote code execution surface must be gone"
    assert captured["path"] == "diarizers-community/callhome"
    assert captured["args"] == ("en",)
    assert captured.get("split") == "data"
    assert captured.get("revision") == callhome_dataset.CALLHOME_DEFAULT_REVISION
    assert samples[0]["call_id"] == "c1"


@pytest.mark.parametrize("env_name,revision_fn,default", REVISION_ENVS)
def test_default_revision_is_full_commit_sha(env_name, revision_fn, default):
    assert len(default) == 40, f"{env_name} default must be a full commit SHA pin"
    assert all(c in "0123456789abcdef" for c in default)


# --------------------------------------------------------------------------- #
# 2. Revision env override + set-but-empty fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env_name,revision_fn,default", REVISION_ENVS)
def test_revision_env_override(monkeypatch, env_name, revision_fn, default):
    monkeypatch.setenv(env_name, "deadbeefcafe")
    assert revision_fn() == "deadbeefcafe"


@pytest.mark.parametrize("env_name,revision_fn,default", REVISION_ENVS)
def test_revision_empty_env_falls_back_to_default(monkeypatch, env_name, revision_fn, default):
    """ENV="" (set-but-empty) must not pass revision=""."""
    monkeypatch.setenv(env_name, "")
    assert revision_fn() == default


# --------------------------------------------------------------------------- #
# 3. Unloadable sources fail LOUDLY; builtin fallback is never silent
# --------------------------------------------------------------------------- #
def test_ami_load_hf_raises_actionable_error():
    with pytest.raises(RuntimeError, match="Edinburgh/ami"):
        ami_dataset.AMIDataset()._load_hf()


def test_aishell4_load_hf_raises_actionable_error():
    with pytest.raises(RuntimeError, match="speechio/aishell4"):
        aishell4_dataset.AISHELL4Dataset()._load_hf()


@pytest.mark.parametrize(
    "ds_cls,builtin",
    [
        (ami_dataset.AMIDataset, ami_dataset.AMI_BUILTIN_SAMPLES),
        (aishell4_dataset.AISHELL4Dataset, aishell4_dataset.AISHELL4_BUILTIN_SAMPLES),
    ],
)
def test_unloadable_fallback_to_builtin_is_loud(caplog, ds_cls, builtin):
    with caplog.at_level(logging.WARNING, logger="benchmark.llama_benchmark.datasets.base_dataset"):
        samples = ds_cls().load()

    assert samples == [dict(s) for s in builtin]
    warning_text = " ".join(r.getMessage() for r in caplog.records)
    assert "内置合成样本" in warning_text, "builtin fallback must be logged loudly"
    assert "不是真实 benchmark 分数" in warning_text


def test_hf_failure_fallback_warns_with_cause(monkeypatch, caplog):
    """Generic HF failure (e.g. network) also falls back loudly, naming the cause."""

    def boom(*args, **kwargs):
        raise ConnectionError("offline-test-not-real")

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = boom
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    with caplog.at_level(logging.WARNING, logger="benchmark.llama_benchmark.datasets.base_dataset"):
        samples = gsm8k_dataset.GSM8KDataset().load()

    assert samples == list(gsm8k_dataset.GSM8K_BUILTIN_SAMPLES)
    warning_text = " ".join(r.getMessage() for r in caplog.records)
    assert "ConnectionError" in warning_text
    assert "offline-test-not-real" in warning_text
