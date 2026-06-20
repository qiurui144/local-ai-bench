"""Flores loader supply-chain + provenance honesty.

Three contracts (per 2026-06-10 intake threat/review findings):

1. No ``trust_remote_code`` — the HF call loads the non-gated pure-parquet
   mirror ``haoranxu/FLORES-200`` pinned to a commit SHA (env-overridable via
   ``FLORES_DATASET`` / ``FLORES_REVISION``) so no upstream-repo code can
   ever execute locally and the data cannot drift silently.
2. Falling back to the 5 builtin synthetic pairs must be LOUD (a warning
   log), never silent.
3. A translation run scored on builtin pairs must say so in the report and
   can never masquerade as a Flores-200 PASS (verdict >= WARN).
"""
import logging
import sys
import types

from benchmark.translation import datasets as tr_datasets
from benchmark.translation.datasets import TranslationPair


# --------------------------------------------------------------------------- #
# 1. HF call shape: parquet revision pinned, no remote code
# --------------------------------------------------------------------------- #
def test_load_flores_hf_pins_sha_no_trust_remote_code(monkeypatch):
    captured = {}

    def fake_load_dataset(path, config, **kwargs):
        captured["path"] = path
        captured["config"] = config
        captured.update(kwargs)
        return [{"zh-en": {"zh": "你好", "en": "hello"}}]

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)
    monkeypatch.delenv("FLORES_REVISION", raising=False)
    monkeypatch.delenv("FLORES_DATASET", raising=False)

    pairs = tr_datasets._load_flores_hf("zh", "en", "devtest")

    assert "trust_remote_code" not in captured, "remote code execution surface must be gone"
    assert captured["path"] == "haoranxu/FLORES-200"
    assert captured["config"] == "zh-en"
    assert captured.get("split") == "test"   # mirror's test split == devtest
    assert captured.get("revision") == tr_datasets._FLORES_DEFAULT_REVISION
    assert len(captured["revision"]) == 40, "default revision must be a full commit SHA pin"
    assert pairs[0].src == "你好" and pairs[0].ref == "hello"
    assert pairs[0].source == "flores"


def test_load_flores_hf_revision_env_override(monkeypatch):
    captured = {}

    def fake_load_dataset(path, config, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return []

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)
    monkeypatch.setenv("FLORES_REVISION", "deadbeefcafe")
    monkeypatch.setenv("FLORES_DATASET", "my-org/flores-mirror")

    tr_datasets._load_flores_hf("zh", "en", "devtest")
    assert captured.get("revision") == "deadbeefcafe"
    assert captured.get("path") == "my-org/flores-mirror"


def test_load_flores_hf_empty_env_falls_back_to_defaults(monkeypatch):
    """FLORES_REVISION="" (set-but-empty) must not pass revision=""."""
    captured = {}

    def fake_load_dataset(path, config, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return []

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)
    monkeypatch.setenv("FLORES_REVISION", "")
    monkeypatch.setenv("FLORES_DATASET", "")

    tr_datasets._load_flores_hf("zh", "en", "devtest")
    assert captured.get("revision") == tr_datasets._FLORES_DEFAULT_REVISION
    assert captured.get("path") == tr_datasets._FLORES_DEFAULT_DATASET


# --------------------------------------------------------------------------- #
# 2. Loud fallback
# --------------------------------------------------------------------------- #
def test_load_flores_fallback_is_loud(monkeypatch, caplog):
    def boom(*a, **kw):
        raise RuntimeError("HF unreachable")

    monkeypatch.setattr(tr_datasets, "_load_flores_hf", boom)
    monkeypatch.delenv("TRANSLATION_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)

    with caplog.at_level(logging.WARNING):
        pairs = tr_datasets.load_flores("zh", "en")

    assert all(p.source == "builtin" for p in pairs)
    warned = [r for r in caplog.records if "builtin" in r.getMessage().lower()]
    assert warned, "fallback to synthetic builtin pairs must log a WARNING"


# --------------------------------------------------------------------------- #
# 3. Report provenance: builtin pairs can never pose as a Flores PASS
# --------------------------------------------------------------------------- #
def test_translation_dimension_flags_builtin_provenance(monkeypatch):
    from pathlib import Path

    import benchmark.translation.dimension as trd

    builtin_pairs = [
        TranslationPair(src="你好", ref="hello", src_lang="zh", tgt_lang="en",
                        source="builtin"),
    ]
    monkeypatch.setattr(trd, "load_flores", lambda *a, **kw: list(builtin_pairs))
    monkeypatch.setattr(trd, "load_custom_jsonl", lambda *a, **kw: [])
    monkeypatch.setattr(
        trd, "run_translation",
        lambda *a, **kw: {"verdict": "PASS", "verdict_reasons": []},
    )
    monkeypatch.setattr(
        trd, "run_translation_performance", lambda *a, **kw: {}
    )

    class _M:
        name = "stub"
        hf_repo = "org/stub"

    out = trd.run_translation_dimension(_M(), {"directions": ["zh->en"]}, Path("."))

    assert out["dataset_sources"]["zh->en"] == ["builtin"]
    assert out["verdict"] != "PASS", "builtin synthetic data must not yield a clean PASS"
    assert any("builtin" in r.lower() or "synthetic" in r.lower()
               for r in out["verdict_reasons"])
