"""Tests for benchmark.rigor.reproducibility."""
from __future__ import annotations

from pathlib import Path


from benchmark.rigor.reproducibility import (
    CodeState,
    DataInputs,
    HardwareSpec,
    PythonEnv,
    ReproducibilitySnapshot,
    sha256_file,
)


def test_code_state_capture_returns_struct(tmp_path: Path):
    state = CodeState.capture(repo_root=tmp_path)
    # Not a git repo; the result should still be well-formed.
    assert state.git_sha is None or isinstance(state.git_sha, str)
    assert state.repo_root


def test_python_env_capture_includes_packages():
    env = PythonEnv.capture()
    assert env.python_version
    assert isinstance(env.pip_freeze, list)


def test_hardware_capture_basic():
    h = HardwareSpec.capture()
    assert h.hostname
    assert h.cpu_count_logical >= 1


def test_sha256_file_known(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"hello world")
    h = sha256_file(path)
    # SHA-256 of "hello world"
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_data_inputs_captures_files(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    di = DataInputs.capture([p])
    assert len(di.files) == 1
    assert di.files[0]["sha256"]


def test_data_inputs_recurses_directory(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("b")
    di = DataInputs.capture([tmp_path])
    assert len(di.files) == 2


def test_snapshot_write_and_read(tmp_path: Path):
    snap = ReproducibilitySnapshot.capture(data_paths=[])
    out = tmp_path / "snap.json"
    p = snap.write(out)
    assert p.exists()
    assert "timestamp_unix" in p.read_text()


def test_data_inputs_missing_file(tmp_path: Path):
    p = tmp_path / "nonexistent.txt"
    di = DataInputs.capture([p])
    assert di.files[0]["missing"]


def test_probes_degrade_gracefully_on_hostile_path(monkeypatch):
    # A hostile PATH (e.g. a file shadowing a dir entry) makes subprocess.run
    # raise NotADirectoryError instead of FileNotFoundError; best-effort tool
    # probes (git / pip / nvidia-smi / rocm-smi) must degrade, not crash.
    import benchmark.rigor.reproducibility as repro

    def _raise(*args, **kwargs):
        raise NotADirectoryError(20, "Not a directory", "git")

    monkeypatch.setattr(repro.subprocess, "run", _raise)

    state = CodeState.capture()
    assert state.git_sha is None

    hw = HardwareSpec.capture()
    assert hw.hostname  # probe failure must not prevent the snapshot
