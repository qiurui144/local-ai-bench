"""Tests for conversation_drift dimension runner."""
import json
from unittest.mock import patch

import common
from benchmark.conversation_drift import runner as drift_mod


class _Model:
    name = "test-model"
    hf_repo = "org/model"
    is_vlm = False
    base_url = "http://localhost:9999/v1"
    api_key_env = None

    @property
    def auth_header(self):
        return "Bearer test"

    @property
    def effective_model_id(self):
        return self.name


def _ok(d):
    return common.InferResult(model="m", ok=True, content=json.dumps(d), parsed_json=d)


def _ok_text(t):
    return common.InferResult(model="m", ok=True, content=t, parsed_json=None)


# ---- filler loading ----

def test_load_filler_returns_list(tmp_path):
    f = tmp_path / "filler_turns.jsonl"
    f.write_text('{"q": "Q1", "a": "A1"}\n{"q": "Q2", "a": "A2"}\n')
    with patch.object(drift_mod, "FILLER_PATH", f):
        turns = drift_mod._load_filler()
    assert len(turns) == 2
    assert turns[0]["q"] == "Q1"


def test_load_filler_missing_returns_empty(tmp_path):
    with patch.object(drift_mod, "FILLER_PATH", tmp_path / "missing.jsonl"):
        turns = drift_mod._load_filler()
    assert turns == []


# ---- prior_messages construction ----

def test_build_prior_messages_zero():
    turns = [{"q": "Q1", "a": "A1"}]
    msgs = drift_mod._build_prior_messages(turns, 0)
    assert msgs == []


def test_build_prior_messages_one_turn():
    turns = [{"q": "Q1", "a": "A1"}]
    msgs = drift_mod._build_prior_messages(turns, 1)
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "Q1"}
    assert msgs[1] == {"role": "assistant", "content": "A1"}


def test_build_prior_messages_cycles():
    turns = [{"q": "Q1", "a": "A1"}]
    msgs = drift_mod._build_prior_messages(turns, 3)
    assert len(msgs) == 6  # 3 turns × 2 messages each


# ---- run_conversation_drift with mock ----

def test_run_blocked_when_filler_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_mod, "FILLER_PATH", tmp_path / "missing.jsonl")
    result = drift_mod.run_conversation_drift(_Model(), cfg={})
    assert result["verdict"] == "BLOCKED"
    assert "filler corpus missing" in result["verdict_reasons"][0]


def test_run_stable_when_quality_constant(tmp_path, monkeypatch):
    """Perfect model with constant quality across all positions → STABLE."""
    # Write filler
    filler = tmp_path / "filler.jsonl"
    filler.write_text("\n".join(
        json.dumps({"q": f"Q{i}", "a": f"A{i}"}) for i in range(25)
    ))
    monkeypatch.setattr(drift_mod, "FILLER_PATH", filler)

    # Write cases for one scenario (instruction_following uses text-only, no VLM)
    cases_dir = tmp_path / "datasets" / "scenarios" / "instruction_following"
    cases_dir.mkdir(parents=True)
    case = {"id": "c1", "provenance": "curated", "payload": {
        "prompt": "Say hello.", "instructions": [{"type": "must_include", "value": "hello"}]
    }}
    (cases_dir / "cases.jsonl").write_text(json.dumps(case))
    monkeypatch.setattr(drift_mod, "ROOT", tmp_path)

    def fake_infer(model_cfg, *, prompt, prior_messages=None, **kw):
        return _ok_text("hello world")

    monkeypatch.setattr(drift_mod, "infer_sync", fake_infer)

    result = drift_mod.run_conversation_drift(_Model(), cfg={"num_cases": 1})
    s = result["per_scenario"].get("instruction_following", {})
    assert s.get("verdict") in ("STABLE", "WARN", "BLOCKED")
    # With perfect constant responses, drop should be near 0
    if s.get("verdict") == "STABLE":
        assert s["max_quality_drop"] <= 0.05


def test_prior_messages_passed_to_infer(tmp_path, monkeypatch):
    """Verify that prior_messages grows with position."""
    filler = tmp_path / "filler.jsonl"
    filler.write_text("\n".join(
        json.dumps({"q": f"Q{i}", "a": f"A{i}"}) for i in range(25)
    ))
    monkeypatch.setattr(drift_mod, "FILLER_PATH", filler)

    cases_dir = tmp_path / "datasets" / "scenarios" / "instruction_following"
    cases_dir.mkdir(parents=True)
    case = {"id": "c1", "provenance": "curated", "payload": {
        "prompt": "Say hello.", "instructions": [{"type": "must_include", "value": "hello"}]
    }}
    (cases_dir / "cases.jsonl").write_text(json.dumps(case))
    monkeypatch.setattr(drift_mod, "ROOT", tmp_path)

    message_counts = []

    def fake_infer(model_cfg, *, prompt, prior_messages=None, **kw):
        message_counts.append(len(prior_messages) if prior_messages else 0)
        return _ok_text("hello world")

    monkeypatch.setattr(drift_mod, "infer_sync", fake_infer)

    drift_mod.run_conversation_drift(_Model(), cfg={"num_cases": 1})

    # positions=[0,5,10,20] → message counts [0, 10, 20, 40]
    assert 0 in message_counts
    assert max(message_counts) == 40  # 20 turns × 2 messages
