"""ScenarioSpec + load_cases 契约:provenance 字段强制,缺失即拒载。"""
import json

import pytest

from benchmark.scenarios.base import ScenarioCase, load_cases  # noqa: F401


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")


def test_load_cases_roundtrip(tmp_path):
    f = tmp_path / "cases.jsonl"
    _write_jsonl(f, [
        {"id": "c1", "provenance": "synthetic", "difficulty": "easy",
         "payload": {"text": "你好"}},
        {"id": "c2", "provenance": "curated", "payload": {"text": "案情"}},
    ])
    cases = load_cases(f)
    assert [c.id for c in cases] == ["c1", "c2"]
    assert cases[0].provenance == "synthetic"
    assert cases[1].difficulty == "normal"          # 缺省值
    assert cases[1].payload["text"] == "案情"


def test_load_cases_rejects_missing_provenance(tmp_path):
    f = tmp_path / "cases.jsonl"
    _write_jsonl(f, [{"id": "c1", "payload": {}}])
    with pytest.raises(ValueError, match="provenance"):
        load_cases(f)


def test_load_cases_rejects_bad_provenance(tmp_path):
    f = tmp_path / "cases.jsonl"
    _write_jsonl(f, [{"id": "c1", "provenance": "real_user_data", "payload": {}}])
    with pytest.raises(ValueError, match="provenance"):
        load_cases(f)


def test_load_cases_missing_file_returns_none(tmp_path):
    assert load_cases(tmp_path / "nope.jsonl") is None


def test_load_cases_num_samples_cap(tmp_path):
    f = tmp_path / "cases.jsonl"
    _write_jsonl(f, [{"id": f"c{i}", "provenance": "synthetic", "payload": {}}
                     for i in range(10)])
    assert len(load_cases(f, num_samples=3)) == 3
