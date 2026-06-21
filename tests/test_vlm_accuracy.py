"""VLM accuracy fixture schema offline tests — no endpoint or GPU required."""
import json
from pathlib import Path

FIXTURES_PATH = Path("benchmark/accuracy/vlm_fixtures/cases.json")


def test_fixture_schema_valid():
    data = json.loads(FIXTURES_PATH.read_text())
    assert "cases" in data
    assert len(data["cases"]) >= 5, "Need at least 5 VLM test cases"


def test_all_cases_have_required_fields():
    data = json.loads(FIXTURES_PATH.read_text())
    required = {"id", "type", "image_url", "prompt", "expected_keywords", "match_mode", "provenance"}
    for case in data["cases"]:
        missing = required - set(case.keys())
        assert not missing, f"Case {case.get('id')} missing fields: {missing}"


def test_local_assets_exist():
    data = json.loads(FIXTURES_PATH.read_text())
    for case in data["cases"]:
        url = case["image_url"]
        if url.startswith("__LOCAL__:"):
            path = Path(url[len("__LOCAL__:"):])
            assert path.exists(), f"Local asset missing: {path}"


def test_synthetic_ratio_within_limit():
    """Synthetic cases cap at 60% — too many synthetics blocks PASS verdict."""
    data = json.loads(FIXTURES_PATH.read_text())
    cases = data["cases"]
    synthetic = [c for c in cases if "synthetic" in c["provenance"]]
    ratio = len(synthetic) / len(cases)
    assert ratio <= 0.6, f"Synthetic VLM cases {ratio:.0%} exceeds 60% cap"
