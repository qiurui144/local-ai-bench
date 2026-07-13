from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "generate_quality_diagnosis_report.py"
    spec = importlib.util.spec_from_file_location("generate_quality_diagnosis_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_quality_diagnosis_keeps_production_gate_and_marks_small_llm_limited(tmp_path):
    mod = _load_module()
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  - name: qwen2.5-1.5b-openvino-intel-linux
    target: intel-linux
    provider: openai
    task_type: text_only
    role: llm_test
    notes: unit
""",
        encoding="utf-8",
    )
    raw = tmp_path / "qwen2.5-1.5b-openvino-intel-linux.json"
    raw.write_text(
        json.dumps({
            "model": "qwen2.5-1.5b-openvino-intel-linux",
            "benchmarks": {
                "translation": {
                    "verdict": "FAIL",
                    "verdict_reasons": ["[zh->en] FAIL: term-match 55% < 80%"],
                    "directions": {
                        "zh->en": {
                            "l1_flores": {
                                "verdict": "PASS",
                                "aggregate": {
                                    "level": "l1",
                                    "num_pairs": 5,
                                    "bleu": 45.0,
                                    "chrf": 73.0,
                                    "data_source": "builtin",
                                },
                            },
                            "l3_terminology": {
                                "verdict": "FAIL",
                                "verdict_reasons": ["FAIL: term-match 55% < 80%"],
                                "aggregate": {
                                    "level": "l3",
                                    "num_pairs": 10,
                                    "bleu": 28.0,
                                    "chrf": 55.0,
                                    "data_source": "custom",
                                    "terminology": {
                                        "matched_terms": 6,
                                        "total_terms": 11,
                                        "term_match_rate": 0.55,
                                    },
                                },
                            },
                        }
                    },
                }
            },
        }),
        encoding="utf-8",
    )
    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()
    (contract_dir / "parameter-matrix.json").write_text(
        json.dumps({
            "target": "intel-linux",
            "rows": [{
                "test_item_id": "llm_chat_boundary",
                "task_class": "llm_chat",
                "model_profile": {"name": "qwen2.5-1.5b-openvino-intel-linux"},
                "product_verdict": "not_recommended",
                "product_verdict_reason": "translation_l3_terminology_failed",
                "quality_profile": {"metric_name": "composite_quality", "score": None, "reason": "translation_l3_terminology_failed"},
                "latency_profile": {"e2e_latency_ms": {"p95": 1000}},
                "runtime": {"resource_class": "igpu"},
            }],
        }),
        encoding="utf-8",
    )

    report = mod.build_report(
        raw_reports=[raw],
        contract_dir=contract_dir,
        models_yaml=models_yaml,
        target="intel-linux",
        run_id="unit",
    )

    assert report["standards_decision"]["production_quality_gate"] == "unchanged"
    assert report["quality_status"] == "caveats"
    diag = report["model_diagnostics"][0]
    assert diag["quality_status"] == "limited_suitability"
    assert diag["adjusted_quality_standard"] == "small_llm_basic_translation_only"
    assert diag["production_recommendation"] == "not_recommended_for_terminology_rag_answer"

    paths = mod.write_artifacts(report, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["tsv"].exists()
    assert paths["markdown"].exists()


def test_quality_diagnosis_marks_general_llm_candidate_with_minor_translation_caveat(tmp_path):
    mod = _load_module()
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  - name: qwen2.5-7b-amd-win
    target: amd-win-x86
    provider: ollama
    task_type: text_only
    role: llm_amd_primary
""",
        encoding="utf-8",
    )
    raw = tmp_path / "qwen2.5-7b-amd-win.json"
    raw.write_text(
        json.dumps({
            "model": "qwen2.5-7b-amd-win",
            "benchmarks": {
                "general_ability": {
                    "verdict": "PASS",
                    "tasks": {
                        "gsm8k": {"accuracy": 0.75, "n": 8, "errors": 0, "verdict": "PASS"},
                        "mmlu": {"accuracy": 0.875, "n": 8, "errors": 0, "verdict": "PASS"},
                    },
                },
                "translation": {
                    "verdict": "FAIL",
                    "verdict_reasons": ["[en->zh] FAIL: chrF 34.2 < 35.0"],
                    "directions": {
                        "en->zh": {
                            "l1_flores": {
                                "verdict": "FAIL",
                                "verdict_reasons": ["FAIL: chrF 34.2 < 35.0"],
                                "aggregate": {
                                    "level": "l1",
                                    "num_pairs": 10,
                                    "bleu": 38.2,
                                    "chrf": 34.2,
                                    "data_source": "flores",
                                },
                            },
                            "l3_terminology": {
                                "verdict": "PASS",
                                "aggregate": {
                                    "level": "l3",
                                    "num_pairs": 10,
                                    "bleu": 35.8,
                                    "chrf": 41.6,
                                    "data_source": "custom",
                                    "terminology": {
                                        "matched_terms": 11,
                                        "total_terms": 13,
                                        "term_match_rate": 0.846,
                                    },
                                },
                            },
                        }
                    },
                },
            },
        }),
        encoding="utf-8",
    )

    report = mod.build_report(
        raw_reports=[raw],
        contract_dir=None,
        models_yaml=models_yaml,
        target="amd-win-x86",
        run_id="unit",
    )

    diag = report["model_diagnostics"][0]
    assert diag["quality_status"] == "candidate_with_quality_caveat"
    assert diag["adjusted_quality_standard"] == "general_llm_candidate_translation_l1_caveat"
    assert diag["production_recommendation"] == "candidate_for_general_llm_not_translation_certified"


def test_quality_diagnosis_marks_large_llm_candidate_with_near_miss_terminology(tmp_path):
    mod = _load_module()
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  - name: qwen2.5-7b-intel-linux
    target: intel-linux
    provider: ollama
    task_type: text_only
    role: llm_intel_linux_primary
""",
        encoding="utf-8",
    )
    raw = tmp_path / "qwen2.5-7b-intel-linux.json"
    raw.write_text(
        json.dumps({
            "model": "qwen2.5-7b-intel-linux",
            "benchmarks": {
                "general_ability": {
                    "verdict": "BLOCKED",
                    "verdict_reasons": ["synthetic fallback rejected"],
                    "tasks": {},
                },
                "translation": {
                    "verdict": "FAIL",
                    "verdict_reasons": ["[en->zh] FAIL: term-match 77% < 80%"],
                    "directions": {
                        "en->zh": {
                            "l1_flores": {
                                "verdict": "PASS",
                                "aggregate": {
                                    "level": "l1",
                                    "num_pairs": 5,
                                    "bleu": 67.0,
                                    "chrf": 61.0,
                                    "data_source": "builtin",
                                },
                            },
                            "l3_terminology": {
                                "verdict": "FAIL",
                                "verdict_reasons": ["FAIL: term-match 77% < 80%"],
                                "aggregate": {
                                    "level": "l3",
                                    "num_pairs": 10,
                                    "bleu": 35.4,
                                    "chrf": 42.1,
                                    "data_source": "custom",
                                    "terminology": {
                                        "matched_terms": 10,
                                        "total_terms": 13,
                                        "term_match_rate": 0.77,
                                    },
                                },
                            },
                        }
                    },
                },
            },
        }),
        encoding="utf-8",
    )

    report = mod.build_report(
        raw_reports=[raw],
        contract_dir=None,
        models_yaml=models_yaml,
        target="intel-linux",
        run_id="unit",
    )

    diag = report["model_diagnostics"][0]
    assert diag["quality_status"] == "candidate_with_quality_caveat"
    assert diag["adjusted_quality_standard"] == "general_llm_candidate_terminology_caveat"
    assert diag["production_recommendation"] == "candidate_for_general_llm_with_terminology_caveat"


def test_quality_diagnosis_keeps_large_llm_caveat_when_general_pass_but_translation_and_terms_fail(tmp_path):
    mod = _load_module()
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  - name: qwen2.5-7b-intel-linux
    target: intel-linux
    provider: ollama
    task_type: text_only
    role: llm_intel_linux_primary
""",
        encoding="utf-8",
    )
    raw = tmp_path / "qwen2.5-7b-intel-linux.json"
    raw.write_text(
        json.dumps({
            "model": "qwen2.5-7b-intel-linux",
            "benchmarks": {
                "general_ability": {
                    "verdict": "PASS",
                    "tasks": {
                        "gsm8k": {"accuracy": 0.75, "n": 8, "errors": 0, "verdict": "PASS"},
                        "mmlu": {"accuracy": 0.625, "n": 8, "errors": 0, "verdict": "PASS"},
                        "hellaswag": {"accuracy": 0.875, "n": 8, "errors": 0, "verdict": "PASS"},
                    },
                },
                "translation": {
                    "verdict": "FAIL",
                    "verdict_reasons": [
                        "[en->zh] FAIL: chrF 33.5 < 35.0",
                        "[en->zh] FAIL: term-match 77% < 80%",
                    ],
                    "directions": {
                        "en->zh": {
                            "l1_flores": {
                                "verdict": "FAIL",
                                "verdict_reasons": ["FAIL: chrF 33.5 < 35.0"],
                                "aggregate": {
                                    "level": "l1",
                                    "num_pairs": 10,
                                    "bleu": 37.5,
                                    "chrf": 33.5,
                                    "data_source": "flores",
                                },
                            },
                            "l3_terminology": {
                                "verdict": "FAIL",
                                "verdict_reasons": ["FAIL: term-match 77% < 80%"],
                                "aggregate": {
                                    "level": "l3",
                                    "num_pairs": 10,
                                    "bleu": 35.4,
                                    "chrf": 42.1,
                                    "data_source": "custom",
                                    "terminology": {
                                        "matched_terms": 10,
                                        "total_terms": 13,
                                        "term_match_rate": 0.769,
                                    },
                                },
                            },
                        }
                    },
                },
            },
        }),
        encoding="utf-8",
    )

    report = mod.build_report(
        raw_reports=[raw],
        contract_dir=None,
        models_yaml=models_yaml,
        target="intel-linux",
        run_id="unit",
    )

    assert report["quality_status"] == "caveats"
    diag = report["model_diagnostics"][0]
    assert diag["quality_status"] == "candidate_with_quality_caveat"
    assert diag["adjusted_quality_standard"] == "general_llm_candidate_translation_and_terminology_caveat"
    assert diag["production_recommendation"] == "candidate_for_general_llm_with_translation_and_terminology_caveat"
