"""DimensionSpec registry mechanics + shared verdict lib (arch review P0.2/P0.4)."""
from pathlib import Path

from benchmark.registry import DimensionSpec, RunContext, cap_warn, worst_verdict


def test_worst_verdict_ranking():
    assert worst_verdict(["PASS", "SKIPPED"]) == "PASS"
    assert worst_verdict(["PASS", "BLOCKED"]) == "WARN"
    assert worst_verdict(["WARN", "FAIL", "PASS"]) == "FAIL"
    assert worst_verdict([]) == "PASS"


def test_cap_warn_only_demotes_pass():
    assert cap_warn("PASS") == "WARN"
    assert cap_warn("FAIL") == "FAIL"
    assert cap_warn("WARN") == "WARN"


def test_dimension_spec_defaults_gate_open():
    spec = DimensionSpec(name="d", quality=True, run=lambda m, c, ctx: {"verdict": "PASS"})
    assert spec.gate(object()) is True
    assert spec.render is None


def test_run_context_fields():
    ctx = RunContext(root=Path("/r"), fixtures=Path("/r/fixtures"), golden={}, bench_cfg={})
    assert ctx.root == Path("/r")


def test_run_benchmark_dimensions_table_covers_all_11():
    # Task 5 (D1): +general_ability → 12 维;Task 7 (D4): +conditioned → 13。
    import run_benchmark as rb
    expected = {"accuracy", "ttft", "throughput", "prefill_decode", "concurrency",
                "stability", "translation", "embedding", "rerank", "asr",
                "general_ability", "conditioned", "scenarios"}
    assert expected <= set(rb.DIMENSIONS)
    assert rb.QUALITY_DIMS == tuple(
        n for n, s in rb.DIMENSIONS.items() if s.quality)


def test_every_dimension_has_render_hook():
    import run_benchmark as rb
    assert all(s.render is not None for s in rb.DIMENSIONS.values())
