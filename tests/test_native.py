import pytest
from benchmark.native.llama_bench import run_llama_bench
from benchmark.native.wrk_runner import _parse_wrk_output

def test_llama_bench_no_binary():
    result = run_llama_bench("/nonexistent.gguf", llama_bench_bin="/nonexistent/llama-bench")
    assert "error" in result
    assert "not found" in result["error"].lower() or "No such" in result["error"]

def test_parse_wrk_output_valid():
    sample = """
  2 threads and 1 connections
  Thread Stats   Avg      Stdev
    Latency   142.30ms   23.14ms
  Requests/sec:     6.89
    """
    result = _parse_wrk_output(sample)
    assert result.get("req_per_s") == pytest.approx(6.89, rel=0.01)
    assert result.get("latency_avg_ms") == pytest.approx(142.30, rel=0.01)

def test_parse_wrk_output_invalid():
    result = _parse_wrk_output("garbage output")
    assert "error" in result
