import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> str:
    proc = subprocess.run(
        ["bash", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_spacemit_manifest_lists_data_sources_without_download():
    out = run_script("scripts/cache_spacemit_model_zoo.sh", "--manifest")
    assert "scope\tcategory\trelative_path\turl" in out
    assert "llm/Qwen3-30B-A3B-Q4_0.gguf" in out
    assert "vlm/Qwen3.5-2B.tar.gz" in out
    assert "vision/ppocr/PP-OCRv5_mobile_rec.onnx" in out
    assert "vlm/qwen3-asr-0.6B.tar.gz" in out
    assert "embed/Bge-Small-Zh-V1.5-Q4_K_M.gguf" in out
    assert "rerank/Bge-Reranker-V2-M3-Q4_0.gguf" in out


def test_spacemit_cache_index_is_written_as_real_tsv():
    script = (ROOT / "scripts/cache_spacemit_model_zoo.sh").read_text(encoding="utf-8")
    assert "stat -c '%s\\t%n'" not in script
    assert "printf '%s\\t%s\\n'" in script


def test_k3_cached_runner_describes_invocation_without_target():
    out = run_script("scripts/run_k3_32g_model_zoo_cached.sh", "--describe")
    assert "ai-sdk.md" in out
    assert "modelzoo.md" in out
    assert "llm_chat" in out
    assert "llama-bench -p 128" in out
    assert "RUN_OFFICIAL_MODELZOO_BENCH=1" in out
    assert "MODE=llm" in out
    assert "MODE=vlm-tar" in out
    assert "MODE=vlm-pair" in out
    assert "/v1/chat/completions" in out
    assert "192.168." not in out
    assert "root@" not in out


def test_k3_nonllm_runner_describes_invocation_without_target():
    out = run_script("scripts/run_k3_32g_model_zoo_nonllm_cached.sh", "--describe")
    assert "ai-sdk.md" in out
    assert "modelzoo.md" in out
    assert "onnxruntime_perf_test" in out
    assert "MODE=embedding" in out
    assert "MODE=rerank" in out
    assert "/v1/embeddings" in out
    assert "/v1/rerank" in out
    assert "192.168." not in out
    assert "root@" not in out


def test_k3_local_scripts_do_not_hardcode_target_defaults():
    checked = [
        ROOT / "scripts/k3_32g_common.sh",
        ROOT / "scripts/run_k3_32g_model_zoo_cached.sh",
        ROOT / "scripts/run_k3_32g_model_zoo_nonllm_cached.sh",
        ROOT / "scripts/run_k3_32g_model_zoo_vlm_full.sh",
        ROOT / "scripts/run_k3_32g_model_zoo_remaining_cached.sh",
        ROOT / "scripts/run_k3_32g_official_modelzoo_vision.sh",
        ROOT / "scripts/run_k3_32g_long_context_20b.sh",
        ROOT / "scripts/run_k3_32g_realistic_stress.py",
        ROOT / "scripts/run_k3_32g_nonllm_broad.py",
        ROOT / "scripts/run_k3_32g_official_modelzoo_llm_retest.sh",
        ROOT / "scripts/run_k3_32g_official_modelzoo_vlm_encoder_probe.sh",
        ROOT / "scripts/build_spacemit_a100_sources.sh",
        ROOT / "scripts/run_k3_32g_source_runtime_compare.sh",
        ROOT / "docs/k3-source-runtime-and-long-context.md",
        ROOT / "docs/spacemit-model-zoo.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked)
    assert "192.168.100.233" not in combined
    assert "192.168.100.215" not in combined
    assert "K3_HOST=\"${K3_HOST:-192" not in combined
    assert "K3_USER=\"${K3_USER:-root}\"" not in combined
    assert "pass: bianbu" not in combined


def test_k3_source_runtime_compare_describes_without_target():
    out = run_script("scripts/run_k3_32g_source_runtime_compare.sh", "--describe")
    assert "llama-bench -m <gguf> -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128" in out
    assert "onnxruntime_perf_test <onnx> -e spacemit" in out
    assert "ORT_MODEL_MANIFEST" in out
    assert "192.168." not in out
    assert "root@" not in out


def test_python_k3_run_configs_redact_connection_fields():
    args = argparse.Namespace(
        k3_host="10.0.0.10",
        k3_user="root",
        k3_pass="secret",
        request_timeout=1,
    )
    for rel in (
        "scripts/run_k3_32g_realistic_stress.py",
        "scripts/run_k3_32g_nonllm_broad.py",
    ):
        module = load_module(ROOT / rel)
        data = module.redacted_run_config(args, "/root/run")
        assert data["k3_host"] == "<redacted>"
        assert data["k3_user"] == "<redacted>"
        assert data["k3_pass"] == "<redacted>"
        assert data["request_timeout"] == 1
        assert data["remote_run_dir"] == "/root/run"


def test_spacemit_docs_pin_official_invocation_and_performance_sources():
    doc = (ROOT / "docs/spacemit-model-zoo.md").read_text(encoding="utf-8")
    report = (ROOT / "reports/k3-riscv-32g.en.md").read_text(encoding="utf-8")
    for text in (doc, report):
        assert "https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/application_tools/ai-sdk.md" in text
        assert "https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/compute_stack/ai_compute_stack/modelzoo.md" in text
    assert "doc tree update time `2026-06-30 20:04:26`" in doc
    assert "doc tree update time `2026-06-09 18:06:37`" in doc
    assert "llm_chat" in doc
    assert "POST /v1/vlm/chat/completions" in doc
    assert "llama-bench -p 128 -n 128 -mmp 0 -fa 1 -ub 128" in doc
    assert "onnxruntime_perf_test" in doc
    assert "RETEST_REQUIRED" in doc
    assert "Neighboring quantizations" in doc
    assert "Official Baseline Alignment Gate" in report
    assert "scripts/run_k3_32g_official_modelzoo_llm_retest.sh" in report
    assert "scripts/run_k3_32g_official_modelzoo_vlm_encoder_probe.sh" in doc
    assert "Official VLM VisionEncoder probe" in report
    assert "Embedding, reranker, and PP-OCRv5 OCR conclusions below are local measurements" in report


def test_official_modelzoo_llm_retest_describes_exact_benchmark_without_target():
    out = run_script("scripts/run_k3_32g_official_modelzoo_llm_retest.sh", "--describe")
    assert "llama-bench -m <model> -t <cores> -p 128 -n 128 -mmp 0 -fa 1 -ub 128" in out
    assert "RUN_OFFICIAL_MODELZOO_BENCH=1" in out
    assert "Do not compare neighboring quantizations" in out
    assert "192.168." not in out
    assert "root@" not in out


def test_official_modelzoo_vlm_encoder_probe_describes_without_target():
    out = run_script("scripts/run_k3_32g_official_modelzoo_vlm_encoder_probe.sh", "--describe")
    assert "onnxruntime_perf_test" in out
    assert "VisionEncoder" in out
    assert "SPACEMIT_EP_INTRA_THREAD_NUM" in out
    assert "192.168." not in out
    assert "root@" not in out
