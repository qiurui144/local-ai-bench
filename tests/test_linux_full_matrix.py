from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from common import ModelConfig


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_linux_full_matrix.py"
    spec = importlib.util.spec_from_file_location("run_linux_full_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_selected_models_filters_by_target_and_requested_names(monkeypatch):
    mod = _load_module()
    models = [
        ModelConfig(name="amd-chat", target="amd-linux-x86"),
        ModelConfig(name="intel-chat", target="intel-linux"),
        ModelConfig(name="local-chat"),
    ]
    monkeypatch.setattr(mod, "load_models", lambda path: list(models))

    selected = mod._selected_models("amd-linux-x86", "amd-chat")

    assert [m.name for m in selected] == ["amd-chat"]


def test_target_env_contains_amd_and_intel_linux_endpoints():
    mod = _load_module()

    assert mod.TARGET_ENV["amd-linux-x86"]["OLLAMA_AMD_LINUX_BASE_URL"] == "http://localhost:11434/v1"
    assert mod.TARGET_ENV["intel-linux"]["OLLAMA_INTEL_LINUX_BASE_URL"] == "http://localhost:11434/v1"
    assert mod.TARGET_ENV["intel-linux"]["OV_INTEL_LINUX_BASE_URL"] == "http://localhost:8080/v1"


def test_run_one_model_clears_model_skip_and_writes_reports(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "REPORTS", tmp_path)
    captured_skips = []
    captured_judges = []

    def fake_run_all(model_cfg, golden, skip, bench_cfg):
        captured_skips.append(model_cfg.benchmarks.get("skip"))
        captured_judges.append((model_cfg.benchmarks.get("scenarios") or {}).get("judge_model"))
        return {
            "model": model_cfg.name,
            "timestamp": "now",
            "benchmarks": {"accuracy": {"verdict": "PASS"}},
        }

    monkeypatch.setattr(mod.rb, "run_all_for_model", fake_run_all)
    monkeypatch.setattr(mod.rb, "aggregate_multi_seed", lambda seed_runs, durations: {"n_seeds": len(seed_runs)})
    monkeypatch.setattr(mod.rb, "render_markdown", lambda result: "# report\n")

    model = ModelConfig(
        name="amd-chat",
        target="amd-linux-x86",
        benchmarks={"skip": ["accuracy", "ttft"]},
    )

    row = mod._run_one_model(model, {}, {}, 2, "unit", [])

    assert captured_skips == [[], []]
    assert captured_judges == ["amd-chat", "amd-chat"]
    assert row["model"] == "amd-chat"
    assert row["benchmarks"]["accuracy"] == "PASS"
    assert sorted(p.name for p in tmp_path.glob("amd-chat_unit_seed*.json")) == [
        "amd-chat_unit_seed0.json",
        "amd-chat_unit_seed1.json",
    ]
    assert list(tmp_path.glob("amd-chat_unit_*.md"))


def test_run_one_model_quality_only_skips_non_quality_and_forces_long_context(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "REPORTS", tmp_path)
    captured = []

    def fake_run_all(model_cfg, golden, skip, bench_cfg):
        captured.append(model_cfg.benchmarks)
        return {
            "model": model_cfg.name,
            "timestamp": "now",
            "benchmarks": {"conditioned": {"verdict": "PASS"}},
        }

    monkeypatch.setattr(mod.rb, "run_all_for_model", fake_run_all)
    monkeypatch.setattr(mod.rb, "render_markdown", lambda result: "# report\n")

    model = ModelConfig(
        name="amd-chat",
        target="amd-linux-x86",
        benchmarks={"skip": ["conditioned", "scenarios", "conversation_drift"]},
    )
    manifest = []

    row = mod._run_one_model(model, {}, {}, 1, "unit", manifest, quality_only=True)

    assert row["benchmarks"]["conditioned"] == "PASS"
    assert captured[0]["skip"] == mod.NON_QUALITY_DIMS
    assert captured[0]["long_context"]["required"] is True
    assert manifest[0]["event"] == "quality_only_policy"


def test_run_one_model_can_allow_local_scenarios_judge(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "REPORTS", tmp_path)
    captured_judges = []

    def fake_run_all(model_cfg, golden, skip, bench_cfg):
        captured_judges.append((model_cfg.benchmarks.get("scenarios") or {}).get("judge_model"))
        return {
            "model": model_cfg.name,
            "timestamp": "now",
            "benchmarks": {"scenarios": {"verdict": "PASS"}},
        }

    monkeypatch.setattr(mod.rb, "run_all_for_model", fake_run_all)
    monkeypatch.setattr(mod.rb, "render_markdown", lambda result: "# report\n")

    model = ModelConfig(name="amd-chat", target="amd-linux-x86")

    mod._run_one_model(
        model,
        {"scenarios": {"judge_model": "llava-7b-amd-linux"}},
        {},
        1,
        "unit",
        [],
        allow_local_scenarios_judge=True,
    )

    assert captured_judges == [None]


def test_repair_model_runtime_records_unready_http_endpoint(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "_http_ok", lambda url, timeout=3.0: False)
    manifest = []
    model = ModelConfig(
        name="intel-api",
        provider="generic",
        port=8088,
        base_url_override="http://localhost:8088/v1",
    )

    assert mod.repair_model_runtime(model, "intel-linux", manifest) is False

    assert manifest == [{
        "event": "repair_failed",
        "model": "intel-api",
        "runtime": "generic",
        "action": "endpoint_not_ready",
        "url": "http://localhost:8088/v1/models",
    }]


def test_needs_intel_linux_ov_llm_only_for_intel_openvino_chat():
    mod = _load_module()
    llm = ModelConfig(
        name="q25",
        provider="openai",
        target="intel-linux",
        base_url_env="OV_INTEL_LINUX_BASE_URL",
        capabilities=("chat",),
    )
    embedding = ModelConfig(
        name="embed",
        provider="openai",
        target="intel-linux",
        base_url_env="OV_INTEL_LINUX_BASE_URL",
        capabilities=("embedding",),
    )

    assert mod._needs_intel_linux_ov_llm(llm, "intel-linux") is True
    assert mod._needs_intel_linux_ov_llm(llm, "amd-linux-x86") is False
    assert mod._needs_intel_linux_ov_llm(embedding, "intel-linux") is False


def test_intel_linux_ov_llm_records_missing_model_dir(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setenv("OV_INTEL_LINUX_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("OV_INTEL_LINUX_MODEL_ROOT", str(tmp_path / "missing"))
    monkeypatch.setattr(mod, "_http_ok", lambda url, timeout=3.0: False)
    monkeypatch.setattr(
        mod,
        "_stop_linux_ov_llm_processes",
        lambda port, name, manifest: manifest.append({"event": "stopped_named", "port": port, "name": name}),
    )
    monkeypatch.setattr(
        mod,
        "_stop_linux_port",
        lambda port, manifest: manifest.append({"event": "stopped", "port": port}),
    )
    manifest = []
    model = ModelConfig(
        name="qwen3-0.6b-openvino-intel-linux",
        provider="openai",
        target="intel-linux",
        base_url_env="OV_INTEL_LINUX_BASE_URL",
        model_id="qwen3-0.6b-int4-ov",
        port=8080,
        capabilities=("chat",),
    )

    assert mod.ensure_intel_linux_ov_llm(model, manifest) is False

    assert manifest[0] == {
        "event": "stopped_named",
        "port": 8080,
        "name": "intel-linux-ov-llm-8080-qwen3-0.6b-openvino-intel-linux",
    }
    assert manifest[1] == {"event": "stopped", "port": 8080}
    assert manifest[2]["action"] == "model_dir_missing"
    assert manifest[2]["model_dir"].endswith("missing/qwen3-0.6b-int4-ov")


def test_intel_linux_ov_llm_starts_service_and_probes(monkeypatch, tmp_path):
    mod = _load_module()
    model_dir = tmp_path / "qwen2.5-1.5b-int4-ov"
    model_dir.mkdir()
    monkeypatch.setenv("OV_INTEL_LINUX_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("OV_INTEL_LINUX_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("OV_INTEL_LINUX_PYTHON", "/opt/ov/bin/python")
    monkeypatch.setenv("OV_INTEL_LINUX_LLM_DEVICE", "GPU")
    monkeypatch.setattr(mod, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    http_checks = []

    def fake_http_ok(url, timeout=3.0):
        http_checks.append(url)
        return len(http_checks) > 1

    captured = {}

    def fake_start_process(cmd, log_prefix, env=None, settle_s=0.0):
        captured["cmd"] = cmd
        captured["log_prefix"] = log_prefix
        captured["settle_s"] = settle_s
        return 6100

    monkeypatch.setattr(mod, "_http_ok", fake_http_ok)
    monkeypatch.setattr(
        mod,
        "_stop_linux_ov_llm_processes",
        lambda port, name, manifest: manifest.append({"event": "stopped_named", "port": port, "name": name}),
    )
    monkeypatch.setattr(
        mod,
        "_stop_linux_port",
        lambda port, manifest: manifest.append({"event": "stopped", "port": port}),
    )
    monkeypatch.setattr(mod, "_start_process", fake_start_process)
    monkeypatch.setattr(mod, "_probe_openai_chat", lambda model: (True, {"content": "OK"}))
    manifest = []
    model = ModelConfig(
        name="qwen2.5-1.5b-openvino-intel-linux",
        provider="openai",
        target="intel-linux",
        base_url_env="OV_INTEL_LINUX_BASE_URL",
        model_id="qwen2.5-1.5b-int4-ov",
        port=8080,
        capabilities=("chat",),
    )

    assert mod.ensure_intel_linux_ov_llm(model, manifest) is True

    assert manifest[0] == {
        "event": "stopped_named",
        "port": 8080,
        "name": "intel-linux-ov-llm-8080-qwen2.5-1.5b-openvino-intel-linux",
    }
    assert manifest[1] == {"event": "stopped", "port": 8080}
    assert manifest[2]["runtime"] == "intel_linux_ov_llm"
    assert manifest[2]["action"] == "start"
    assert manifest[2]["python"] == "/opt/ov/bin/python"
    assert manifest[2]["model_dir"] == str(model_dir)
    assert manifest[2]["supervisor_name"] == "intel-linux-ov-llm-8080-qwen2.5-1.5b-openvino-intel-linux"
    assert manifest[2]["max_concurrent"] == 1
    assert manifest[2]["reload_every"] == 0
    assert manifest[2]["exit_every"] == 0
    assert manifest[2]["supervisor"] is True
    assert manifest[-1]["event"] == "runtime_probe"
    assert captured["settle_s"] == 3.0
    assert captured["cmd"][:3] == ["/opt/ov/bin/python", "-u", str(mod.ROOT / "scripts" / "supervise_process.py")]
    assert "intel-linux-ov-llm-8080-qwen2.5-1.5b-openvino-intel-linux" in captured["cmd"]
    service_cmd = captured["cmd"][captured["cmd"].index("--") + 1:]
    assert service_cmd[:3] == ["/opt/ov/bin/python", "-u", str(mod.ROOT / "scripts" / "serve_ov_intel.py")]
    assert service_cmd[service_cmd.index("--llm") + 1] == str(model_dir)
    assert service_cmd[service_cmd.index("--llm-device") + 1] == "GPU"
    assert service_cmd[service_cmd.index("--port") + 1] == "8080"
    assert service_cmd[service_cmd.index("--llm-max-concurrent") + 1] == "1"
    assert service_cmd[service_cmd.index("--llm-reload-every") + 1] == "0"
    assert service_cmd[service_cmd.index("--llm-exit-every") + 1] == "0"


def test_selection_policy_requires_explicit_model():
    mod = _load_module()
    args = mod.argparse.Namespace(child_model="", models="", allow_batch=False)

    with pytest.raises(SystemExit) as exc:
        mod._validate_selection_policy(args)

    assert "--models must name exactly one model" in str(exc.value)


def test_selection_policy_blocks_batch_by_default():
    mod = _load_module()
    args = mod.argparse.Namespace(child_model="", models="all", allow_batch=False)

    with pytest.raises(SystemExit) as exc:
        mod._validate_selection_policy(args)

    assert "batch model selection is disabled" in str(exc.value)


def test_selection_policy_allows_single_model_by_default():
    mod = _load_module()
    args = mod.argparse.Namespace(child_model="", models="amd-chat", allow_batch=False)

    mod._validate_selection_policy(args)


def test_selection_policy_allows_batch_with_override():
    mod = _load_module()
    args = mod.argparse.Namespace(child_model="", models="amd-chat,intel-chat", allow_batch=True)

    mod._validate_selection_policy(args)


def test_ollama_ps_acceleration_state():
    mod = _load_module()
    gpu_text = "NAME ID SIZE PROCESSOR UNTIL\nqwen2.5:7b abc 6 GB 100% GPU 4 minutes\n"
    cpu_text = "NAME ID SIZE PROCESSOR UNTIL\nqwen2.5:7b abc 6 GB 100% CPU 4 minutes\n"

    assert mod._ollama_ps_acceleration_state(gpu_text, "qwen2.5:7b") == "gpu"
    assert mod._ollama_ps_acceleration_state(cpu_text, "qwen2.5:7b") == "cpu"
    assert mod._ollama_ps_acceleration_state(gpu_text, "llama3.2:1b") == "missing"


def test_check_llm_vlm_acceleration_blocks_ollama_cpu(monkeypatch):
    mod = _load_module()
    model = ModelConfig(
        name="intel-chat",
        provider="ollama",
        model_id="qwen2.5:3b",
        capabilities=("chat",),
    )
    monkeypatch.setattr(mod, "_load_ollama_model_for_accel_check", lambda model_id, manifest: True)
    monkeypatch.setattr(
        mod,
        "_ollama_ps_text",
        lambda: "NAME ID SIZE PROCESSOR UNTIL\nqwen2.5:3b abc 2 GB 100% CPU 4 minutes\n",
    )
    manifest = []

    assert mod._check_llm_vlm_acceleration(
        model,
        "intel-linux",
        manifest,
        allow_cpu_llm_vlm=False,
    ) is False
    assert manifest[-1]["state"] == "cpu"


def test_check_llm_vlm_acceleration_allows_ollama_gpu(monkeypatch):
    mod = _load_module()
    model = ModelConfig(
        name="amd-chat",
        provider="ollama",
        model_id="qwen2.5:7b",
        capabilities=("chat",),
    )
    monkeypatch.setattr(mod, "_load_ollama_model_for_accel_check", lambda model_id, manifest: True)
    monkeypatch.setattr(
        mod,
        "_ollama_ps_text",
        lambda: "NAME ID SIZE PROCESSOR UNTIL\nqwen2.5:7b abc 6 GB 100% GPU 4 minutes\n",
    )
    manifest = []

    assert mod._check_llm_vlm_acceleration(
        model,
        "amd-linux-x86",
        manifest,
        allow_cpu_llm_vlm=False,
    ) is True
    assert manifest[-1]["state"] == "gpu"


def test_check_llm_vlm_acceleration_cpu_override_skips_ollama_probe(monkeypatch):
    mod = _load_module()
    model = ModelConfig(
        name="intel-chat",
        provider="ollama",
        model_id="qwen2.5:3b",
        capabilities=("chat",),
    )

    def fail_probe(model_id, manifest):
        raise AssertionError("probe should not run when CPU baseline is explicit")

    monkeypatch.setattr(mod, "_load_ollama_model_for_accel_check", fail_probe)
    manifest = []

    assert mod._check_llm_vlm_acceleration(
        model,
        "intel-linux",
        manifest,
        allow_cpu_llm_vlm=True,
    ) is True
    assert manifest[-1]["event"] == "cpu_llm_vlm_allowed"


def test_check_llm_vlm_acceleration_blocks_configured_openvino_cpu(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("OV_INTEL_LINUX_LLM_DEVICE", "CPU")
    model = ModelConfig(
        name="qwen3-ov",
        provider="openai",
        base_url_env="OV_INTEL_LINUX_BASE_URL",
        model_id="qwen3-0.6b-int4-ov",
        capabilities=("chat",),
    )
    manifest = []

    assert mod._check_llm_vlm_acceleration(
        model,
        "intel-linux",
        manifest,
        allow_cpu_llm_vlm=False,
    ) is False

    assert manifest[-1]["state"] == "cpu_configured"


def test_spawn_detached_writes_launcher_manifest(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "RUNS", tmp_path)
    captured = {}

    def fake_start_process(cmd, log_prefix, env=None, settle_s=0.0):
        captured["cmd"] = cmd
        captured["log_prefix"] = log_prefix
        captured["env"] = env
        captured["settle_s"] = settle_s
        return 4242

    monkeypatch.setattr(mod, "_start_process", fake_start_process)
    args = mod.argparse.Namespace(
        target="intel-linux",
        models="qwen3-0.6b-openvino-intel-linux",
        seeds=3,
        start_index=2,
        limit=5,
        allow_batch=False,
        allow_cpu_llm_vlm=False,
        allow_local_scenarios_judge=False,
    )

    pid = mod._spawn_detached(args, "unit-tag")

    assert pid == 4242
    assert captured["log_prefix"] == tmp_path / "unit-tag"
    assert captured["settle_s"] == 5.0
    assert captured["cmd"][:2] == [mod.sys.executable, "-u"]
    assert "--detach" not in captured["cmd"]
    assert captured["cmd"][-4:] == ["--start-index", "2", "--limit", "5"]
    manifest = (tmp_path / "unit-tag_launcher.json").read_text(encoding="utf-8")
    assert '"pid": 4242' in manifest
    assert '"tag": "unit-tag"' in manifest


def test_run_one_model_isolated_reads_child_row(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "RUNS", tmp_path)

    def fake_run(cmd, cwd, env, stdout, stderr):
        row_path = Path(cmd[cmd.index("--child-output") + 1])
        row_path.write_text(
            mod.json.dumps({"model": "amd-chat", "benchmarks": {"ttft": "PASS"}}),
            encoding="utf-8",
        )
        stdout.write("child ok\n")
        return mod.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    model = ModelConfig(name="amd-chat", target="amd-linux-x86", provider="ollama")

    row = mod._run_one_model_isolated(model, "amd-linux-x86", 3, "unit", [])

    assert row["model"] == "amd-chat"
    assert row["benchmarks"]["ttft"] == "PASS"
    assert row["child_logs"]["stdout"].endswith("unit_amd-chat.out.log")


def test_run_one_model_isolated_records_process_failure(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "RUNS", tmp_path)

    def fake_run(cmd, cwd, env, stdout, stderr):
        stderr.write("native crash\n")
        return mod.subprocess.CompletedProcess(cmd, 134)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    manifest = []
    model = ModelConfig(name="intel-ocr", target="intel-linux", provider="local_onnx")

    row = mod._run_one_model_isolated(model, "intel-linux", 3, "unit", manifest)

    assert row["error"] == "model_process_failed"
    assert row["returncode"] == 134
    assert "native crash" in row["stderr_tail"]
    assert manifest[0]["event"] == "model_process_failed"
