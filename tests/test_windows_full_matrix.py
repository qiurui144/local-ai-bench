from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from common import ModelConfig


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_windows_full_matrix.py"
    spec = importlib.util.spec_from_file_location("run_windows_full_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_selected_models_filters_by_target_and_requested_names(monkeypatch):
    mod = _load_module()
    models = [
        ModelConfig(name="amd-chat", target="amd-win-x86"),
        ModelConfig(name="intel-chat", target="intel-win-x86"),
        ModelConfig(name="local-chat"),
    ]
    monkeypatch.setattr(mod, "load_models", lambda path: list(models))

    selected = mod._selected_models("amd-win-x86", "amd-chat")

    assert [m.name for m in selected] == ["amd-chat"]


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
        target="amd-win-x86",
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

    model = ModelConfig(name="amd-chat", target="amd-win-x86")

    mod._run_one_model(
        model,
        {"scenarios": {"judge_model": "llava-7b-amd-win"}},
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
        name="intel-ov",
        provider="generic",
        port=8082,
        base_url_override="http://localhost:8082/v1",
    )

    assert mod.repair_model_runtime(model, "intel-win-x86", manifest) is False

    assert manifest == [{
        "event": "repair_failed",
        "model": "intel-ov",
        "runtime": "generic",
        "action": "endpoint_not_ready",
        "url": "http://localhost:8082/v1/models",
    }]


def test_needs_target_specific_extras_only():
    mod = _load_module()
    amd = ModelConfig(name="amd-embed", base_url_env="ORT_AMD_EXTRAS_BASE_URL")
    intel = ModelConfig(name="intel-embed", base_url_env="OV_EXTRAS_INTEL_BASE_URL")

    assert mod._needs_extras(amd, "amd-win-x86") is True
    assert mod._needs_extras(amd, "intel-win-x86") is False
    assert mod._needs_extras(intel, "intel-win-x86") is True
    assert mod._needs_extras(intel, "amd-win-x86") is False


def test_needs_intel_ov_llm_only_for_intel_chat_openai():
    mod = _load_module()
    llm = ModelConfig(
        name="q25",
        provider="openai",
        target="intel-win-x86",
        base_url_env="OV_INTEL_QWEN25_7B_BASE_URL",
        capabilities=("chat",),
    )
    extras = ModelConfig(
        name="embed",
        provider="openai",
        target="intel-win-x86",
        base_url_env="OV_EXTRAS_INTEL_BASE_URL",
        capabilities=("embedding",),
    )

    assert mod._needs_intel_ov_llm(llm, "intel-win-x86") is True
    assert mod._needs_intel_ov_llm(llm, "amd-win-x86") is False
    assert mod._needs_intel_ov_llm(extras, "intel-win-x86") is False


def test_intel_ov_llm_python_env_override(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("OV_INTEL_LLM_PYTHON", r"C:\custom\python.exe")

    assert mod._intel_ov_llm_python() == r"C:\custom\python.exe"


def test_intel_ov_llm_conservative_service_defaults(monkeypatch):
    mod = _load_module()
    monkeypatch.delenv("OV_INTEL_LLM_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("OV_INTEL_LLM_RELOAD_EVERY", raising=False)
    monkeypatch.delenv("OV_INTEL_LLM_EXIT_EVERY", raising=False)

    assert mod._intel_ov_llm_max_concurrent() == 1
    assert mod._intel_ov_llm_reload_every() == 0
    assert mod._intel_ov_llm_exit_every() == 0


def test_intel_ov_llm_env_ints_are_clamped(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("OV_INTEL_LLM_MAX_CONCURRENT", "0")
    monkeypatch.setenv("OV_INTEL_LLM_RELOAD_EVERY", "-7")
    monkeypatch.setenv("OV_INTEL_LLM_EXIT_EVERY", "-9")

    assert mod._intel_ov_llm_max_concurrent() == 1
    assert mod._intel_ov_llm_reload_every() == 0
    assert mod._intel_ov_llm_exit_every() == 0


def test_repair_model_runtime_starts_intel_ov_llm_and_probes(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("OV_INTEL_QWEN25_7B_BASE_URL", "http://localhost:8085/v1")
    monkeypatch.setenv("OV_INTEL_LLM_PYTHON", r"C:\Users\happy\ov-llm-venv\Scripts\python.exe")
    monkeypatch.setenv("OV_INTEL_MODEL_ROOT", r"C:\ov_models")
    monkeypatch.setenv("OV_INTEL_LLM_DEVICE", "GPU")
    monkeypatch.setattr(mod, "RUNS", Path("C:/reports"))
    http_checks = []

    def fake_http_ok(url, timeout=3.0):
        http_checks.append(url)
        return len(http_checks) > 1

    captured = {}

    def fake_start_process(cmd, log_prefix, env=None, settle_s=0.0):
        captured["cmd"] = cmd
        captured["log_prefix"] = log_prefix
        captured["settle_s"] = settle_s
        return 5150

    monkeypatch.setattr(mod, "_http_ok", fake_http_ok)
    monkeypatch.setattr(
        mod,
        "_stop_windows_ov_llm_processes",
        lambda port, name, manifest: manifest.append({"event": "stopped_named", "port": port, "name": name}),
    )
    monkeypatch.setattr(mod, "_stop_windows_port", lambda port, manifest: manifest.append({"event": "stopped", "port": port}))
    monkeypatch.setattr(mod, "_start_process", fake_start_process)
    monkeypatch.setattr(mod, "_probe_openai_chat", lambda model: (True, {"content": "OK"}))
    manifest = []
    model = ModelConfig(
        name="qwen2.5-7b-igpu-intel-win",
        provider="openai",
        target="intel-win-x86",
        base_url_env="OV_INTEL_QWEN25_7B_BASE_URL",
        model_id="qwen2.5-7b-int4-ov",
        port=8085,
        capabilities=("chat",),
    )

    assert mod.repair_model_runtime(model, "intel-win-x86", manifest) is True

    assert manifest[0] == {
        "event": "stopped_named",
        "port": 8085,
        "name": "intel-ov-llm-8085-qwen2.5-7b-igpu-intel-win",
    }
    assert manifest[1] == {"event": "stopped", "port": 8085}
    assert manifest[2]["runtime"] == "intel_ov_llm"
    assert manifest[2]["action"] == "start"
    assert manifest[2]["python"] == r"C:\Users\happy\ov-llm-venv\Scripts\python.exe"
    assert manifest[2]["supervisor_name"] == "intel-ov-llm-8085-qwen2.5-7b-igpu-intel-win"
    assert manifest[2]["max_concurrent"] == 1
    assert manifest[2]["reload_every"] == 0
    assert manifest[2]["exit_every"] == 0
    assert manifest[2]["supervisor"] is True
    assert manifest[-1]["event"] == "runtime_probe"
    assert captured["settle_s"] == 3.0
    assert captured["cmd"][:3] == [
        r"C:\Users\happy\ov-llm-venv\Scripts\python.exe",
        "-u",
        str(mod.ROOT / "scripts" / "supervise_process.py"),
    ]
    assert "intel-ov-llm-8085-qwen2.5-7b-igpu-intel-win" in captured["cmd"]
    service_cmd = captured["cmd"][captured["cmd"].index("--") + 1:]
    assert service_cmd[:3] == [
        r"C:\Users\happy\ov-llm-venv\Scripts\python.exe",
        "-u",
        str(mod.ROOT / "scripts" / "serve_ov_intel.py"),
    ]
    assert service_cmd[service_cmd.index("--llm") + 1] == str(Path(r"C:\ov_models") / "qwen2.5-7b-int4-ov")
    assert service_cmd[service_cmd.index("--llm-device") + 1] == "GPU"
    assert service_cmd[service_cmd.index("--port") + 1] == "8085"
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
        "intel-win-x86",
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
        "amd-win-x86",
        manifest,
        allow_cpu_llm_vlm=False,
    ) is True
    assert manifest[-1]["state"] == "gpu"


def test_check_llm_vlm_acceleration_cpu_override_skips_probe(monkeypatch):
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
        "intel-win-x86",
        manifest,
        allow_cpu_llm_vlm=True,
    ) is True
    assert manifest[-1]["event"] == "cpu_llm_vlm_allowed"


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
        target="intel-win-x86",
        models="llama3.2-1b-intel-win",
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
    model = ModelConfig(name="amd-chat", target="amd-win-x86", provider="ollama")

    row = mod._run_one_model_isolated(model, "amd-win-x86", 3, "unit", [])

    assert row["model"] == "amd-chat"
    assert row["benchmarks"]["ttft"] == "PASS"
    assert row["child_logs"]["stdout"].endswith("unit_amd-chat.out.log")


def test_run_one_model_isolated_records_process_failure(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "RUNS", tmp_path)

    def fake_run(cmd, cwd, env, stdout, stderr):
        stderr.write("native crash\n")
        return mod.subprocess.CompletedProcess(cmd, 3221225477)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    manifest = []
    model = ModelConfig(name="intel-ocr", target="intel-win-x86", provider="local_onnx")

    row = mod._run_one_model_isolated(model, "intel-win-x86", 3, "unit", manifest)

    assert row["error"] == "model_process_failed"
    assert row["returncode"] == 3221225477
    assert "native crash" in row["stderr_tail"]
    assert manifest[0]["event"] == "model_process_failed"


def test_windows_detach_cmd_lines_keep_python_command_on_one_line(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "RUNS", tmp_path)
    args = mod.argparse.Namespace(
        target="amd-win-x86",
        models="all",
        seeds=3,
        start_index=6,
        limit=0,
        allow_batch=True,
        allow_cpu_llm_vlm=False,
        allow_local_scenarios_judge=False,
    )

    cmd_path, lines = mod._windows_detach_cmd_lines(args, "unit-tag")

    assert cmd_path == tmp_path / "unit-tag.cmd"
    assert len(lines) == 5
    assert all("\n" not in line and "\r" not in line for line in lines)
    assert "run_windows_full_matrix.py" in lines[-1]
    assert "--start-index 6" in lines[-1]
    assert "--allow-batch" in lines[-1]
    assert "unit-tag.cmd.out.log" in lines[-1]
    assert "unit-tag.cmd.err.log" in lines[-1]


def test_spawn_detached_windows_task_registers_task(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setattr(mod, "RUNS", tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return mod.subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    args = mod.argparse.Namespace(
        target="intel-win-x86",
        models="all",
        seeds=3,
        start_index=6,
        limit=0,
        allow_batch=True,
        allow_cpu_llm_vlm=False,
        allow_local_scenarios_judge=False,
    )

    assert mod._spawn_detached_windows_task(args, "unit-tag") == 0

    assert (tmp_path / "unit-tag.cmd").exists()
    assert any(cmd[:2] == ["schtasks", "/Create"] for cmd in calls)
    assert any(cmd[:2] == ["schtasks", "/Run"] for cmd in calls)
    manifest = (tmp_path / "unit-tag_launcher.json").read_text(encoding="utf-8")
    assert '"event": "scheduled_task_start"' in manifest
    assert '"task_name": "vlm-unit-tag"' in manifest
