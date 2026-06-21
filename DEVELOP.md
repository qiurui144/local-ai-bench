# DEVELOP — developer onboarding

How the harness is put together, and how to extend it. User-facing docs live in [README.md](README.md); version history in [RELEASE.md](RELEASE.md).

## Quick dev setup

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # full offline suite (no GPU / no served model needed)
ruff check .                   # lint — repo is ruff-green; keep it that way
```

CI (`.github/workflows/ci.yml`) runs lint, syntax, shellcheck **and the full offline pytest suite** on every push/PR.

## Architecture map

```
models.yaml ──► run_benchmark.py (orchestrator) ──► benchmark/<dim> runners ──► output/reports/
```

| Module | Role |
|---|---|
| `run_benchmark.py` | Orchestrator: the 13-entry `DIMENSIONS` registry (dispatch + capability gating + render hooks), judge resolution (`_resolve_judge`), per-dimension config plumbing, multi-seed aggregation, Markdown rendering, CLI / exit-code policy, `--compare` entry |
| `benchmark/registry.py` | `DimensionSpec` / `RunContext` dataclasses + single-source verdict semantics (`worst_verdict` / `cap_warn`) + quality-leaf collection |
| `benchmark/report/sections.py` | Per-dimension Markdown render hooks (one `render_<dim>` per registry entry) |
| `benchmark/compare.py` | `--compare` replaceability verdict: offline 2σ comparison of saved schema-v1 reports, exit 0/1/2 |
| `common.py` | `ModelConfig` + yaml loaders, 4 HTTP clients (chat sync/stream/async, embed, rerank), shared prompts, health check, VRAM snapshot, percentile stats, RSS |
| `benchmark/performance.py` | ttft / throughput / prefill_decode / concurrency / stability |
| `benchmark/accuracy.py` | Golden-set VLM accuracy (`golden/expectations.json`) |
| `benchmark/translation/` `embedding/` `rerank/` `asr/` `scenarios/` `general_ability/` `conditioned/` | One package per quality dimension: datasets + metrics + accuracy/runner (translation/embedding/rerank/asr also own their `dimension.py` orchestration) |
| `benchmark/rigor/` | Reused statistics library — `multi_seed_runner` is used by the orchestrator; the other 10 modules mostly serve `rag/` and tests |
| `benchmark/rag/` | RAG validation framework (12 chapters + labs/rubrics/schemas) — educational/methodology track, not wired into the verdict chain |
| `benchmark/llama_benchmark/` | Absorbed second harness (own config/registry/backends/datasets/reporters), consumed as a **library**, 0 imports to/from the active harness. Config: `benchmark/llama_configs/`; K1 measured baselines: `benchmark/llama_baselines/` |

### Dimension dispatch flow

For each model, `run_all_for_model` loops the `DIMENSIONS` registry (13 entries; registration order = dispatch order = report-section order): skip if the dimension is in `--skip` or its `gate(model_cfg)` is closed → call `spec.run(model_cfg, dim_cfg, ctx)` → store the returned block under `results["benchmarks"][<dim>]`. Gates come from `ModelConfig.capabilities`, a typed positive set derived by `load_models` from the `models.yaml` `*_capable` hints (hints kept as alias for one minor); `general_ability` / `conditioned` / `scenarios` gate on chat-capable, and scenarios additionally resolves an LLM judge that must differ from the model under test. `QUALITY_DIMS` is derived from the registry (`quality=True` entries), never hand-written.

**Scenarios judge auto-selection (`select_judge_model`):** When no explicit `judge_model` is configured in `models.yaml`, `_resolve_judge` builds a pool of all non-DUT chat-capable models with reachable endpoints and calls `benchmark/scenarios/judge.py::select_judge_model(pool)`. Priority order: `["7b", "14b", "3b", "1.5b", "0.6b"]` — matched by substring against `model.name` (case-insensitive); within a tier, the highest `vram_estimate_gb` wins; falls back to `pool[0]` if no tier pattern matches. A `RuntimeError` (empty pool) returns `None` and disables L2 judging.

**Scenarios and conditioned independence:** `scenarios` does NOT require `conditioned` to have run first. Both gate on `chat_capable`; if `conditioned` is skipped, `scenarios` still dispatches independently.

### Verdict and exit-code semantics

- Per-dimension verdict: `PASS` / `WARN` / `FAIL` / `BLOCKED` (BLOCKED = prerequisites missing, counted as WARN — never a fake PASS). Synthetic-provenance data caps a dimension at WARN.
- Overall verdict = worst-of across dimensions → exit code `0` PASS / `1` WARN / `2` FAIL.
- A named `--model <name>` that errors exits `2`. Under `--model all`, down models are skipped — but if **zero** models produce any measurement the run exits `2`.
- `--seeds N` re-runs the whole suite N times; the exit verdict is the **worst** across seeds (never averaged); `--seeds <1` exits `2`.
- `--compare BASELINE CANDIDATE` is a separate offline path (`benchmark/compare.py`): `REPLACEABLE` exit `0` / `INCONCLUSIVE` exit `1` / `NOT_REPLACEABLE` exit `2`. Single-seed input is hard-capped at INCONCLUSIVE; `schema_version`-less, `harness_version`-mismatched or `condition`-mismatched reports are refused; `hardware_profile` mismatch forces the performance side INCONCLUSIVE.

### Evidence / report layout (`output/reports/`)

- `{model}_{timestamp}.json` — full machine-readable report, **schema v1**: `schema_version / harness_version (git SHA) / hardware_profile (gpu, driver, cuda, vllm, hostname_hash — probes degrade to "unknown") / condition`, plus `model / hf_repo / quantization / timestamp / vram_snapshot / vram_after / benchmarks / [multi_seed]`
- `{model}_{timestamp}.md` — rendered per-model Markdown report
- `{model}_{timestamp}.html` — self-contained HTML report with Chart.js radar/bar/drift charts; compare mode produces `compare_{baseline}_vs_{candidate}_{timestamp}.html`
- `{model}_{timestamp}_seed{k}.json` — per-seed raw archives under `--seeds N` (evidence retention: aggregates are never the only record)
- `matrix_{timestamp}.md` — cross-model comparison matrix
- `compare_{baseline}_vs_{candidate}_{timestamp}.{json,md}` — `--compare` verdict + per-metric Δ/σ evidence

Tooling that consumes reports should pin on `schema_version == 1`; legacy (pre-v0.3.0) reports lack the field and are refused by `--compare`.

## How to add a benchmark dimension

The `DIMENSIONS` registry (landed v0.3.0, per `reports/2026-06-11-architecture-review.md` P0.2) replaced the old 7-site hand-wiring. A new dimension now touches:

1. `benchmark/<dim>/` — the new package (datasets / metrics / runner returning a `{verdict, verdict_reasons, ...}` block; BLOCKED when prerequisites are missing, synthetic data caps at WARN).
2. `run_benchmark.py` — a thin module-level wrapper `_run_<dim>_dim(m, c, ctx)` + one `DimensionSpec` entry in `DIMENSIONS` (name, `quality=` flag, `run=`, optional `gate=` capability check, `render=`) + the `--skip` help string. `QUALITY_DIMS` is derived — do not edit it.
3. `benchmark/report/sections.py` — a `render_<dim>(block) -> list[str]` hook (+ `render_matrix` in `run_benchmark.py` if it should appear in the cross-model matrix).
4. `models.yaml` — a `benchmarks.<dim>` block with thresholds (+ a `<dim>_capable` hint if gated; `load_models` derives the typed `capabilities` set from hints).
5. `tests/<dim>/` + the README dimension table.

`benchmark/general_ability/` (wired in the v0.3.0 sprint) is the most recent worked example.

## How to add a scenario

Scenarios have a registry (`benchmark/scenarios/__init__.py::SCENARIOS`) — much cheaper than a new dimension. The checklist below is self-contained; `scripts/verify_benchmark.py` auto-detects your new scenario via the registry, so no extra plumbing is needed there.

### Checklist (S9 example: `my_new_scenario`)

**1. Dataset** — `datasets/scenarios/my_new_scenario/cases.jsonl`

Each line is one JSON object:
```json
{"id": "c1", "provenance": "curated", "payload": {"field1": "...", "field2": "..."}}
```
- `provenance`: `"synthetic"` caps verdict at WARN; `"curated"` / `"dataset"` unlock PASS.
- Aim for ≥30 cases; ≥50 for robust benchmarking.
- A missing `cases.jsonl` → `BLOCKED` (never a silent empty run).
- Ingest reviewed real-world cases via `scripts/curate_scenario_case.py`; dataset-track derivations: see `scripts/derive_cail_cases.py` / `scripts/derive_cail_dialogs.py` (HF revision-pinned, license-checked).

**2. Spec** — `benchmark/scenarios/my_new_scenario.py`

```python
from .base import ScenarioCase, ScenarioSpec

def _build_prompt(case: ScenarioCase) -> tuple[str, str | None]:
    ...  # returns (prompt_text, image_path_or_None)

def _l1_score(case: ScenarioCase, parsed: dict | None, text: str) -> dict:
    ...  # returns {"metric_name": float, ...}

def _aggregate(scores: list[dict]) -> dict:
    ...  # returns {"metric_name": float} aggregated over all cases

JUDGE_RUBRIC = "Score 1-5: ..."

SPEC = ScenarioSpec(
    name="my_new_scenario",
    cases_path="datasets/scenarios/my_new_scenario/cases.jsonl",
    build_prompt=_build_prompt,
    l1_score=_l1_score,
    aggregate_l1=_aggregate,
    judge_rubric=JUDGE_RUBRIC,
    requires_vlm=False,                         # set True if the model must handle images
    default_thresholds={"metric_name_min": 0.80},
    payload_required_fields=["field1", "field2"],  # drives verify_benchmark.py — no other update needed
)
```

If your scenario shares extraction logic with S5/S7, import from `benchmark/scenarios/_extraction_common.py`.

**3. Register** — `benchmark/scenarios/__init__.py`

```python
from .my_new_scenario import SPEC as _MY_NEW
SCENARIOS = {
    ...,
    _MY_NEW.name: _MY_NEW,
}
```

**4. Image fixtures** (VLM scenarios only)

Put fixtures under `fixtures/scenarios/my_new_scenario/` and run `scripts/check_no_real_images.sh` — real screenshots never enter git (synthetic renders: `scripts/render_wechat_case.py`).

**5. Tests** — `tests/scenarios/test_my_new_scenario.py`

- Test `build_prompt`, `l1_score`, `aggregate_l1`, and SPEC attribute values.
- Add the scenario to `tests/scenarios/test_runner.py`'s `_patch_perfect_run` rows.
- `test_registry.py::test_all_scenarios_have_required_attributes` automatically validates your SPEC — no count update needed.

**6. models.yaml** (optional)

Add threshold overrides under `benchmarks.scenarios.thresholds` and L2 judge anchors in `golden/scenarios.json`.

**7. Verify**

```bash
python3 scripts/verify_benchmark.py   # auto-detects new scenario via SCENARIOS registry
python3 -m pytest tests/ -q           # full offline suite
```

## Testing

See [tests/TESTING.md](tests/TESTING.md) for the test outline (categories, run commands, pass criteria). Quick version: `python -m pytest tests/ -q` runs everything offline; per-area suites live in `tests/{translation,embedding,rerank,asr,scenarios,performance,rag,rigor,llama_benchmark}/`.

## Debugging Guide

### Step 1: Probe the endpoint first

Before running a full benchmark, always verify the endpoint is functional:

```bash
python3 scripts/probe_provider.py --model <name>
```

The probe checks: endpoint reachability · chat completion · JSON mode · seed consistency (local only) · VLM image acceptance. A `READY` result means all applicable checks passed.

### Common errors and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` at `localhost:8000` | Server not running | Start vLLM/Ollama/llama-server; wait for `wait_for_server` |
| `HTTP 401 Unauthorized` | Wrong or missing API key | Check `api_key_env` in models.yaml; `export OPENAI_API_KEY=sk-...` |
| `HTTP 422 Unprocessable Entity` | `model` field mismatch | Set `model_id: <actual-model-name>` in models.yaml to match what the server expects |
| `HTTP 429 Too Many Requests` (persists) | Cloud rate limit exceeded | Reduce `--num-cases`; cloud providers auto-retry 3× with backoff — if 429 still persists, wait and retry |
| `Timeout (600s)` during inference | Model too slow / GPU saturated | Use `--skip stability` for large models; check `nvidia-smi` for OOM or throttling |
| `BLOCKED` verdict on a dimension | Prerequisite missing | Check the dataset path exists; `--skip <dim>` to exclude; see dimension-specific notes below |
| `INCONCLUSIVE` from `--compare` | Single-seed data | Re-run with `--seeds 3`; see "Comparability" in DEVELOP.md |
| All scenario dims show `WARN` | Only synthetic provenance | Add curated cases via `scripts/curate_scenario_case.py` |
| `ImportError` on startup | Missing Python packages | `pip install -r requirements.txt` |
| `CUDA out of memory` | Model too large | Use quantized (GGUF/INT4) model or reduce `max_model_len` in models.yaml |
| JSON report missing fields | Old report + new harness | Reports from different `harness_version` cannot be `--compare`d; re-run both |

### Dimension-specific BLOCKED causes

| Dimension | BLOCKED when | Fix |
|-----------|-------------|-----|
| `general_ability` | HuggingFace datasets unreachable | Set `TRANSLATION_OFFLINE=1`; or `--skip general_ability` on air-gapped hosts |
| `conditioned` | Needle dataset missing | Check `datasets/conditioned/` exists |
| `translation` | Both Flores (network) and offline fallback fail | `TRANSLATION_OFFLINE=1` forces builtin fallback (caps at WARN); or `--skip translation` |
| `asr` | sherpa-onnx / model / audio manifest not found | Check `models.yaml::benchmarks.asr.model_dir` and `manifest_path`; `--skip asr` to bypass |
| `embedding` / `rerank` | Model not `embedding_capable` / `rerank_capable` | Add the hint in models.yaml |
| `scenarios` | `cases.jsonl` missing for any scenario | Run `scripts/verify_benchmark.py` to identify which; `--skip scenarios` to bypass |

### Environment variable reference

```bash
# Authentication (cloud providers)
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=sk-...
export DASHSCOPE_API_KEY=sk-...

# Remote Ollama endpoints (device IP configurable — never hardcode in models.yaml)
export OLLAMA_AMD_BASE_URL=http://<AMD_DEVICE_IP>:11434/v1   # AMD Ryzen / 780M Vulkan GPU
# export OLLAMA_REMOTE_BASE_URL=http://<OTHER_IP>:11434/v1   # any other remote Ollama node

# Force offline mode for datasets
export TRANSLATION_OFFLINE=1        # skip Flores network fetch, use builtin fallback (WARN cap)

# Debug logging
export BENCHMARK_LOG_LEVEL=DEBUG    # verbose HTTP + inference timing logs

# Override Flores mirror (if default haoranxu/FLORES-200 is rate-limited)
export FLORES_DATASET=<hf-repo>
export FLORES_REVISION=<commit-sha>
```

Copy `.env.example` as a starting point: `cp .env.example .env && vi .env && source .env`.

### Reading the HTML report

The auto-generated `output/reports/<model>_<ts>.html` opens in any browser:

- **Quality radar**: 9 axes, values 0–1. A dim showing 0 is SKIPPED or BLOCKED (not a failure — it just wasn't evaluated). `PASS=1.0`, `WARN=0.6`, `FAIL=0.2`.
- **Performance table**: TTFT p50/p95 in ms (lower = better); throughput TPS (higher = better).
- **Scenarios bar chart**: L1 primary metric per S1–S8.
- **Conversation drift line**: quality at 0/5/10/20 prior turns — a flat line is ideal; a steep drop indicates the model can't maintain quality in long sessions.
- **Compare HTML** (`compare_*.html`): two radar overlays. Green badge = REPLACEABLE. Yellow = INCONCLUSIVE. Red = NOT_REPLACEABLE.

### GA quality thresholds by model tier

`general_ability` uses `DEFAULT_THRESHOLDS = {gsm8k_min: 0.55, mmlu_min: 0.55, hellaswag_min: 0.60}` for 3-7B models. Smaller models need lower floors set per-model in `models.yaml`:

| Tier | gsm8k_min | mmlu_min | hellaswag_min | Where to set |
|------|-----------|----------|---------------|-------------|
| ≤0.6B | 0.20 | 0.40 | 0.45 | `models.yaml::benchmarks.general_ability.thresholds` |
| 1.5B | 0.30 | 0.45 | 0.50 | `models.yaml::benchmarks.general_ability.thresholds` |
| 3-7B | 0.55 | 0.55 | 0.60 | DEFAULT_THRESHOLDS (no override needed) |

Per-model thresholds in `models.yaml` merge with and override `DEFAULT_THRESHOLDS`. Only set the keys you want to override.

### VLM accuracy fixtures (`benchmark/accuracy/vlm_fixtures/`)

The VLM accuracy golden set lives at `benchmark/accuracy/vlm_fixtures/cases.json`. Schema per case:
- `id` — unique slug
- `type` — `ocr_in_image` / `basic_visual` / `counting` / `color_recognition` / `scene_description`
- `image_url` — HTTPS URL or `__LOCAL__:relative/path` for repo-bundled assets
- `prompt` — question to ask the VLM
- `expected_keywords` — list; any match counts (keyword_any mode)
- `match_mode` — always `keyword_any` for now
- `provenance` — `wikipedia_commons` or `synthetic_local`; >60% `synthetic_local` caps the dimension at WARN

Local image assets (simple synthetic test images) live in `benchmark/accuracy/vlm_fixtures/assets/`. Add new assets with `scripts/gen_vlm_test_images.py`.

### Reporting a false FAIL

If a dimension fails unexpectedly:
1. Run `python3 scripts/probe_provider.py --model <name>` — confirms the endpoint is correct.
2. Check `output/reports/<model>_<ts>.json` → the dimension's `verdict_reasons` array tells exactly why it failed.
3. Lower the threshold in `models.yaml::benchmarks.<dim>.thresholds` if the baseline floor was wrong for your model class.
4. Run `python3 scripts/verify_benchmark.py` to confirm dataset integrity.

## Provider Testing Guide

This benchmark is designed to test **any OpenAI-compatible API endpoint** — vLLM, Ollama,
Dashscope, OpenAI, Anthropic (via proxy), HuggingFace TGI, etc.

### Configuring a model endpoint

Add your model to `models.yaml` under `models:`:

```yaml
models:
  - name: qwen3-vl-7b-local         # unique label used in reports
    hf_repo: Qwen/Qwen2.5-VL-7B-Instruct  # for reference only
    is_vlm: true                    # set true if the model handles images
    base_url: http://localhost:8000/v1   # OpenAI-compatible base URL
    api_key_env: VLLM_API_KEY       # env var name containing the API key (not the key itself)
    capabilities:
      - chat_capable                # required for scenarios/general_ability/conditioned
      # - embedding_capable         # uncomment if model is an embedder
      # - asr_capable               # uncomment if model does speech recognition
    benchmarks:
      scenarios:
        thresholds: {}              # per-scenario threshold overrides (optional)
```

**Remote / configurable endpoints** — use `base_url_env` to keep device IPs out of committed files:

```yaml
  - name: llama3.2-3b-amd-win
    provider: ollama
    base_url_env: OLLAMA_AMD_BASE_URL       # read IP from env — override in .env or shell
    base_url_env: OLLAMA_AMD_BASE_URL              # set to http://<amd-windows-ip>:11434/v1
    model_id: llama3.2:3b
    port: 11434
```

Priority chain: `base_url_env` env var → `base_url_override` → provider routing default. The `base_url_override` in models.yaml is documentation-only default — never treated as secret.

### Common provider configurations

| Provider | base_url | api_key_env | Notes |
|----------|----------|-------------|-------|
| Local vLLM | `http://localhost:8000/v1` | `VLLM_API_KEY` | Set `VLLM_API_KEY=any` if no auth |
| Ollama | `http://localhost:11434/v1` | — | No key needed; set `api_key_env: OLLAMA_KEY` and `OLLAMA_KEY=ollama` |
| OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` | |
| Dashscope (阿里云) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | |
| DeepSeek | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | |
| HuggingFace TGI | `http://<host>:<port>/v1` | `HF_API_KEY` | |
| Anthropic (proxy) | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` | Requires OpenAI-compat proxy |

### Probing an endpoint before benchmarking

```bash
# Verify any configured model endpoint
python3 scripts/probe_provider.py --model <model-name>

# Example outputs:
# ✓ Endpoint reachable (0.08s)
# ✓ Chat completion works (1.23s, 15 tokens)
# ✓ JSON mode: valid JSON returned
# ✓ Seed consistency: outputs match (seed=42)   ← local only
# ✓ VLM: image accepted                          ← if is_vlm=True
# READY — all checks passed
```

Cloud providers skip the endpoint-reachability check (expected) and the seed-consistency check (cloud models don't guarantee deterministic sampling).

### Remote Targets (multi-platform)

For the full multi-platform deployment SOP covering AMD Windows (Vulkan), Intel Windows,
Rockchip RK3588 (RKNN NPU), SpacemiT K3 (RISC-V), Intel Linux, and Jetson, see
[docs/DEPLOY_TARGETS.md](docs/DEPLOY_TARGETS.md).

Quick start for the AMD Windows / Radeon 780M target (most common):

```bash
# 1. On the Windows AMD machine — start Ollama with Vulkan GPU and LAN access:
set OLLAMA_HOST=0.0.0.0 && set HSA_OVERRIDE_GFX_VERSION=gfx1102 && ollama.exe serve
# Open Windows Firewall → TCP 11434 inbound

# 2. On this Linux host — set the env var (never hardcode the IP):
export OLLAMA_AMD_BASE_URL=http://$AMD_HOST:11434/v1

# 3. Benchmark
python run_benchmark.py --target amd-win-x86 --model llama3.2-3b-amd-win \
  --seeds 3 --skip stability,embedding,rerank,asr
```

**Hardware notes (Ryzen 7 8845H + Radeon 780M)**:
- GPU: RDNA3 iGPU, 17.9 GiB shared memory pool — Ollama uses Vulkan backend (29/29 layers on GPU)
- NPU: AMD XDNA 16-TOPS — NOT accessible via Ollama (Ollama supports Vulkan/CUDA/ROCm only)
- NPU path (future): sherpa-onnx built with VitisAI EP + AMD RyzenAI SDK; documented in `models.yaml` comments

### ASR local ONNX model

```bash
# sensevoice-small uses sherpa-onnx on CPU (no HTTP endpoint, port: 0)
# Model dir configured in models.yaml::benchmarks.asr.model_dir:
#   datasets/asr/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
# Manifest: datasets/asr/manifest.jsonl (audio paths relative to audio_root: datasets/asr)

python run_benchmark.py --model sensevoice-small --seeds 3 \
  --skip ttft,throughput,prefill_decode,concurrency,stability,translation,embedding,rerank,accuracy,general_ability,conditioned,scenarios
```

### llama.cpp server

```bash
# Start the server (Linux or Windows WSL)
llama-server --port 8080 --model /path/to/qwen2.5-7b-q4_k_m.gguf --alias qwen2.5-7b

# models.yaml entry
- name: llama-cpp-qwen2.5-7b
  provider: llama_cpp
  port: 8080
  model_id: qwen2.5-7b     # matches --alias
  task_type: text_only

# Probe
python3 scripts/probe_provider.py --model llama-cpp-qwen2.5-7b
```

### Running benchmarks

```bash
# Single model, skip long-running dims
python run_benchmark.py --model qwen3-vl-7b-local --skip stability,embedding,rerank,asr

# Compare two models
python run_benchmark.py --model qwen2.5-vl-7b-fp16 --seeds 3
python run_benchmark.py --model qwen3-vl-8b-instruct --seeds 3
python run_benchmark.py --compare qwen2.5-vl-7b-fp16 qwen3-vl-8b-instruct

# Quick quality check (scenarios + general_ability only)
python run_benchmark.py --model mymodel --skip ttft,throughput,prefill_decode,concurrency,stability,translation,embedding,rerank,asr

# Test output consistency (repeat each case 3 times)
python run_benchmark.py --model mymodel --consistency-runs 3 --skip stability
```

### Output consistency testing (多次询问不跑偏)

The `--consistency-runs N` flag runs each scenario case N times and reports:
- `consistency_rate`: fraction of cases that give consistent verdicts across all N runs
- `l1_mean_std`: mean standard deviation of the primary metric across runs per case

A high-quality model should achieve `consistency_rate ≥ 0.90` at `N=3`.

### Verification before benchmarking

Always run the dataset integrity check before a benchmark session:

```bash
python3 scripts/verify_benchmark.py
```

This verifies all `cases.jsonl` files are valid, provenances are correct, and VLM image paths exist.

## Documentation structure rules

- `README.md` — entry point + quick start (user-facing, English).
- `RELEASE.md` — version-history SSOT; every shipped change gets a note in its version section, never a separate notes file.
- `DEVELOP.md` (this file) — developer onboarding + architecture.
- `docs/` — deep dives (ACADEMIC-RIGOR / BASELINES / REPRODUCIBILITY / CITATION / CONTRIBUTING-methodology / CROSS-BENCH-MAPPING / case-studies).
- `docs/superpowers/specs/` — design specs (one per feature, dated). Specs for shipped work get archived; implementation plans are deleted after the sprint. One-off sprint reports go to PR descriptions or `RELEASE.md`, not `docs/` top level.
