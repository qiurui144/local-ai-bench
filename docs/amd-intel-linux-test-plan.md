# AMD and Intel Linux Full-Matrix Test Plan

Date: 2026-06-30

This plan starts after the active AMD Windows and Intel Windows full-matrix runs
finish, or after a run is explicitly invalidated and stopped. Linux tests should
not be started while the corresponding Windows machine is still running a model
load that must be preserved for the Windows report.

## Policy

- AMD Linux and Intel Linux may run in parallel because they are separate
  physical targets.
- On each Linux target, run one model at a time. Do not overlap model loads,
  local judges, service probes, or repair runs on the same machine.
- For Intel OpenVINO LLM services, the runner must clean old same-port
  `supervise_process.py` and `serve_ov_intel.py` processes before starting the
  next model service. Do not reuse a stale endpoint just because `/v1/models`
  still answers.
- Normal LLM/VLM tests must use the target accelerator. CPU-only LLM/VLM runs
  are explicit CPU baselines only and require `--allow-cpu-llm-vlm`.
- Target-local scenario tests are L1-only by default. Do not load a same-machine
  L2 judge unless the run is explicitly marked as a diagnostic exception.
- If a model invocation fails, diagnose and repair the runtime, endpoint, model
  directory, or dependency issue. Do not silently skip and call the model
  unavailable.
- Public GitHub content is limited to scripts and designated reports. Raw logs,
  pulled reports, temporary manifests, credentials, and private notes stay under
  ignored paths such as `output/`.

## Parameter-Matrix Contract

Linux retests must follow `docs/k3-parameter-evaluation-plan.zh.md` for result
shape, even when the target is AMD or Intel instead of K3. A model can no longer
be closed only as "1K-4K context PASS", "measured", or "blocked" without the
parameter profiles below. Test items and parameter axes are fixed in
`docs/evaluation-contract.json`; `parameter-matrix.json` must conform to
`docs/parameter-matrix.schema.json` and declare
`contract_id=vlm-llm-nas-evaluation-contract`. Reports must include the
top-level `environment` block so OS build, accelerator provider, memory and
storage state are reproducible across AMD, Intel and K3 targets. JSON artifacts
must use the schema suite declared by `docs/evaluation-contract.json`, including
`run-summary.json`, `model-profile.json` and `scheduler-contract.json`.

For each model and parameter bucket, the report must include:

- `latency_profile`: TTFT, prefill, decode/TPOT, end-to-end p50/p90/p95/p99/max.
- `memory_profile`: peak RSS, minimum available memory, OOM/kill/restart state.
- `quality_profile`: task score, threshold, pass/fail reason, evidence status
  for RAG/VLM.
- `startup_profile`: hot/warm/cold state, startup wait, model I/O and cache
  state.
- `queue_profile`: queue wait, queue depth, priority, deadline/cutover result.
- `resource_profile`: accelerator or CPU hold time, lease conflict, fallback.
- `product_verdict`: one of `sync_default`, `sync_bounded`, `async_default`,
  `async_only`, `offline_only`, `not_recommended`, `blocked`.

Scheduler fields such as `max_context_tokens_sync`,
`max_output_tokens_sync`, ETA weights, hot-set, RAG chunk/top-k/evidence
defaults, VLM sync/async boundary and UI timeout are not fixed by this Linux
plan. They are derived only after the cross-platform parameter matrix is
complete.

## Runner

Use the target-local Linux runner:

```bash
python scripts/run_linux_full_matrix.py \
  --target <amd-linux-x86|intel-linux> \
  --models <single-model-name> \
  --seeds 3 \
  --tag <traceable-run-tag> \
  --detach
```

The runner writes manifests and summaries to:

```text
output/reports/linux-full-matrix/
```

Per-model JSON/Markdown/HTML reports remain in:

```text
output/reports/
```

## Preflight

Run these checks on each Linux target before launching the first model:

```bash
git status --short
python -m py_compile scripts/run_linux_full_matrix.py scripts/serve_ov_intel.py
python -m pytest tests/test_linux_full_matrix.py -q
python -m pytest tests/test_platform_model_coverage.py -q
```

For AMD Linux:

```bash
ollama --version
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama ps
```

The runner will pull missing Ollama model IDs and will block Ollama LLM/VLM if
`ollama ps` reports CPU instead of GPU.

For Intel Linux:

```bash
curl -fsS http://localhost:8080/v1/models
```

The preferred path is an accelerator-backed OpenAI-compatible endpoint such as
OpenVINO/vLLM. If using the bundled OpenVINO service, configure model paths
before launch:

```bash
export OV_INTEL_LINUX_BASE_URL=http://localhost:8080/v1
export OV_INTEL_LINUX_MODEL_ROOT=/path/to/openvino-models
export OV_INTEL_LINUX_LLM_DEVICE=GPU
```

If `OV_INTEL_LINUX_LLM_DEVICE=CPU`, the runner blocks chat/VLM models unless
`--allow-cpu-llm-vlm` is set and the tag clearly says `cpu-baseline`.

## Queue After Windows Completion

### AMD Linux

1. LLM accelerator smoke and primary performance:

```bash
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models qwen2.5-7b-amd-linux --seeds 3 --tag amd-linux-20260701-q25-7b --detach
```

2. VLM accelerator:

```bash
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models llava-7b-amd-linux --seeds 3 --tag amd-linux-20260701-llava7b-vlm --detach
```

3. Small and mid LLM coverage, one at a time:

```bash
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models qwen2.5-3b-amd-linux --seeds 3 --tag amd-linux-20260701-q25-3b --detach
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models qwen3-1.7b-amd-linux --seeds 3 --tag amd-linux-20260701-qwen3-17b --detach
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models qwen3-4b-amd-linux --seeds 3 --tag amd-linux-20260701-qwen3-4b --detach
```

4. Embedding and reranker coverage:

```bash
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models bge-m3-amd-linux --seeds 3 --tag amd-linux-20260701-bge-m3 --detach
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models qwen2.5-3b-reranker-amd-linux --seeds 3 --tag amd-linux-20260701-rerank-q25-3b --detach
```

5. OCR and ASR:

```bash
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models rapidocr-amd-linux --seeds 3 --tag amd-linux-20260701-rapidocr --detach
python scripts/run_linux_full_matrix.py --target amd-linux-x86 --models sensevoice-small-amd-linux --seeds 3 --tag amd-linux-20260701-sensevoice --detach
```

AMD Linux probe models such as DirectML/VitisAI OCR or NPU ASR should be run
only after the primary accelerator path is reported, and the result must be
marked as probe or blocked if the Linux provider is unavailable.

### Intel Linux

1. OpenVINO accelerated LLM path:

```bash
python scripts/run_linux_full_matrix.py --target intel-linux --models qwen3-0.6b-openvino-intel-linux --seeds 3 --tag intel-linux-20260701-qwen3-06b-ov --detach
python scripts/run_linux_full_matrix.py --target intel-linux --models qwen2.5-1.5b-openvino-intel-linux --seeds 3 --tag intel-linux-20260701-q25-15b-ov --detach
```

2. Embedding and reranker:

```bash
python scripts/run_linux_full_matrix.py --target intel-linux --models qwen3-embedding-0.6b-intel-linux --seeds 3 --tag intel-linux-20260701-qwen3-embed --detach
python scripts/run_linux_full_matrix.py --target intel-linux --models qwen2.5-3b-reranker-intel-linux --seeds 3 --tag intel-linux-20260701-rerank-q25-3b --detach
```

3. OCR and ASR:

```bash
python scripts/run_linux_full_matrix.py --target intel-linux --models paddleocr-openvino-intel-linux --seeds 3 --tag intel-linux-20260701-paddleocr-ov --detach
python scripts/run_linux_full_matrix.py --target intel-linux --models whisper-tiny-openvino-intel-linux --seeds 3 --tag intel-linux-20260701-whisper-ov --detach
```

4. Ollama CPU baselines only if explicitly needed:

```bash
python scripts/run_linux_full_matrix.py \
  --target intel-linux \
  --models qwen2.5-7b-intel-linux \
  --seeds 3 \
  --tag intel-linux-20260701-q25-7b-cpu-baseline \
  --allow-cpu-llm-vlm \
  --detach
```

CPU-baseline reports must not be mixed into accelerator performance
recommendations.

## Completion Criteria

Linux coverage is complete when each target has a recorded outcome for:

- LLM
- VLM
- embedding
- reranker
- OCR
- ASR

For each model, the report should state one of:

- measured on accelerator
- measured as explicit CPU baseline
- blocked with root cause and repair notes
- invalidated with reason and replacement run tag

These are runner outcomes only. Final product reporting must also provide a
parameter-level `product_verdict` from the fixed verdict set in the
Parameter-Matrix Contract.

After review, copy only maintained scripts and designated summary reports into
the public GitHub surface.
