> [中文文档](./README.zh.md)

# local-ai-bench

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/qiurui144/local-ai-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/qiurui144/local-ai-bench/actions/workflows/ci.yml)

**local-ai-bench** is a model selection benchmark platform for local AI deployments — edge devices, Windows laptops, and embedded AI-box hardware. It answers one question:

> *"Can model X replace model Y in production? Is performance sufficient and quality acceptable?"*

Run benchmarks on your target hardware, then compare:

```bash
python run_benchmark.py --model qwen2.5-7b   --seeds 3
python run_benchmark.py --model qwen3-4b     --seeds 3
python run_benchmark.py --compare qwen2.5-7b qwen3-4b
# → REPLACEABLE / NOT_REPLACEABLE / INCONCLUSIVE
```

The verdict is automatic and evidence-backed: every quality metric is compared within 2σ tolerance, single-seed results are capped at INCONCLUSIVE, and performance/quality axes are evaluated separately.

---

## What it covers

**13 benchmark dimensions** across two axes:

| Axis | Dimensions |
|---|---|
| **Performance** | TTFT · throughput · prefill/decode split · concurrency · stability |
| **Model quality** | accuracy · translation (zh↔en) · embedding · rerank · ASR · general ability (GSM8K / MMLU / HellaSwag) · conditioned capability curves · real-product scenarios |

Each dimension has a PASS / WARN / FAIL verdict with configurable thresholds. The final exit code (`0/1/2`) is CI-ready.

→ Full dimension reference: [DEVELOP.md § Dimensions](DEVELOP.md)

---

## Supported platforms

| Platform | Status | Results |
|---|---|---|
| AMD Linux (Ryzen 8845H + 780M iGPU) | ✅ Calibrated | [amd-linux.en.md](reports/amd-linux.en.md) |
| AMD Windows (same Ryzen 8845H hardware; historical dual-boot run) | ✅ Calibrated | [amd-windows.en.md](reports/amd-windows.en.md) |
| Intel Windows (Core Ultra 7 155H + Arc iGPU) | ✅ Calibrated | [intel-windows.en.md](reports/intel-windows.en.md) |
| Intel Linux (OpenVINO/vLLM; CPU baseline explicit) | 🔧 In progress | — |
| RK3588 Linux (RKNN3 NPU) | 🔧 In progress | — |
| K3 RISC-V | 🔧 In progress | — |
| vLLM server (Linux + NVIDIA GPU) | ✅ Supported | — |

Any **OpenAI-compatible endpoint** works — vLLM, Ollama, llama.cpp server, OpenAI, DashScope, DeepSeek.

Project run policy:

- Run one model at a time on the same target machine. Different physical targets can run in parallel.
- LLM/VLM CPU-only runs are special-case CPU baselines only; normal LLM/VLM benchmark runs must use the target accelerator.
- Scenario L2 judges must run on separate hardware or an external service. Target-local single-model runs use L1-only scenarios by default.
- Windows and Linux target-local full-matrix runners enforce these rules: `scripts/run_windows_full_matrix.py` and `scripts/run_linux_full_matrix.py`.

---

## Quick start

**Prerequisites:** Python 3.10+, an OpenAI-compatible model endpoint.

```bash
git clone https://github.com/qiurui144/local-ai-bench.git
cd local-ai-bench
pip install -r requirements.txt

# Verify your endpoint is reachable
python3 scripts/probe_provider.py --model <your-model-name>

# Run benchmark (skip dims you haven't set up)
python run_benchmark.py --model <your-model-name> --skip stability,embedding,rerank,asr

# Offline unit tests (no GPU required)
python -m pytest tests/ -q
```

→ Model registration, provider setup, and Windows/Ollama quickstart: [DEVELOP.md § Setup](DEVELOP.md)

---

## Model configuration

Models are declared in `models.yaml`. The minimum required fields are `name`, `provider`, and `model_id`; thresholds and skip lists are optional per-dimension overrides.

→ Full `models.yaml` schema and examples: [DEVELOP.md § Model Configuration](DEVELOP.md)

---

## Documentation

| Document | Contents |
|---|---|
| [docs/index.md](docs/index.md) | Documentation map, naming rules, and public/private report boundary |
| [DEVELOP.md](DEVELOP.md) | Developer setup, architecture, dimension reference, model config schema, contributing guide |
| [RELEASE.md](RELEASE.md) | Version history, breaking changes, migration notes |
| [reports/amd-linux.en.md](reports/amd-linux.en.md) | AMD Linux benchmark results — current 192.168.100.201 target state and model selection recommendations |
| [reports/amd-windows.en.md](reports/amd-windows.en.md) | AMD Windows benchmark results — historical dual-boot run with calibrated thresholds |
| [reports/intel-windows.en.md](reports/intel-windows.en.md) | Intel Windows benchmark results |
| [docs/amd-intel-linux-test-plan.md](docs/amd-intel-linux-test-plan.md) | AMD/Intel Linux post-Windows execution plan and CPU-baseline policy |
| [docs/contributing.md](docs/contributing.md) | How to add models, dimensions, and hardware targets |
| [docs/academic-rigor.md](docs/academic-rigor.md) | Statistical rigor principles (multi-seed, effect sizes, calibration) |

---

## License

[Apache License 2.0](LICENSE)
