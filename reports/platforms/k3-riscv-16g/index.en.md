# K3 RISC-V 16G

**Last updated:** 2026-07-14
**Chinese version:** [index.zh.md](index.zh.md)
**Legacy source:** [../../k3-riscv.en.md](../../k3-riscv.en.md)

## Scope

This platform page maps the K3 16GB legacy calibration evidence into the standard reports layout. The NAS contract lists `k3-riscv-16g` as a P3 retest target, but this repository does not currently contain a v1 `parameter-matrix.json` / `run-summary.json` artifact for that target.

Rows here are therefore legacy calibrated evidence, not current NAS contract product verdicts.

## Contract Baseline

| Item | Value |
|---|---|
| target | `k3-riscv-16g` |
| contract_artifacts | none in current repo |
| report_status | `legacy_calibrated_contract_retest_pending` |
| legacy_source | [../../k3-riscv.en.md](../../k3-riscv.en.md) |

## Hardware Path Summary

| Path | Runtime | Workloads | Status |
|---|---|---|---|
| [X100 CPU + IME2](x100-ime2.en.md) | llama.cpp / llama-server v8355 | LLM chat and translation/GA probes | Legacy calibrated; contract retest pending |
| [CPU ORT / sherpa](cpu-ort.en.md) | ONNX Runtime / sherpa-onnx | embedding, reranker, OCR, ASR | Legacy calibrated; contract retest pending |
| [A100 NPU](a100-npu.en.md) | A100 NPU offload | candidate acceleration path | No current calibrated contract path |

## Decision

K3 16G remains a valid legacy-calibrated platform for Qwen2.5 3B/7B-class local LLM and bottom-model OCR/ASR/retrieval workflows. It must not be reported as NAS contract complete until a P3 contract retest produces the required artifacts.

## Evidence

| Detail | Report |
|---|---|
| X100 CPU + IME2 path | [x100-ime2.en.md](x100-ime2.en.md) |
| CPU ORT / sherpa path | [cpu-ort.en.md](cpu-ort.en.md) |
| A100 NPU path | [a100-npu.en.md](a100-npu.en.md) |
| Legacy full report | [../../k3-riscv.en.md](../../k3-riscv.en.md) |
