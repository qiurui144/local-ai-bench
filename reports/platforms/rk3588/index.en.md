# RK3588 + RK1828

**Last updated:** 2026-07-08
**Chinese version:** [index.zh.md](index.zh.md)
**Legacy source:** [../../rk3588.en.md](../../rk3588.en.md)

## Scope

This platform has two independent NPU paths: RK3588 on-die RKNPU3 for embedding and RK1828 PCIe NPU for LLM/VLM/ASR. The historical model IDs use `rk1820`, but the measured device is RK1828.

## Execution Path Summary

| Path | Runtime/service | Best-fit workloads | Status |
|---|---|---|---|
| [RK3588 RKNPU3](rk3588-rknpu.en.md) | uvicorn embedding service | Embedding | PASS |
| [RK1828 NPU](rk1828-npu.en.md) | rkllm3 / ASR services | LLM/VLM and ASR | PASS primary rows; context risk |

## Selection Notes

| Role | Current choice | Decision |
|---|---|---|
| LLM/VLM | `qwen3-vl-2b-rk1820` | TTFT/throughput/translation pass; conversation drift fails due 768-token runtime limit. |
| ASR | `rk-asr-rk1820` | 3-seed ASR pass; zh CER inflated by charset mismatch. |
| Embedding | `minicpm-embed-rk3588` | hit@1/MRR/nDCG 1.0; p50 143ms. |
| RKNN3 cached models | 46/46 artifacts cached | Registered but pending service load and harness calibration. |

## Evidence

| Detail | Report |
|---|---|
| RK3588 RKNPU3 | [rk3588-rknpu.en.md](rk3588-rknpu.en.md) |
| RK1828 NPU | [rk1828-npu.en.md](rk1828-npu.en.md) |
| Legacy full report | [../../rk3588.en.md](../../rk3588.en.md) |
