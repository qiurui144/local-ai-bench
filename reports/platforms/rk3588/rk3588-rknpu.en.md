# RK3588 RKNPU3 Path

**Last updated:** 2026-07-08
**Chinese version:** [rk3588-rknpu.zh.md](rk3588-rknpu.zh.md)
**Legacy source:** [../../rk3588.en.md](../../rk3588.en.md)

## Scope

The RK3588 on-die RKNPU3 path is used for embedding in the current deployment. LLM/VLM/ASR are served by the RK1828 PCIe NPU path, not this path.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| Embedding | `minicpm-embed-rk3588` | hit@1 1.0, MRR 1.0, nDCG@10 1.0, p50 143ms | PASS | Default RK embedding path |
| LLM/VLM | RK3588 RKNPU3 | not assigned | N/A | Use RK1828 NPU path |
| ASR/TTS | RK3588 RKNPU3 | not assigned | N/A | Use RK1828 NPU path |

## Decision

Keep RK3588 RKNPU3 as the embedding-only path in selection reports. Do not mix its metrics with RK1828 LLM/VLM/ASR results.
