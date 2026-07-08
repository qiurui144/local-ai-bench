# RK1828 NPU Path

**Last updated:** 2026-07-08
**Chinese version:** [rk1828-npu.zh.md](rk1828-npu.zh.md)
**Legacy source:** [../../rk3588.en.md](../../rk3588.en.md)

## Scope

The RK1828 PCIe NPU path runs LLM/VLM and ASR services. Model IDs retain the historical `rk1820` suffix, but the measured device is RK1828.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| LLM/VLM | `qwen3-vl-2b-rk1820` | TTFT p50/p95 143/244ms; TPS 108.5 | PASS primary dims | Default RK NPU LLM/VLM |
| Translation | `qwen3-vl-2b-rk1820` | all tested directions pass | PASS | Usable within context limit |
| Conversation drift | `qwen3-vl-2b-rk1820` | structured extraction max drop 21.19% | FAIL | 768-token runtime limit risk |
| ASR | `rk-asr-rk1820` | RTF 0.0768 +/- 0.0045, CER 4.73% | PASS | Default RK ASR |
| RKNN3 cache | v1.0.4 LLM/VLM/OCR-relevant artifacts | 46/46 cached | PENDING-VERIFY | Needs service load and harness calibration |

## RKNN3 Cached Coverage

| Group | Coverage | Status |
|---|---|---|
| LLM | Qwen2.5, Qwen3, Copaw Flash rows registered | Cached, pending verify |
| VLM | FastVLM, InternVL, Janus, MiniCPM, Qwen2.5-VL, Qwen3-VL and others | Cached, pending verify |
| OCR/VLM | `paddleocr-vl-rk1820` | Cached, pending verify |

## Decision

Use RK1828 for RK-side LLM/VLM/ASR decisions. The main blocker is not throughput; it is the current RKLLM context limit and drift behavior under longer conversations.
