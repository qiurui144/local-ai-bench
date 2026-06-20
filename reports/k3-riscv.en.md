# K3 RISC-V Platform — Model Selection & Benchmark Report

**Platform:** k3-riscv | SpacemiT K3, 16 GB RAM, RISC-V RVV, llama.cpp (llama-server port 8080)
**Last calibrated:** 2026-06-20. This file is updated in place.

---

## Selection Summary

| Role | Selected Model | Rationale |
|---|---|---|
| LLM (RISC-V) | `qwen2.5-0.5b-k3-riscv` | Only calibrated model on platform; TTFT/TPS constrained by RISC-V CPU; general_ability PASS |

---

## Model Results

| Model | Role | Status | Key Metrics |
|---|---|---|---|
| `qwen2.5-0.5b-k3-riscv` | llm_riscv_primary | **PASS (partial)** | TTFT p50 ≈ 640 ms; TPS ≈ 1.4 t/s; general_ability PASS (gsm8k 66%); translation PENDING |

**Status legend:** PASS = all thresholds met. PASS (partial) = measured dims pass but at least one dim is PENDING. PENDING = not yet measured.

---

## Calibrated Thresholds

### `qwen2.5-0.5b-k3-riscv`

| Metric | Threshold |
|---|---|
| TTFT p50 | ≤ 800 ms |
| TTFT p95 | ≤ 1200 ms |
| Throughput | ≥ 1.0 t/s |

---

## Known Limitations

- **Extremely low throughput** — ~1.4 t/s is constrained by RISC-V CPU compute; this platform is not suitable for interactive or high-concurrency workloads.
- **No high-concurrency support** — Single-user / single-request workloads only.
- **translation PENDING** — Translation dimension not yet measured; needs a dedicated calibration run.
- **conditioned PENDING** — Long-context conditioning not tested.
- **Model selection limited** — Only 0.5B parameter models are practical at this throughput level; larger models are untested.

---

## Calibration History

| Date | Event |
|---|---|
| 2026-06-20 | Initial calibration: TTFT, throughput, general_ability (gsm8k) measured; thresholds set from E2E device runs |

---

## 中文摘要

**平台：** k3-riscv | SpacemiT K3，16 GB RAM，RISC-V RVV，llama.cpp（llama-server 端口 8080）
**最后校准：** 2026-06-20。本文件原地更新。

### 选型摘要

| 角色 | 推荐模型 | 选型理由 |
|---|---|---|
| LLM（RISC-V） | `qwen2.5-0.5b-k3-riscv` | 本平台目前唯一已校准模型；TTFT/TPS 受 RISC-V CPU 限制；general_ability PASS |

### 模型测试结果

| 模型 | 角色 | 状态 | 关键指标 |
|---|---|---|---|
| `qwen2.5-0.5b-k3-riscv` | llm_riscv_primary | **PASS（部分）** | TTFT p50 ≈ 640 ms；TPS ≈ 1.4 t/s；general_ability PASS（gsm8k 66%）；translation PENDING |

### 已校准阈值（`qwen2.5-0.5b-k3-riscv`）

| 指标 | 阈值 |
|---|---|
| TTFT p50 | ≤ 800 ms |
| TTFT p95 | ≤ 1200 ms |
| 吞吐量 | ≥ 1.0 t/s |

### 已知局限

- **吞吐极低** — ~1.4 t/s 受 RISC-V CPU 算力限制，本平台不适合交互或高并发工作负载。
- **不支持高并发** — 仅适合单用户/单请求场景。
- **translation PENDING** — 翻译维度尚未测量，需专项校准跑。
- **conditioned PENDING** — 长上下文 conditioning 未测试。
- **可选模型有限** — 在此吞吐水平下仅 0.5B 参数量模型可用，更大模型未经测试。
