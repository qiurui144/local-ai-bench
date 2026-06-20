> [English](./k3-riscv.en.md)

# K3 RISC-V 平台 — 模型选型与基准测试报告

**平台：** k3-riscv | SpacemiT K3，16 GB RAM，RISC-V RVV，llama.cpp（llama-server port 8080）  
**最后校准：** 2026-06-20。本文件原地更新。

---

## 选型摘要

| 角色 | 推荐模型 | 选型理由 |
|---|---|---|
| LLM（RISC-V） | `qwen2.5-0.5b-k3-riscv` | 本平台目前唯一已校准模型；TTFT/TPS 受 RISC-V CPU 限制；general_ability PASS |

---

## 模型测试结果

| 模型 | 角色 | 状态 | 关键指标 |
|---|---|---|---|
| `qwen2.5-0.5b-k3-riscv` | llm_riscv_primary | **PASS（部分）** | TTFT p50 ≈ 640 ms；TPS ≈ 1.4 t/s；general_ability PASS（gsm8k 66%）；translation PENDING |

**状态说明：** PASS = 所有阈值达标。PASS（部分）= 已测维度通过，但至少有一个维度为 PENDING。PENDING = 尚未测量。

---

## 已校准阈值

### `qwen2.5-0.5b-k3-riscv`

| 指标 | 阈值 |
|---|---|
| TTFT p50 | ≤ 800 ms |
| TTFT p95 | ≤ 1200 ms |
| 吞吐量 | ≥ 1.0 t/s |

---

## 已知局限

- **吞吐极低** — ~1.4 t/s 受 RISC-V CPU 算力限制；本平台不适合交互或高并发工作负载。
- **不支持高并发** — 仅适合单用户 / 单请求场景。
- **translation PENDING** — 翻译维度尚未测量；需专项校准跑。
- **conditioned PENDING** — 长上下文 conditioning 未测试。
- **可选模型有限** — 在此吞吐水平下仅 0.5B 参数量模型可用；更大模型未经测试。

---

## 校准历史

| 日期 | 事件 |
|---|---|
| 2026-06-20 | 初次校准：TTFT、吞吐量、general_ability（gsm8k）已从真实设备 E2E 跑数据中测量并设定阈值 |
