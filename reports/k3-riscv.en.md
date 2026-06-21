# K3 RISC-V Platform — Model Selection & Benchmark Report

**Platform:** k3-riscv | SpacemiT K3, 8×X100 RISC-V RVV, A100 NPU + IME2, 16 GB LPDDR5
**Primary framework:** llama.cpp with IME2 acceleration (llama-server v8355; port 11434 = 3B, port 8081 = 1.5B, port 11435 = 7B)
**Reference:** attune-k3/docs/k3-16g-model-selection.md (2026-06-20 E2E verified)
**SSH:** root@192.168.100.215 (pass: bianbu)
**Last calibrated:** 2026-06-21. This file is updated in place.

---

## 16 GB Memory Budget

| Component | Estimated | Notes |
|---|---|---|
| OS + NAS services (OMV/samba/nfs) | ~1–2 GB | Idle measured: ~1 GB used |
| Bottom AI models (load-on-demand) | ~2–3 GB | 4 models, 2.4 GB disk; runtime subset resident |
| **Chat LLM** | **~4–6 GB** | 7B-class q4 allocation |
| Working memory + index + concurrent buffers | ~3–4 GB | Multi-user, 8-core, swap=0 |
| **Total** | **~12–15 GB** | 16 GB can fit 7B-class LLM with margin |

> **swap=0 hard constraint**: no swap at full load → OOM kills sshd/services instantly.
> Chat LLM + bottom models must Σ ≤ ~13 GB with safety margin. Concurrency requires thread partitioning + cgroup MemoryMax.

---

## Bottom Models (4 Types — K3 E2E Verified)

All four bottom models run as local ONNX via ORT/sherpa-onnx on the K3 X100 CPU.
**Source:** attune-k3 `reports/2026-06-18_full-verify.md`

| Capability | Model | Framework | Disk | K3 Measured |
|---|---|---|---|---|
| **Embedding** | Xenova/bge-m3 (ONNX) | ORT (load-dynamic) | 560 MB | search 77 ms cold; grounding correct ✓ |
| **Reranker** | BAAI/bge-reranker-base full | ORT | 1.1 GB | ranking correct ✓ (Xenova quantized has zh long-doc bug) |
| **OCR** | PP-OCRv4 (RapidOCR) + layout PicoDet (CDLA) | ORT | 23 MB | 315 ms, accurate ✓ + structured layout functional ✓ |
| **ASR** | sherpa SenseVoice + diarization (pyannote+campplus) | sherpa-onnx | 767 MB | en/zh/ja correct; RTF 0.17 (10× faster than Whisper) + 4-speaker separation RTF 0.76 ✓ |

**Total bottom-model disk: ~2.4 GB.** All 4 verified on K3 real hardware.

---

## Chat LLM — Dual-Framework Architecture (User Decision 2026-06-20)

**Framework: local + cloud dual-track, cloud off by default, local = best framework.**

### Local (enabled by default — privacy-first)

- **Best framework:** llama.cpp with IME2 acceleration — SpacemiT has IME2 integration on K3 X100; this is the best local LLM framework for RISC-V currently. (Ollama = vendored llama.cpp; either works.)

| Candidate | q4 RAM | Suitability | Status |
|---|---|---|---|
| **Qwen2.5-7B-Instruct q4** | ~4.5 GB | **16 GB primary recommendation** — quality/resource balance | **PENDING-VERIFY** (benchmark queued, starts after 1.5B ~17:15) |
| Qwen2.5-3B-Instruct q4 | ~2.2 GB | Low-resource / high-concurrency fallback | **PASS** (2026-06-21): PP 572 t/s / TG 7.1 t/s / TTFT P50 184ms; GA PASS; translation PASS |
| Qwen2.5-1.5B-Instruct q4 | ~1.1 GB | Minimal-footprint option | **PENDING-VERIFY** (benchmark running 2026-06-21, est. ~17:45 completion) |
| Qwen3-30B-A3B q4 (MoE) | ~16–18 GB | ❌ **Exceeds 16 GB** — reserved for 32 GB device | 32 GB: measured TG 13.3 t/s (SpacemiT modelzoo) |

- **Acceleration:** X100 RVV + IME2 (INT8/INT4); A100 NPU offload under evaluation.

### Cloud (off by default — opt-in)

When off: data and inference stay on K3 (privacy flagship).
When on (per global §4.5H): text default **deepseek-v4**, multimodal **qwen-3.6/qwen-3.7**; via OpenAI-compat gateway.

---

## Selection Summary

| Slot | Selected | Default |
|---|---|---|
| Embedding | bge-m3 ONNX (ORT) | ✅ Local |
| Reranker | bge-reranker-base full (ORT) | ✅ Local |
| OCR | PP-OCRv4 + layout (ORT) | ✅ Local |
| ASR | sherpa SenseVoice + diarization | ✅ Local |
| **Chat LLM** | **Local llama.cpp+IME2 (Qwen2.5-7B q4 primary)** + cloud deepseek-v4 (off) | Local first, cloud opt-in |

**16 GB budget total:** bottom ~2.5 GB + 7B LLM ~4.5 GB + working memory ≈ 12–13 GB — feasible with margin.

---

## LLM Benchmark Results

### qwen2.5-3b-k3-riscv (2026-06-21 — Clean Calibration Run)

**llama-server v8355 on port 11434, Qwen2.5-3B-Instruct Q4_K_M via Ollama K3_LLM_BASE_URL**

| Metric | Measured | Threshold | Status |
|---|---|---|---|
| TTFT warm P50 | 184 ms (177/178/184/197 ms) | ≤ 300 ms | **PASS** |
| TTFT cold (1st call) | 1031 ms | ≤ 1200 ms | **PASS** |
| Prefill (PP) | 572 t/s mean (545/571/602) | ≥ 300 t/s | **PASS** |
| Decode (TG) | 7.1 t/s mean (7.0/7.2/7.0) | ≥ 4 t/s | **PASS** |
| Throughput | ~4–7 t/s | ≥ 2 t/s | **PASS** |
| Translation zh→en | chrF 57.5 (flores) / 70.4 (term, 74% term-match) | BLEU≥14 / chrF≥30 | **PASS** |
| Translation en→zh | chrF 33.6 (flores) / 32.4 (term, 57% term-match) | chrF≥30 / term≥50% | **PASS** |
| General ability (GSM8K/MMLU/HellaSwag) | 0.550 / 0.500 / 0.750 (n=20 each) | ≥40% each | **PASS** |

> Note: Previous calibration showed PP≈361 t/s and TG≈4.4 t/s due to duplicate background processes contaminating measurements. Clean-run values (PP≈572, TG≈7.1) are the authoritative baseline.

### qwen2.5-0.5b-k3-riscv (2026-06-20 historical reference — BLOCKED on current K3)

> **BLOCKED (2026-06-21):** No Qwen2.5-0.5B GGUF file found in `/root/models/` on current K3 device.
> Historical calibration data below preserved for reference; translation was PENDING-VERIFY and never completed.

| Metric | Measured | Threshold | Status |
|---|---|---|---|
| TTFT p50 | ~640 ms | ≤ 800 ms | *(historical)* |
| TTFT p95 | — | ≤ 1200 ms | *(historical)* |
| Throughput | ~1.4 t/s | ≥ 1.0 t/s | *(historical)* |
| general_ability (gsm8k) | 66% | — | *(historical)* |
| translation | — | — | **PENDING-VERIFY** |

### Calibrated Thresholds (qwen2.5-3b-k3-riscv) — 2026-06-21

| Metric | Threshold | Source |
|---|---|---|
| TTFT p50 | ≤ 300 ms | warm P50=184ms × 1.6x |
| TTFT p95 | ≤ 1200 ms | cold first-call 1031ms × 1.16x |
| PP t/s | ≥ 300 t/s | measured min 545 t/s |
| TG t/s | ≥ 4 t/s | measured min 7.0 t/s |
| Throughput | ≥ 2 t/s | TG floor with margin |

### Calibrated Thresholds (qwen2.5-0.5b-k3-riscv) — 2026-06-20

| Metric | Threshold |
|---|---|
| TTFT p50 | ≤ 800 ms |
| TTFT p95 | ≤ 1200 ms |
| Throughput | ≥ 1.0 t/s |

---

## PENDING-VERIFY (must run on K3, per §1.6)

**Benchmarks in progress (2026-06-21):**
- `qwen2.5-1.5b-k3-riscv` — 3-seed benchmark running, run 2/3 started 15:37 (est. ~17:15 completion); conditioned/conversation_drift skipped (-c 4096 server)
- `qwen2.5-7b-k3-riscv` — server running on port 11435 (confirmed 2026-06-21); benchmark queued after 1.5B completes (~17:15)

**Remaining verification items:**
1. **Qwen2.5-1.5B-Instruct q4** — TTFT/PP/TG/GA/translation thresholds (benchmark running; clean re-run for perf dims needed after GA completes)
2. **Qwen2.5-7B-Instruct q4** t/s + peak RAM on K3 X100+IME2 (confirm 16 GB fits + usable speed)
3. Peak RAM with 4 bottom models + 7B LLM resident simultaneously (16 GB swap=0 OOM boundary)
4. Multi-user concurrent chat with thread partitioning + cgroup MemoryMax
5. A100 NPU offload benefit for LLM/embedding

---

## Known Limitations

- **Extremely low throughput (0.5B):** ~1.4 t/s is constrained by RISC-V CPU; not suitable for interactive or high-concurrency workloads at this model size.
- **7B model throughput unknown:** Expected to be ~5–10 t/s with IME2; needs verification.
- **No high-concurrency support:** Single-user / single-request workloads only (swap=0, no OOM buffer).
- **Model selection limited at 0.5B:** Only 0.5B models are practical at current calibrated throughput; 7B is primary recommendation pending verification.
- **30B MoE out of range:** Qwen3-30B-A3B q4 requires ~16–18 GB — exceeds 16 GB device budget; reserved for 32 GB variant.

---

## Calibration History

| Date | Event |
|---|---|
| 2026-06-20 | Initial calibration: TTFT, throughput, general_ability (gsm8k) measured; thresholds set from E2E device runs (qwen2.5-0.5b) |
| 2026-06-20 | Expanded: K3 model selection from attune-k3 reference (bottom models verified, 7B LLM dual-framework decision, memory budget analysis) |
| 2026-06-21 | Added qwen2.5-3b-k3-riscv: new llama-server port 11434 (IP 215); clean single-process run: PP≈572 t/s (545/571/602), TG≈7.1 t/s (7.0/7.2/7.0), TTFT warm P50≈184ms (cold 1031ms); translation PASS (zh→en chrF 57.5/70.4; en→zh chrF 33.6/32.4); GA PASS (GSM8K 0.550/MMLU 0.500/HellaSwag 0.750) |
| 2026-06-21 | Added qwen2.5-1.5b-k3-riscv (port 8081) and qwen2.5-7b-k3-riscv (port 11435); 7B server confirmed running; 1.5B benchmark in progress (GA phase, ~14:40 est.); 7B benchmark queued after 1.5B completes |

---

## 中文摘要

**平台：** k3-riscv | SpacemiT K3，8×X100 RISC-V RVV，A100 NPU + IME2，16 GB LPDDR5
**主框架：** llama.cpp（IME2 加速，llama-server v8355；端口 11434=3B，8081=1.5B，11435=7B，8080=nginx NAS 禁用）
**参考：** attune-k3/docs/k3-16g-model-selection.md（2026-06-20 端到端验证）  
**SSH：** root@192.168.100.215（密码：bianbu）

### 内存预算（16 GB）

| 组件 | 估算 | 说明 |
|---|---|---|
| OS + NAS 服务（OMV/samba 等） | ~1–2 GB | 空载实测约 1 GB |
| 底座 AI 模型（按需加载） | ~2–3 GB | 4 类模型磁盘共 2.4 GB |
| **Chat LLM** | **~4–6 GB** | 7B 量化模型（q4）分配 |
| 工作内存 + 索引 + 并发缓冲 | ~3–4 GB | 多用户场景 |
| **合计** | **~12–15 GB** | 16 GB **可容纳 7B LLM**，留余量 |

### 底座模型（4 类，K3 真机已验证）

| 能力 | 模型 | 框架 | 磁盘 | K3 实测 |
|---|---|---|---|---|
| Embedding | Xenova/bge-m3 (ONNX) | ORT | 560 MB | 搜索 77 ms 冷启动；检索准确 ✓ |
| 重排序 | BAAI/bge-reranker-base full | ORT | 1.1 GB | 排序正确 ✓（Xenova 量化版中文长文档有 bug，故用 full）|
| OCR | PP-OCRv4 + layout PicoDet | ORT | 23 MB | 315 ms，准确 ✓ + 结构化版式 ✓ |
| ASR | sherpa SenseVoice + 说话人分离 | sherpa-onnx | 767 MB | en/zh/ja 准确；RTF 0.17；4 人分离 RTF 0.76 ✓ |

### Chat LLM 选型（双模型框架，用户 2026-06-20 决策）

| 候选 | q4 内存 | 状态 |
|---|---|---|
| **Qwen2.5-7B-Instruct q4**（主推） | ~4.5 GB | **PENDING-VERIFY**（benchmark 排队中，预计 1.5B 完成后 ~17:15 启动）|
| Qwen2.5-3B-Instruct q4（兜底） | ~2.2 GB | **已验证 2026-06-21**：TTFT warm P50≈184ms，PP≈572 t/s，TG≈7.1 t/s；翻译 PASS；GA PASS（GSM8K 0.55/MMLU 0.50/HellaSwag 0.75） |
| Qwen2.5-1.5B-Instruct q4（极简） | ~1.1 GB | **PENDING-VERIFY**（benchmark 运行中，run 2/3，2026-06-21 约 17:15 完成）|
| Qwen3-30B-A3B q4（MoE） | ~16–18 GB | ❌ 超 16 GB，留 32G 设备 |

**本地框架**：llama.cpp + IME2（最优，RVV + INT8/INT4 加速）。**云端**：deepseek-v4（默认关闭，隐私优先）。
