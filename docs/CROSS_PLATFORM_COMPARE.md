# Cross-Platform Comparison Guide

> 跨平台可替换性评估的语义定义与操作流程。

## 背景

北极星问题："模型 X 放进 AI-box 产品，能不能替换 Y？性能够不够、效果掉不掉？"

跨平台场景：同一问题的评估对象跨越不同硬件（AMD Vulkan / RK3588 RKNN / K3 RISC-V / Intel CPU），
不同于 `--compare` 的默认同平台模式。本文定义三类跨平台场景及其操作 SOP。

---

## 场景分类

### 场景 A：同模型异平台（验证 OK）

**目的**：确认同一模型在不同硬件上效果一致、推理结果无退化。

**典型用例**：
```
qwen2.5:7b on AMD Windows (Vulkan) vs qwen2.5:7b on Intel Linux (CPU)
```

**结论语义**：
- 质量维度（accuracy/translation/general_ability 等）：期望 REPLACEABLE（同模型权重）
- 性能维度（ttft/throughput）：必然 INCONCLUSIVE（hardware_profile 不同是预期行为，不是错误）
- **综合结论**：若质量 REPLACEABLE + 性能 INCONCLUSIVE → **"质量等价，性能需独立评估"**

**操作 SOP**：
```bash
# Step 1: 两台机器各跑 --seeds 3
export OLLAMA_AMD_BASE_URL=http://<AMD_IP>:11434/v1
python run_benchmark.py --model qwen2.5-7b-amd-win --seeds 3 \
  --skip stability,concurrency

export OLLAMA_INTEL_LINUX_BASE_URL=http://<INTEL_IP>:11434/v1
python run_benchmark.py --model qwen2.5-7b-intel-linux --seeds 3 \
  --skip stability,concurrency

# Step 2: 跨平台质量对比（忽略性能侧 INCONCLUSIVE）
python run_benchmark.py --compare qwen2.5-7b-amd-win qwen2.5-7b-intel-linux
# 预期：INCONCLUSIVE（因 hardware_profile 不同），但 quality 部分所有维度应在 2σ 内
```

**解读规则**：当 `--compare` 返回 INCONCLUSIVE，需手动检查 `quality` 字段：
- 所有 significant=false 或 direction=better/equal → **质量等价（可在报告中标注）**
- 任何 significant=true + direction=worse → **质量退化，NOT_REPLACEABLE**

---

### 场景 B：异模型异平台（可替换性评估）

**目的**：评估是否能用目标平台上的某模型替换基准平台上的另一模型。

**典型用例**：
```
baseline: llama3.2:3b on AMD Windows (780M Vulkan) — 当前部署
candidate: qwen2.5-0.5b on RK3588 (RKNN NPU) — 候选替换
```

**结论语义**：
- 这是最复杂的场景，性能和质量都可能不同
- 质量：使用 2σ 显著性检验
- 性能：候选自身与其 models.yaml 阈值比对（不与 baseline 性能比）
- hardware_profile 不同 → 性能侧强制 INCONCLUSIVE（设计如此）

**操作 SOP**：
```bash
# Step 1: baseline 完整评测
export OLLAMA_AMD_BASE_URL=http://<AMD_IP>:11434/v1
python run_benchmark.py --model llama3.2-3b-amd-win --seeds 3

# Step 2: candidate 完整评测（在目标平台）
export RK3588_LLM_BASE_URL=http://<RK3588_IP>:8080/v1
python run_benchmark.py --model qwen2.5-0.5b-rk3588 --seeds 3 \
  --skip stability,conditioned,scenarios

# Step 3: 异平台对比（性能侧必然 INCONCLUSIVE）
python run_benchmark.py --compare llama3.2-3b-amd-win qwen2.5-0.5b-rk3588

# Step 4: 人工补充性能对比（输出对比表）
python -c "
import json
from pathlib import Path
amd = json.loads(sorted(Path('output/reports').glob('llama3.2-3b-amd-win_*.json'))[-1].read_text())
rk = json.loads(sorted(Path('output/reports').glob('qwen2.5-0.5b-rk3588_*.json'))[-1].read_text())
print('TTFT AMD:', amd.get('benchmarks', {}).get('ttft', {}).get('ttft_ms_stats', {}).get('p95', 'N/A'), 'ms')
print('TTFT RK:', rk.get('benchmarks', {}).get('ttft', {}).get('ttft_ms_stats', {}).get('p95', 'N/A'), 'ms')
print('TPS AMD:', amd.get('benchmarks', {}).get('throughput', {}).get('aggregate_tps', 'N/A'))
print('TPS RK:', rk.get('benchmarks', {}).get('throughput', {}).get('aggregate_tps', 'N/A'))
"
```

---

### 场景 C：同平台模型选型对比（标准场景）

**目的**：在相同硬件上对比两个不同模型，确定哪个更适合部署。

**典型用例**：
```
llama3.2:3b vs qwen2.5:3b，均在 Intel Windows CPU (Ollama)
qwen3-0.6b vs qwen3-embedding-0.6b，均在 AMD Windows (Vulkan Ollama)
```

**操作 SOP**（标准 `--compare` 流程）：
```bash
# 两个模型均在同一平台跑
export OLLAMA_INTEL_WIN_BASE_URL=http://<INTEL_WIN_IP>:11434/v1
python run_benchmark.py --target intel-win-x86 --model llama3.2-1b-intel-win --seeds 3
python run_benchmark.py --target intel-win-x86 --model qwen2.5-3b-intel-win --seeds 3

# 标准对比（hardware_profile 相同，REPLACEABLE 可出完整结论）
python run_benchmark.py --compare llama3.2-1b-intel-win qwen2.5-3b-intel-win
```

---

## 六平台优先评测矩阵

按用户优先级排序的推荐对比组合（E2E 阶段执行）：

| 优先级 | 平台 | 基准模型 | 候选比较 | 跳过维度 |
|--------|------|---------|---------|---------|
| P1 | AMD Windows | qwen2.5-7b-amd-win | llama3.2-3b-amd-win | stability |
| P1 | AMD Windows | qwen2.5-7b-amd-win | qwen3-0.6b-amd | stability |
| P2 | Intel Windows | qwen2.5-3b-intel-win | llama3.2-1b-intel-win | stability,concurrency |
| P2 | Rockchip Linux | qwen2.5-0.5b-rk3588 | minicpm-v-rk3588 (VLM) | stability,conditioned,scenarios |
| P3 | K3 RISC-V | qwen2.5-0.5b-k3-riscv | llama3.2-1b-k3-riscv | stability,concurrency,conditioned,scenarios,embedding,rerank,asr,ocr |
| P3 | Intel Linux | qwen2.5-7b-intel-linux | llama3.2-3b-intel-linux | stability |

### 跨平台对比组合（场景 A，验证一致性）

| 模型 | 基准平台 | 目标平台 | 目的 |
|------|---------|---------|------|
| qwen2.5:7b | AMD Windows | Intel Linux | 确认 Vulkan vs CPU 效果等价 |
| llama3.2:3b | AMD Windows | Intel Windows | 确认 Vulkan vs CPU 效果等价 |
| qwen3-embedding:0.6b | AMD Windows | Intel Windows | 确认 embedding 效果平台无关 |
| qwen3-embedding:0.6b | AMD Windows | Intel Linux | 同上 |

---

## 性能阈值定义原则

各平台模型的 performance threshold 定义在 `models.yaml::benchmarks` 字段下。
原则：
- 阈值以**目标平台为准**，不与跨平台基准对比
- CPU 平台阈值显著宽于 GPU/NPU 平台（TTFT 可放宽 10-20x）
- 阈值**未校准**的状态用 `# PENDING-VERIFY` 注释标记（见 RELEASE.md Known Limitations）

### 各平台预期性能区间（理论估算，E2E 后校准）

| 平台 | 模型规模 | 预期 TTFT (p95) | 预期 TPS (输出) |
|------|---------|----------------|----------------|
| AMD 780M Vulkan | 0.6B Q4 | ≤ 500ms | ≥ 15 tok/s |
| AMD 780M Vulkan | 3B Q4 | ≤ 2000ms | ≥ 8 tok/s |
| AMD 780M Vulkan | 7B Q4 | ≤ 8000ms | ≥ 3 tok/s |
| Intel CPU | 1B Q4 | ≤ 5000ms | ≥ 2 tok/s |
| Intel CPU | 3B Q4 | ≤ 15000ms | ≥ 0.8 tok/s |
| RK3588 RKNN NPU | 0.5B INT8 | ≤ 3000ms | ≥ 5 tok/s |
| K3 RISC-V RVV | 0.5B Q4 | ≤ 30000ms | ≥ 0.5 tok/s |

> 以上数据为预填写值，**PENDING-VERIFY**，首次 E2E 跑完后更新此表。

---

## 条件（condition）管理

`--compare` 要求 baseline 和 candidate 的 `condition` 字段必须相同。

跨平台场景时 condition 的处理方式：
1. **同 condition**（默认 `"standard"`）：直接 `--compare`，质量维度可比
2. **不同 condition**（如一个是 `"standard"` 一个是 `"edge-low-mem"`）：结果 INCONCLUSIVE

建议跨平台时统一用 `condition: standard` 确保可比性。

---

## 常见误判规避

| 误判 | 根因 | 规避方法 |
|------|------|---------|
| 跨平台 INCONCLUSIVE 被当作"不可替换" | hardware_profile 不同导致 INCONCLUSIVE 是预期行为，不代表质量问题 | 手动检查 `quality` 字段是否有 significant regression |
| 单 seed 结论当 REPLACEABLE | 单 seed 评测噪声大，2σ 无意义 | 必须 `--seeds 3`，否则 `--compare` 硬封顶 INCONCLUSIVE |
| CPU 平台 TTFT 达不到 GPU 阈值 → NOT_REPLACEABLE | 阈值未按平台分开设置 | `models.yaml` 每个平台模型单独设 `benchmarks.ttft.thresholds` |
| 跨平台性能比较结论 | 不同硬件性能必然不同 | 性能比较仅在同平台内有意义；跨平台只比质量 |

---

## 参考

- 详细部署 SOP → [docs/DEPLOY_TARGETS.md](DEPLOY_TARGETS.md)
- targets.yaml 平台注册 → [targets.yaml](../targets.yaml)
- models.yaml 模型矩阵 → [models.yaml](../models.yaml)
- compare 算法实现 → [benchmark/compare.py](../benchmark/compare.py)
