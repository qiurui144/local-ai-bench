> [English](./amd-windows-igpu.en.md)

# AMD Windows — iGPU（Radeon 780M / Vulkan + DirectML）性能报告

**硬件：** AMD Ryzen 8845H + Radeon 780M（RDNA3），17.9 GiB 共享显存  
**后端：** Ollama Vulkan（LLM / Embedding）· ONNX DirectML（OCR）  
**最后校准：** 2026-06-19

---

## 配置

### Ollama — 启用 Vulkan GPU 卸载

Radeon 780M 需要设置 `HSA_OVERRIDE_GFX_VERSION=gfx1102` 才能被 Ollama 的 ROCm/Vulkan 路径识别。在启动 `ollama serve` 之前设置：

```cmd
setx /M OLLAMA_HOST 0.0.0.0
setx /M HSA_OVERRIDE_GFX_VERSION gfx1102
ollama.exe serve
```

验证 GPU 已激活 — `ollama ps` 应显示正在运行的模型为 `100% GPU`：

```cmd
ollama ps
# NAME         ID       SIZE    PROCESSOR  UNTIL
# qwen2.5:7b   ...      5.2GB   100% GPU   ...
```

### ONNX DirectML OCR — 无需额外配置

`rapidocr-amd-directml` 使用 `onnxruntime-directml` 的 DirectML 后端。无需额外环境变量；DirectML 自动选择 780M。

---

## LLM 测试结果（Ollama Vulkan）

| 模型 | TPS | TTFT p50 | TTFT p95 | PP t/s | TG t/s | 最大上下文 | 状态 |
|---|---|---|---|---|---|---|---|
| `qwen2.5-7b-amd-win` | 13.33 | 953 ms | 6241 ms | 116 | 16 | 16k | FAIL（质量）|
| `qwen2.5-14b-amd-win` | 7.67 | 8274 ms | 14792 ms | 94 | 9 | 16k | MEASURED |
| `llama3.2-3b-amd-win` | 28.99 | 890 ms | 5207 ms | 124 | 39 | 32k | FAIL（质量）|
| `qwen3-0.6b-amd` | 91.09 | 1781 ms | 1781 ms | — | — | — | FAIL（质量）|
| `llava-7b-amd-win` | 16.84 | 890 ms | 891 ms | 835 | 19 | — | FAIL（准确率）|

**PP/TG** = 预填充 / 生成 token 速率（llama-bench 风格）。  
**状态说明：** 质量维度 FAIL（translation/scenarios/conditioned）表示模型在本硬件层级未通过质量门控。性能数据有效且已测量。

### 并发（Vulkan）

| 模型 | 峰值并发 | 峰值下持续 TPS |
|---|---|---|
| `llama3.2-3b-amd-win` | c50 | 36.21 t/s |
| `llama3.2-3b-amd-win` | c16 上限 | 37.88 t/s |
| `qwen2.5-7b-amd-win` | c8 | 16.70 t/s |
| `qwen2.5-14b-amd-win` | c8 上限 | 8.95 t/s |

---

## Embedding 测试结果（Ollama Vulkan）

| 模型 | hit@1 | nDCG@10 | MRR | p50 延迟 | 状态 |
|---|---|---|---|---|---|
| `qwen3-embedding-0.6b-amd` | 1.000 | 1.000 | 1.000 | 875 ms | **PASS** |
| `bge-m3-amd` | 1.000 | 1.000 | 1.000 | 914 ms | **PASS** |

---

## OCR 测试结果（ONNX DirectML — 780M）

| 模型 | CER | NED | p50 | 结构化字段准确率 | 结构化 p50 | 状态 |
|---|---|---|---|---|---|---|
| `rapidocr-amd-directml` | 7.04% | 6.18% | 468.5 ms | 92.86% | 476.5 ms | **PASS** |

**DirectML 是 AMD Windows 上最快的 OCR 路径** — 比 CPU ONNX 快 3.4×，比 VitisAI NPU 快 4.3×。
