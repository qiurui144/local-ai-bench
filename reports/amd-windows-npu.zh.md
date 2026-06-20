> [English](./amd-windows-npu.en.md)

# AMD Windows — NPU（AMD XDNA / VitisAI）性能报告

**硬件：** AMD Ryzen 8845H，AMD AI 300 系列 NPU（XDNA，16 TOPS）  
**后端：** ONNX Runtime VitisAI（OCR）· ONNX DirectML（ASR）  
**最后校准：** 2026-06-19

---

## AMD Windows 上 NPU 的覆盖范围

| 任务 | NPU 路径 | 状态 |
|---|---|---|
| OCR（文字 + 结构化） | VitisAI EP（onnxruntime-vitisai） | **PASS** |
| ASR | DirectML（onnxruntime-directml 通过 XDNA） | **PASS** |
| LLM 推理 | 不支持 — AMD XDNA 需专有 AMD NPU SDK；Ollama 使用 Vulkan iGPU | N/A |
| Embedding / Reranker | NPU 不支持；使用 Vulkan（Ollama）或 CPU（ONNX） | N/A |

AMD XDNA NPU 面向固定功能 ONNX 模型执行。通用 LLM 服务（如 Ollama）不使用 NPU — LLM 推理通过 Vulkan 路由到 780M iGPU。LLM 性能请见 [iGPU 模式文档](./amd-windows-igpu.zh.md)。

---

## 配置

### VitisAI OCR

需安装 AMD RyzenAI 运行时及 `onnxruntime-vitisai` 包：

```cmd
pip install onnxruntime-vitisai
```

当模型目标为 `amd-npu` 时，VitisAI EP 会自动被选中：
```bash
# 运行前设置 ONNX 后端
export OCR_BACKEND=vitisai
python run_benchmark.py --model rapidocr-amd-npu --target amd-win-x86
```

### DirectML ASR

无需在标准 `onnxruntime-directml` 之外额外安装。ASR 基准测试会自动调用 DirectML：
```bash
python run_benchmark.py --model sensevoice-small-amd-win --target amd-win-x86 --skip ttft,throughput,embedding,rerank,ocr
```

---

## OCR 测试结果（VitisAI NPU）

| 模型 | CER | NED | p50 | 结构化字段准确率 | 结构化 p50 | 状态 |
|---|---|---|---|---|---|---|
| `rapidocr-amd-npu` | 7.04% | 6.18% | 2031 ms | 92.86% | 1867.5 ms | **PASS** |

**质量与 DirectML 和 CPU 路径完全一致**（CER 7.04% 是数据集下限）。  
**延迟高于 DirectML**：VitisAI p50 为 2031 ms，DirectML 为 469 ms。  
在热功耗/功耗受限的批处理场景下使用 VitisAI，可节省 iGPU 带宽给并发 LLM 使用。

---

## ASR 测试结果（DirectML — AMD 平台）

| 模型 | CER | RTF | 状态 |
|---|---|---|---|
| `sensevoice-small-amd-win` | 7.69% | 0.073 | **PASS** |

**RTF 0.073** 意味着 1 秒音频仅需 73 ms 处理 — 比实时快 13.7×。这是 AMD Windows 上最佳的 ASR 路径。

---

## OCR 三条路径性能对比

| 路径 | 后端 | p50 OCR | p50 结构化 | 备注 |
|---|---|---|---|---|
| CPU ONNX | CPU | 1593 ms | 859 ms | [→ CPU 文档](./amd-windows-cpu.zh.md) |
| **iGPU DirectML** | 780M | **469 ms** | **477 ms** | **最快 — 推荐** |
| NPU VitisAI | XDNA | 2031 ms | 1868 ms | 批处理/低功耗场景 |
