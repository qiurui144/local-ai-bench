> [English](./intel-windows-igpu.en.md)

# Intel Windows — iGPU / OpenVINO / DirectML 路径

**硬件：** Intel Core Ultra（集成显卡）  
**后端：** OpenVINO EP（OCR — PASS）· ONNX DirectML（OCR — FAIL，ASR — PASS）  
**最后校准：** 2026-06-19

---

## Intel Windows 上 iGPU 的覆盖范围

| 任务 | 路径 | 后端 | 状态 |
|---|---|---|---|
| OCR 文字 | `rapidocr-intel-openvino` | OpenVINO | **PASS** |
| OCR 结构化 | `rapidocr-intel-openvino` | OpenVINO | **PASS** |
| OCR 文字 | `rapidocr-intel-directml` | DirectML | **FAIL** |
| OCR 结构化 | `rapidocr-intel-directml` | DirectML | **FAIL** |
| ASR | `sensevoice-small-intel-win` | DirectML | **PASS** |
| LLM | 未配置 iGPU 加速 | — | 未测试 |
| Embedding | 未配置 iGPU 加速 | — | 未测试 |

Intel Windows 上 LLM 推理目前通过 Ollama 纯 CPU 运行（见 [CPU 模式文档](./intel-windows-cpu.zh.md)）。Intel iGPU LLM 加速需配置 OpenVINO 后端 — 尚未测试。

---

## OCR 测试结果

### OpenVINO（PASS）

| 模型 | CER | NED | p50 OCR | 结构化字段准确率 | 结构化 p50 | 状态 |
|---|---|---|---|---|---|---|
| `rapidocr-intel-openvino` | 7.04% | 6.18% | 797 ms | 92.86% | 867.5 ms | **PASS** |

OpenVINO 根据可用性自动选择计算设备（CPU / iGPU / NPU）。797 ms 结果来自本平台的默认设备选择。

### DirectML（FAIL）

| 模型 | CER | NED | p50 | 状态 | 根因 |
|---|---|---|---|---|---|
| `rapidocr-intel-directml` | 202.35% | 97.77% | 945.5 ms | **FAIL** | 驱动精度问题 — 输出为乱码 |

Intel DirectML OCR **不可用**（CER 202% 意味着输出比空文本还差）。根因：Intel 硬件上 DirectML 执行路径中 PP-OCRv4 模型的 FP16 精度不匹配。请改用 OpenVINO 路径。

---

## ASR 测试结果（DirectML）

| 模型 | CER | RTF | 状态 |
|---|---|---|---|
| `sensevoice-small-intel-win` | 7.69% | 0.341 | **PASS** |

**RTF 0.341** 意味着 1 秒音频仅需 341 ms 处理 — 比实时快 2.9×。Intel ASR RTF（0.341）比 AMD（0.073）慢 4.7×，由 Radeon 780M RDNA3 与 Intel 集成显卡之间的 DirectML 吞吐差异驱动。

---

## OCR 路径对比（Intel 平台）

| 路径 | 后端 | p50 OCR | p50 结构化 | 状态 |
|---|---|---|---|---|
| Intel DirectML | ONNX DirectML | 946 ms | 985 ms | **FAIL** — 请勿使用 |
| **Intel OpenVINO** | **ONNX OpenVINO** | **797 ms** | **868 ms** | **PASS — 推荐** |
| CPU ONNX（参考） | ONNX CPU | 1593 ms | 859 ms | PASS（参考值） |

---

## Intel NPU（尚未测试）

Intel Core Ultra 处理器内置 Intel NPU（神经处理单元），可通过 OpenVINO NPU 插件或 Windows NPU SDK 访问。

当前状态：**未测试**。OpenVINO 使用 `device="NPU"` 是 Intel NPU 加速 OCR/ASR ONNX 模型的预期路径。
