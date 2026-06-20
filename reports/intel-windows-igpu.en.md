> [← Intel Windows overview](./intel-windows.en.md)

# Intel Windows — iGPU / OpenVINO / DirectML Paths

**Hardware:** Intel Core Ultra (integrated graphics)  
**Backends:** OpenVINO EP (OCR — PASS) · ONNX DirectML (OCR — FAIL, ASR — PASS)  
**Last calibrated:** 2026-06-19

---

## iGPU Scope on Intel Windows

| Workload | Path | Backend | Status |
|---|---|---|---|
| OCR text | `rapidocr-intel-openvino` | OpenVINO | **PASS** |
| OCR structured | `rapidocr-intel-openvino` | OpenVINO | **PASS** |
| OCR text | `rapidocr-intel-directml` | DirectML | **FAIL** |
| OCR structured | `rapidocr-intel-directml` | DirectML | **FAIL** |
| ASR | `sensevoice-small-intel-win` | DirectML | **PASS** |
| LLM | Not configured for iGPU | — | Not tested |
| Embedding | Not configured for iGPU | — | Not tested |

LLM inference on Intel Windows currently runs CPU-only via Ollama (see
[cpu mode](./intel-windows-cpu.en.md)). Intel iGPU LLM acceleration would require
configuring the OpenVINO backend in the serving stack — not yet tested.

---

## OCR Results

### OpenVINO (PASS)

| Model | CER | NED | p50 OCR | Structured field acc | Structured p50 | Status |
|---|---|---|---|---|---|---|
| `rapidocr-intel-openvino` | 7.04% | 6.18% | 797 ms | 92.86% | 867.5 ms | **PASS** |

OpenVINO auto-selects the compute device (CPU / iGPU / NPU) based on availability.
The 797 ms result is from the default device selection on this platform.

### DirectML (FAIL)

| Model | CER | NED | p50 | Status | Root cause |
|---|---|---|---|---|---|
| `rapidocr-intel-directml` | 202.35% | 97.77% | 945.5 ms | **FAIL** | Driver precision issue — output is garbled text |

Intel DirectML OCR is **not usable** (CER 202% means the output is worse than empty).
Root cause: likely FP16 precision mismatch in the DirectML execution path for
the PP-OCRv4 model on Intel hardware. Use the OpenVINO path instead.

---

## ASR Results (DirectML)

| Model | CER | RTF | Status |
|---|---|---|---|
| `sensevoice-small-intel-win` | 7.69% | 0.341 | **PASS** |

**RTF 0.341** means 1 second of audio processes in 341 ms — 2.9× faster than real-time.
Intel ASR RTF (0.341) is 4.7× slower than AMD (0.073), driven by DirectML throughput
differences between Radeon 780M RDNA3 and Intel integrated graphics.

---

## OCR Path Comparison (Intel platform)

| Path | Backend | p50 OCR | p50 Structured | Status |
|---|---|---|---|---|
| Intel DirectML | ONNX DirectML | 946 ms | 985 ms | **FAIL** — do not use |
| **Intel OpenVINO** | **ONNX OpenVINO** | **797 ms** | **868 ms** | **PASS — recommended** |
| CPU ONNX (reference) | ONNX CPU | 1593 ms | 859 ms | PASS (from reference) |

---

## Intel NPU (Not Yet Tested)

Intel Core Ultra processors include an Intel NPU (neural processing unit) accessible via
OpenVINO's NPU plugin or the Windows NPU SDK.

Current status: **not benchmarked**. OpenVINO with `device="NPU"` is the expected path
for Intel NPU acceleration of OCR/ASR ONNX models.

---

## 中文摘要

**硬件：** Intel Core Ultra 集成显卡，OpenVINO + DirectML  
**最后校准：** 2026-06-19

### iGPU 路径覆盖范围

| 任务 | 路径 | 状态 |
|---|---|---|
| OCR | OpenVINO | **PASS**（p50 797 ms） |
| OCR | DirectML | **FAIL**（CER 202%，不可用） |
| ASR | DirectML | **PASS**（RTF 0.341） |
| LLM / Embedding | 未配置 iGPU | — |

### 关键数据

- `rapidocr-intel-openvino`：CER 7.04%，p50 797 ms，结构化 p50 868 ms — **推荐 OCR 路径**
- `rapidocr-intel-directml`：CER 202.35% — **不可用，驱动精度问题**
- `sensevoice-small-intel-win`：CER 7.69%，RTF 0.341 — **ASR PASS**
