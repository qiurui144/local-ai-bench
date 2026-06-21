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

## Intel NPU / OpenVINO LLM Path (Not Yet Tested)

Intel Core Ultra processors include an Intel NPU (AI Boost) accessible via OpenVINO.
OpenVINO has also landed in llama.cpp as a first-class compute backend (OpenVINO 2026.0.0),
enabling LLM inference on Intel CPU / iGPU / NPU via the same GGUF model files used by Ollama.

### OpenVINO llama.cpp Integration (Reference — not benchmarked on this platform)

OpenVINO became a supported backend in llama.cpp, replacing the older SYCL approach:

**Build:**
```bash
cmake -B build -DGGML_OPENVINO=ON
cmake --build build --config Release -j $(nproc)
```

**Runtime device selection via env:**
```bash
# Use Intel iGPU (Arc/Xe)
export GGML_OPENVINO_DEVICE=GPU
./build/bin/llama-cli -m model.gguf -p "Hello" -n 128

# Use Intel NPU (AI Boost)
export GGML_OPENVINO_DEVICE=NPU
./build/bin/llama-cli -m model.gguf -p "Hello" -n 128

# Use Intel CPU via OpenVINO (optimized path)
export GGML_OPENVINO_DEVICE=CPU
./build/bin/llama-cli -m model.gguf -p "Hello" -n 128
```

**Supported quantization formats:** FP16, Q8_0, Q4_0, Q4_1, Q4_K, Q4_K_M

**OpenVINO version required:** 2026.0.0 or later (earlier versions may lack NPU support)

This gives a LLM acceleration path beyond CPU-only Ollama — the iGPU and NPU are reachable
via llama.cpp without changing the GGUF model format.

### OCR/ASR: NPU path via OpenVINO ONNX

```bash
# OpenVINO NPU for OCR ONNX models
export OV_DEVICE=NPU
python run_benchmark.py --model rapidocr-intel-openvino --target intel-win-x86
```

`GGML_OPENVINO_DEVICE=NPU` (llama.cpp) and `OV_DEVICE=NPU` (ONNX Runtime) use the same
OpenVINO NPU plugin — both routes are available once OpenVINO 2026.0.0 is installed.

**Current status:** LLM via llama.cpp+OpenVINO and NPU OCR are **not yet benchmarked** on this
specific Intel Windows test device. The paths above are the expected configuration.

---

## 中文摘要

**硬件：** Intel Core Ultra 集成显卡 + Intel AI Boost NPU，OpenVINO + DirectML  
**最后校准：** 2026-06-19

### iGPU 路径覆盖范围

| 任务 | 路径 | 状态 |
|---|---|---|
| OCR | OpenVINO | **PASS**（p50 797 ms） |
| OCR | DirectML | **FAIL**（CER 202%，不可用） |
| ASR | DirectML | **PASS**（RTF 0.341） |
| LLM（CPU） | Ollama CPU | 见 cpu 文档 |
| **LLM（iGPU/NPU via llama.cpp+OpenVINO）** | `GGML_OPENVINO_DEVICE=GPU\|NPU` | **未测试，路径已记录** |

### OpenVINO llama.cpp LLM 路径（未测试，已记录配置）

OpenVINO 2026.0.0 已集成进 llama.cpp，通过 `-DGGML_OPENVINO=ON` 编译后可选择 Intel CPU/iGPU/NPU 执行 GGUF 模型推理。设备选择：`export GGML_OPENVINO_DEVICE=GPU`（iGPU）或 `=NPU`（AI Boost）。支持 FP16/Q8_0/Q4_0/Q4_1/Q4_K/Q4_K_M 量化格式。

### 关键数据

- `rapidocr-intel-openvino`：CER 7.04%，p50 797 ms，结构化 p50 868 ms — **推荐 OCR 路径**
- `rapidocr-intel-directml`：CER 202.35% — **不可用，驱动精度问题**
- `sensevoice-small-intel-win`：CER 7.69%，RTF 0.341 — **ASR PASS**
- Intel iGPU LLM（llama.cpp+OpenVINO）：未测试，待校准
