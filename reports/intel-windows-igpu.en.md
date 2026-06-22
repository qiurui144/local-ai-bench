> [← Intel Windows overview](./intel-windows.en.md)

# Intel Windows — iGPU / OpenVINO / NPU Paths

**Hardware:** Intel Core Ultra 7 155H · Intel Arc iGPU (8 Xe-cores) · Intel AI Boost NPU  
**Software stack:** OpenVINO 2026.2.1 · optimum-intel 2.0.0 · onnxruntime-directml 1.24.4  
**Last calibrated:** 2026-06-22

---

## Version Constraints & Known Issues

| Package | Installed | Status |
|---|---|---|
| `openvino` | **2026.2.1** | OK — devices: CPU + GPU (Arc) + NPU ✓ |
| `openvino-genai` | 2026.2.1 (upgraded) | **BROKEN** — DLL load failed (system DLL conflict, not version mismatch) |
| `optimum-intel` | 2.0.0 | OK — `OVModelForCausalLM` / `OVModelForFeatureExtraction` device=GPU ✓ |
| `rapidocr-openvino` | 1.4.4 | OK — works with OV 2026.2.1 ✓ |

**`openvino-genai` DLL issue:** `ImportError: DLL load failed while importing py_openvino_genai: 找不到指定的程序`. Not a version mismatch — `os.add_dll_directory()` + upgrading to 2026.2.1.0 both fail. Root cause: system DLL export conflict. Workaround: use `optimum-intel OVModelForCausalLM` (slower but functional). **Upstream plan**: contribute rapidocr OV 2026 compatibility PR; revisit genai once resolved.

**Key insight (2026-06-22):** Intel Ollama runs **100% CPU** for all models (Intel Arc iGPU not supported in standard Ollama). AMD Ollama runs **100% GPU** (Radeon 780M). iGPU LLM on Intel requires OpenVINO path.

---

## Confirmed iGPU Paths (optimum-intel, device='GPU')

| Workload | Model | Cold load | Warm latency / TPS | Status |
|---|---|---|---|---|
| **LLM 7B** | `qwen2.5-7b-int4-ov` (OVModelForCausalLM) | **115 s** (GPU kernel compile) | **8.1 TPS** | ✓ CONFIRMED |
| **LLM 1.5B** | `qwen2.5-1.5b-int4-ov` (OVModelForCausalLM) | **54 s** (GPU kernel compile) | **10.6 TPS** | ✓ CONFIRMED |
| **Embedding INT8** | `bge-base-en-v1.5-int8-ov` (OVModelForFeatureExtraction) | ~1.7 s | **~25 ms warm** | ✓ CONFIRMED |
| **Reranker INT8** | `bge-reranker-base-int8-ov` (OVModelForSequenceClassification) | 4.9 s | **36.4 ms avg** | ✓ CONFIRMED |
| OCR (text/structured) | `rapidocr-openvino` (OpenVINO EP, auto-device) | — | p50 **797 ms** / 867 ms | ✓ CONFIRMED |

> Note: `OVModelForCausalLM` GPU is 3× slower than `openvino_genai.LLMPipeline` GPU (1.5B: 10.6 vs 34 TPS) due to missing KV-cache optimization in the transformers generation loop. LLMPipeline is the preferred path once DLL issue is resolved.

**Model files on machine** (`C:\ov_models\`):
```
qwen2.5-1.5b-int4-ov/    ← from OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov
qwen2.5-7b-int4-ov/       ← from OpenVINO/Qwen2.5-7B-Instruct-int4-ov
embedding/bge-base-en-v1.5-int8-ov/
reranker/bge-reranker-base-int8-ov/
asr/whisper-base-int8-ov/
```

**Still to download** (Qwen3 series, requires HF access):
```
qwen3-0.6b-int4-ov/    ← OpenVINO/Qwen3-0.6B-int4-ov
qwen3-4b-int4-ov/       ← OpenVINO/Qwen3-4B-int4-ov
qwen3-8b-int4-ov/       ← OpenVINO/Qwen3-8B-int4-ov (optional, large)
```

---

## NPU Results (Intel AI Boost, 11 TOPS INT8)

| Task | Model | Result | Latency | Status |
|---|---|---|---|---|
| **OCR det** | ch_PP-OCRv4_det (static [1,3,640,640]) | compile 4.6s; inference avg | **33 ms** | ✓ PASS |
| **OCR rec** | ch_PP-OCRv4_rec (static [1,3,**48**,320]; H=48 required) | compile 2.9s | **11 ms** | ✓ PASS |
| **OCR cls** | ch_PP-OCRv4_cls (static [1,3,48,192]) | compile 2.0s | **3 ms** | ✓ PASS |
| **ASR encoder** | whisper-base-int8-ov encoder (static [1,80,3000]) | compile ~15s cold / ~0.5s cached | **115 ms** | ✓ PASS |
| ASR decoder | whisper-base-int8-ov decoder | compile 1.0s | — | CPU (dynamic autoregressive) |
| Embedding | bge-base-en-v1.5-int8-ov | — | — | **FAIL** (dynamic shapes: upper bounds unspecified) |
| Reranker | bge-reranker-base-int8-ov | — | — | **FAIL** (dynamic shapes) |
| SenseVoice ASR | model.int8.onnx | — | — | **FAIL** (dynamic self-attn mask; needs re-export) |

**NPU OCR note:** rec model requires `H=48` (not the default H=32). This is due to an `AvgPool` kernel constraint in PP-OCRv4 rec. Static reshape to [1,3,48,320] is mandatory for NPU VPUX.

---

## OCR Paths Comparison

| Path | Backend | p50 OCR | p50 Structured | Status |
|---|---|---|---|---|
| **Intel OpenVINO** | OpenVINO EP (iGPU auto-select) | **797 ms** | **867 ms** | ✓ **PASS — recommended** |
| Intel DirectML | ONNX DirectML | 946 ms | 985 ms | **FAIL** — CER 202%, not usable |
| CPU reference | ONNX CPU | 1593 ms | 859 ms | PASS (reference only) |
| NPU (PP-OCRv4 det+rec+cls) | NPU VPUX static | 33+11+3 = **47 ms** | same | ✓ PASS (pipeline mode; production-ready) |

---

## ASR Paths Comparison

| Path | Backend | Latency | CER | Status |
|---|---|---|---|---|
| **SenseVoice** | DirectML (sherpa-onnx) | RTF 0.341 | 7.69% | ✓ **PASS — primary** |
| Whisper-base INT8 | iGPU OpenVINO GPU | 567 ms full (58s first-run compile) | — | ✓ PASS — alternative |
| Whisper-base encoder | NPU VPUX | 115 ms encoder | — | ✓ PASS encoder; decoder on CPU |

---

## LLM iGPU Path: Roadmap

| Step | Status | Action |
|---|---|---|
| OV model downloaded (1.5B, 7B) | ✓ Done | — |
| iGPU inference verified (optimum-intel) | ✓ Done | 7B: 8.1 TPS; 1.5B: 10.6 TPS |
| openvino_genai LLMPipeline | ❌ DLL broken | Diagnose DLL export conflict; or use OVMS |
| Qwen3 INT4 OV download | ⏳ Pending | `huggingface-cli download OpenVINO/Qwen3-4B-int4-ov` |
| HTTP serving layer for benchmark | ⏳ Pending | OVMS Docker (needs Docker install) or thin FastAPI wrapper around OVModelForCausalLM |
| models.yaml iGPU entries | ⏳ Pending | Add after serving layer confirmed |

---

## 中文摘要

**已确认可用（2026-06-22）：**
- iGPU 嵌入（BGE-base INT8）：`OVModelForFeatureExtraction` device=GPU → ~25ms warm ✓
- iGPU 重排序（BGE-reranker-base INT8）：device=GPU → 36.4ms avg ✓
- iGPU LLM 推理（optimum-intel）：Qwen2.5-7B → 8.1 TPS；1.5B → 10.6 TPS ✓（速度慢于 LLMPipeline，需 HTTP 包装层）
- OCR：OpenVINO EP → p50 797ms ✓；NPU PP-OCRv4 → det 33ms/rec 11ms/cls 3ms ✓
- ASR：SenseVoice DirectML RTF 0.341 ✓；Whisper NPU encoder 115ms ✓

**待解决：**
- `openvino_genai` DLL 冲突（系统级）→ 建议尝试 OVMS；上游 rapidocr PR 计划
- Qwen3 INT4 OV 模型下载（0.6B/4B/8B）→ `drivers/intel-win/ov_models/llm/`
- iGPU LLM 的 HTTP 服务层（benchmark harness 需要）
- 所有模型文件统一存放 `drivers/intel-win/ov_models/`（见 CLAUDE.md）
