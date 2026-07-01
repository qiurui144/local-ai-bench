# AMD Linux Platform - Full Model Benchmark Report

**Platform:** amd-linux-x86 | AMD Ryzen 7 8845H + Radeon 780M iGPU, Ubuntu Linux
**Target host:** configured by `AMD_LINUX_HOST`; private lab address intentionally not published
**Last calibrated:** 2026-06-28. This file excludes the Intel/NVIDIA development host from AMD results.

## Software Stack And Accelerator Evidence

| Layer | Status | Evidence |
|---|---|---|
| Repository | DEPLOYED | `/home/qiurui/vlm-llm-benchmark` |
| Python | DEPLOYED | `python3.14` venv at `/home/qiurui/vlm-llm-benchmark/.venv` |
| Ollama | DEPLOYED | `0.22.1`, local bind `127.0.0.1:11434` |
| Radeon 780M GPU | USED by Ollama models | `ollama ps` reported `100% GPU` for `llama3.2:3b`, `qwen3:0.6b`, `qwen3-embedding:0.6b`, `qwen2.5:7b`, `bge-m3`, `llava:7b`, and `minicpm-v:8b` when loaded |
| ONNX Runtime | CPU only | providers = `['AzureExecutionProvider', 'CPUExecutionProvider']`; no `DmlExecutionProvider`, `VitisAIExecutionProvider`, or ROCm EP |
| NPU / XDNA | NOT USED on Linux | AMD IPU device exists in PCI inventory, but no RyzenAI/VitisAI runtime is exposed to this Python environment |
| DirectML | NOT AVAILABLE on Linux | DirectML OCR probe returned blocked because `DmlExecutionProvider` is absent |

Important distinction from Windows AMD: Windows has measured DirectML and VitisAI OCR paths. This Linux host currently matches the Windows test dimensions by running the same probes and recording explicit blocked states where the Linux software stack cannot invoke that accelerator.

## Coverage Matrix

| Category | Model entry | Runtime path | 3-seed status | Key result |
|---|---|---|---|---|
| LLM primary | `qwen2.5-7b-amd-linux` | Ollama / Radeon 780M GPU | PARTIAL PASS | 14.9 TPS; TTFT p50/p95 140/292 ms; translation WARN; GA BLOCKED |
| LLM lightweight | `qwen2.5-3b-amd-linux` | Ollama / Radeon 780M GPU | FAIL | 31.0 TPS; TTFT p50/p95 101/171 ms; translation FAIL on en->zh terminology; GA BLOCKED |
| LLM supplemental | `llama3.2-3b-amd-linux` | Ollama / Radeon 780M GPU | PERF PASS | 31.8 TPS; TG 32.9 tok/s; quality dims intentionally skipped in extra run |
| LLM micro | `qwen3-0.6b-amd-linux` | Ollama / Radeon 780M GPU | PERF PASS | 102.2 TPS; TTFT p50/p95 2446/4183 ms; TG 278.9 tok/s |
| Embedding | `bge-m3-amd-linux` | Ollama / Radeon 780M GPU | FAIL | mean hit@1/MRR/nDCG@10 0.944; seed0 recall@10 failed |
| Embedding supplemental | `qwen3-embedding-0.6b-amd-linux` | Ollama / Radeon 780M GPU | PASS | hit@1/MRR/nDCG@10 1.0; p50 latency 83.9 ms; dim 1024 |
| Reranker | `qwen2.5-3b-reranker-amd-linux` | Ollama generative proxy / GPU | PASS | nDCG@10 0.9866; MRR 1.0; pair p50 215 ms |
| Reranker supplemental | `qwen2.5-7b-reranker-amd-linux` | Ollama generative proxy / GPU | PASS | nDCG@10 1.0; MRR 1.0; pair p50 mean 333.7 ms |
| ASR | `sensevoice-small-amd-linux` | sherpa-onnx CPU | PASS | CER 7.69%; RTF 0.069 |
| ASR int8 alias | `sensevoice-small-int8-amd-linux` | sherpa-onnx CPU | PASS | CER 7.69%; RTF mean 0.0245; p50 latency mean 136.9 ms |
| OCR CPU | `rapidocr-amd-linux` | RapidOCR ONNX CPU | FAIL | CER 17.37% > 10%; p50/p95 490/739 ms |
| OCR DirectML probe | `rapidocr-amd-linux-directml` | ONNX DirectML | BLOCKED | `DmlExecutionProvider` not available on Linux |
| OCR NPU probe | `rapidocr-amd-linux-npu` | VitisAI / XDNA | BLOCKED | VitisAI/RyzenAI OCR helper cannot initialize; no VitisAI EP in ORT |
| OCR Paddle probe | `paddleocr-amd-linux` | PaddleOCR CPU | BLOCKED | `paddleocr` is not installed in remote venv |
| VLM baseline | `llava-7b-amd-linux` | Ollama / Radeon 780M GPU | FAIL | 14.5 TPS; TTFT p50 2111 ms; seed0 image errors caused FAIL |
| VLM supplemental | `minicpm-v-8b-amd-linux` | Ollama / Radeon 780M GPU | FAIL | 14.35 TPS; TTFT p50 2246 ms; category precision 77.8%, entity recall mean 55.6% |

## Notes On Test Parity With Windows

| Dimension | Windows AMD condition | Linux AMD result |
|---|---|---|
| LLM | Ollama GPU path with translation/perf/GA gates | Qwen2.5 7B/3B full dimensions completed; added llama3.2 3B and qwen3 0.6B performance runs |
| Embedding | GPU-backed embedding model coverage | `bge-m3` plus `qwen3-embedding:0.6b`; Qwen3 embedding passed 3-seed |
| Reranker | CPU/DirectML cross-encoder and proxy coverage | Linux has GPU-backed generative proxy only; both 3B and 7B proxy entries tested |
| ASR | SenseVoice ONNX, Windows can use DirectML path | Linux SenseVoice runs CPU ONNX only; base and int8 alias both pass |
| OCR CPU | RapidOCR/PaddleOCR CPU baselines | RapidOCR CPU ran and failed quality gate; PaddleOCR probe blocked because package is absent |
| OCR iGPU | Windows DirectML path | Linux DirectML probe blocked; DirectML EP absent |
| OCR NPU | Windows VitisAI XDNA path | Linux NPU probe blocked; no VitisAI EP/runtime exposed |
| VLM | Ollama GPU path | `llava:7b` and `minicpm-v:8b` both ran on GPU offload; both fail current accuracy gates |

## Artifacts

Raw benchmark outputs copied from the AMD Linux host:

| Group | Local report files |
|---|---|
| Original full run | `output/reports/amd-linux-x86/*_20260628_110330.*`, `*_111400.*`, `*_112238.*`, `*_152146.*`, `*_152206.*`, `*_152226.*`, `*_152232.*` |
| Extra model run | `output/reports/amd-linux-x86/*_20260628_161222.*` through `*_20260628_161714.*` |
| Logs | `output/logs/amd-linux-x86/*_amd_linux_full_20260628_*.log`, `*_amd_linux_extra_20260628_161222.log` |

## Conclusions

1. AMD Linux now has expanded model coverage in every requested category: embedding, reranker, ASR, OCR, LLM, and VLM.
2. GPU invocation is confirmed for Ollama-backed LLM, embedding, reranker-proxy, and VLM models through `ollama ps` reporting `100% GPU`.
3. NPU invocation is not available on this Linux software stack. The VitisAI/NPU and DirectML OCR entries were tested and recorded as blocked, not silently skipped.
4. Recommended Linux candidates from this run: `qwen2.5-7b-amd-linux` for LLM follow-up, `qwen3-embedding-0.6b-amd-linux` for embedding, `qwen2.5-7b-reranker-amd-linux` or `qwen2.5-3b-reranker-amd-linux` for proxy rerank coverage, and `sensevoice-small-int8-amd-linux` for ASR.
5. Not qualified yet: Linux OCR quality, Linux OCR DirectML/NPU paths, and both VLM candidates under the current accuracy gates.
