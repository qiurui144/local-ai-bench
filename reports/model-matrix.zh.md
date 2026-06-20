> [English](./model-matrix.en.md)

# 全模型矩阵 — 评测结果与最佳选型

> 最近校准：2026-06-20。本文件原地更新。RK3588 节 2026-06-20 修订：区分 RK1822 PCIe NPU 服务与 RK3588 RKNPU3 服务。



## 选型结论

- AMD Windows 默认 LLM：`qwen2.5-7b-amd-win`，质量/吞吐均衡；高参数上限用 `qwen2.5-14b-amd-win`，但吞吐约 7.67 tok/s。
- Intel Windows 默认 LLM：`qwen2.5-7b-intel-win` 用于更高质量，`qwen2.5-3b-intel-win` 用于轻量默认回归。
- 高并发/长上下文轻量 LLM：AMD `llama3.2-3b-amd-win`，Intel `llama3.2-1b-intel-win`，两者均验证 32k 上下文和 32 并发稳定。
- Embedding：AMD 首选 `qwen3-embedding-0.6b-amd`，Intel 首选 `qwen3-embedding-0.6b-intel-win`；AMD `bge-m3-amd` 可作多语言替代。
- Reranker：两端默认 `bge-reranker-base-*-win`；`bge-reranker-v2-m3-*-win` 质量同过但 CPU 延迟约 3.7 倍。
- OCR：AMD 首选 `rapidocr-amd-directml`；Intel 首选 `rapidocr-intel-openvino`；Intel DirectML OCR 不可用。
- ASR：两端均使用 `sensevoice-small-*-win`，均已 PASS。
- VLM：`llava:7b` 路径可运行但质量 fixture FAIL，不建议作为当前最佳 VLM。

## Matrix

| Model | Target | Provider | Role | Caps | Status | Verdicts | Key metrics | Latest report |
|---|---|---|---|---|---|---|---|---|
| qwen3-vl-8b-instruct | local/reference |  | vlm_primary | - | REGISTERED | - | - | - |
| qwen2.5-vl-7b-fp16 | local/reference |  | vlm_baseline | - | REGISTERED | - | - | - |
| qwen3-30b-a3b-instruct-2507-fp8 | local/reference |  | llm_primary | translation | REGISTERED | - | - | - |
| qwen3-235b-a22b-instruct-2507-fp8 | local/reference |  | llm_ultra_flagship | translation | REGISTERED | - | - | - |
| qwen3-embedding-0.6b | local/reference |  | embedding_primary | embedding | REGISTERED | - | - | - |
| qwen3-embedding-4b | local/reference |  | embedding_high_accuracy | embedding | REGISTERED | - | - | - |
| qwen3-reranker-4b | local/reference |  | reranker_primary | rerank | REGISTERED | - | - | - |
| bge-reranker-v2-m3 | local/reference |  | reranker_realtime | rerank | REGISTERED | - | - | - |
| bge-reranker-base | local/reference |  | reranker_realtime_min | rerank | REGISTERED | - | - | - |
| rapidocr-cpu | local/reference | local_onnx | ocr_cpu_baseline | ocr | PASS | ocr:PASS, structured_ocr:PASS | ocr CER 7.04%, NED 6.18%, p50 1592.5 ms; structured OCR field 92.86%, CER 7.04%, p50 859.0 ms | output/reports/amd-win-x86/reports/structured_ocr_amd_rapidocr_cpu_20260619.json |
| paddleocr-cpu | local/reference | local_onnx | ocr_cpu_paddle | ocr | PASS | ocr:PASS | ocr CER 7.04%, NED 6.18%, p50 1829.5 ms | output/reports/paddleocr-cpu_20260617_093241_seed2.json |
| rapidocr-amd-npu | amd-win-x86 | local_onnx | ocr_npu | ocr | PASS | ocr:PASS, structured_ocr:PASS | ocr CER 7.04%, NED 6.18%, p50 2031.0 ms; structured OCR field 92.86%, CER 7.04%, p50 1867.5 ms | output/reports/amd-win-x86/reports/structured_ocr_amd_vitisai_20260619.json |
| rapidocr-amd-directml | amd-win-x86 | local_onnx | ocr_gpu_directml | ocr | PASS | ocr:PASS, structured_ocr:PASS | ocr CER 7.04%, NED 6.18%, p50 468.5 ms; structured OCR field 92.86%, CER 7.04%, p50 476.5 ms | output/reports/amd-win-x86/reports/structured_ocr_amd_directml_20260619.json |
| rapidocr-intel-directml | intel-win-x86 | local_onnx | ocr_gpu_directml | ocr | FAIL | ocr:FAIL, structured_ocr:FAIL | ocr CER 202.35%, NED 97.77%, p50 945.5 ms; structured OCR field 0.00%, CER 207.51%, p50 984.5 ms | output/reports/intel-win-x86/reports/structured_ocr_intel_directml_20260619.json |
| rapidocr-intel-openvino | intel-win-x86 | local_onnx | ocr_openvino_probe | ocr | PASS | ocr:PASS, structured_ocr:PASS | ocr CER 7.04%, NED 6.18%, p50 797.0 ms; structured OCR field 92.86%, CER 7.04%, p50 867.5 ms | output/reports/intel-win-x86/reports/structured_ocr_intel_openvino_20260619.json |
| sensevoice-small | local/reference |  | asr_primary | asr | FAIL | asr:FAIL | asr CER 23.08%, RTF 0.098 | output/reports/sensevoice-small_20260616_224728_seed1.json |
| sensevoice-small-amd-win | amd-win-x86 | local_onnx | asr_amd_win | asr | PASS | asr:PASS | asr CER 7.69%, RTF 0.073 | output/reports/amd-win-x86/reports/sensevoice-small-amd-win_20260618_172918.json |
| sensevoice-small-intel-win | intel-win-x86 | local_onnx | asr_intel_win | asr | PASS | asr:PASS | asr CER 7.69%, RTF 0.341 | output/reports/intel-win-x86/reports/sensevoice-small-intel-win_20260618_172951.json |
| llama3.2-3b-amd-win | amd-win-x86 | ollama | llm_amd_baseline | translation | FAIL | conditioned:BLOCKED, conversation_drift:FAIL, general_ability:FAIL, model_limits:MEASURED, scenarios:SKIPPED, translation:FAIL | PP/TG 123.64/39.08 tok/s; TPS 28.99; TTFT p50/p95 890.0/5207.3 ms; concurrency c50 36.21 tok/s; conditioned BLOCKED; conversation_drift FAIL; general_ability FAIL; limit concurrency c16 37.88 tok/s; max context 32k; scenarios SKIPPED; translation FAIL | output/reports/amd-win-x86/reports/ollama_model_limits_amd_llama32_3b.json |
| qwen2.5-7b-amd-win | amd-win-x86 | ollama | llm_amd_primary | translation | FAIL | conditioned:FAIL, conversation_drift:FAIL, general_ability:PASS, scenarios:FAIL, translation:FAIL | PP/TG 116.05/16.07 tok/s; TPS 13.33; TTFT p50/p95 953.0/6240.5 ms; concurrency c8 16.70 tok/s; conditioned FAIL; conversation_drift FAIL; general_ability PASS; scenarios FAIL; stability drift 1.00; translation FAIL | output/reports/amd-win-x86/reports/qwen2.5-7b-amd-win_scenarios-20260619_20260619_002810.json |
| llava-7b-amd-win | amd-win-x86 | ollama | vlm_amd_baseline | vlm | FAIL | accuracy:FAIL, conversation_drift:FAIL, general_ability:FAIL | PP/TG 835.02/18.88 tok/s; TPS 16.84; TTFT p50/p95 890.0/890.9 ms; conversation_drift FAIL; general_ability FAIL | output/reports/amd-win-x86/reports/llava-7b-amd-win_20260618_133905.json |
| qwen3-embedding-0.6b-amd | amd-win-x86 | ollama | embedding_primary | embedding | PASS | embedding:PASS | embed hit@1 1.000, nDCG 1.000, p50 875.0 ms | output/reports/amd-win-x86/reports/qwen3-embedding-0.6b-amd_20260618_222139.json |
| bge-m3-amd | amd-win-x86 | ollama | embedding_bge | embedding | PASS | embedding:PASS | embed hit@1 1.000, nDCG 1.000, p50 914.0 ms | output/reports/amd-win-x86/reports/bge-m3-amd_20260618_222219.json |
| qwen3-0.6b-amd | amd-win-x86 | ollama | llm_nano | translation | FAIL | conversation_drift:FAIL, general_ability:FAIL | PP/TG 0.00/0.00 tok/s; TPS 91.09; TTFT p50/p95 1781.0/1781.0 ms; conversation_drift FAIL; general_ability FAIL | output/reports/amd-win-x86/reports/qwen3-0.6b-amd_20260618_134324.json |
| bge-reranker-base-amd-win | amd-win-x86 | local_reranker | reranker_amd_win_cross_encoder | rerank | PASS | rerank:PASS | rerank nDCG 1.000, MRR 1.000, p50 78.0 ms | output/reports/amd-win-x86/reports/bge-reranker-base-amd-win_20260619_191441.json |
| bge-reranker-v2-m3-amd-win | amd-win-x86 | local_reranker | reranker_amd_win_cross_encoder_stronger | rerank | PASS | rerank:PASS | rerank nDCG 1.000, MRR 1.000, p50 289.0 ms | output/reports/amd-win-x86/reports/bge-reranker-v2-m3-amd-win_20260619_191544.json |
| qwen2.5-14b-amd-win | amd-win-x86 | ollama | llm_amd_win_parameter_uplift | translation | MEASURED | model_limits:MEASURED | PP/TG 94.25/9.14 tok/s; TPS 7.67; TTFT p50/p95 8273.5/14791.7 ms; limit concurrency c8 8.95 tok/s; max context 16k | output/reports/amd-win-x86/reports/ollama_model_limits_amd_qwen25_14b.json |
| qwen3-vl-2b-rk3588 | rk3588-linux | generic | llm_npu_primary | - | PASS | translation:PASS | TTFT p50/p95 144.6/167.8 ms; TPS 109 t/s; translation BLEU 19.8, chrF 35.5 | reports/rk3588.en.md |
| minicpm-embed-rk1822 | rk182x-linux | generic | embedding_npu | embedding | PASS | embedding:PASS | hit@1 1.000, nDCG 1.000, MRR 1.000, p50 143 ms | reports/rk3588.en.md |
| qwen2.5-0.5b-rk3588 | rk3588-linux | generic |  | - | REGISTERED | - | - | - |
| minicpm-v-rk3588 | rk3588-linux | generic |  | - | REGISTERED | - | - | - |
| llama3.2-1b-intel-win | intel-win-x86 | ollama | llm_intel_win_nano | translation | FAIL | conversation_drift:FAIL, general_ability:BLOCKED, model_limits:MEASURED, translation:FAIL | PP/TG 130.29/34.78 tok/s; TPS 25.26; TTFT p50/p95 875.0/3307.7 ms; conversation_drift FAIL; general_ability BLOCKED; limit concurrency c32 32.52 tok/s; max context 32k; translation FAIL | output/reports/intel-win-x86/reports/ollama_model_limits_intel_llama32_1b.json |
| qwen2.5-3b-intel-win | intel-win-x86 | ollama | llm_intel_win_baseline | translation | FAIL | conditioned:BLOCKED, conversation_drift:WARN, general_ability:BLOCKED, scenarios:FAIL, translation:FAIL | PP/TG 124.42/26.21 tok/s; TPS 19.47; TTFT p50/p95 781.0/3495.4 ms; concurrency c8 24.68 tok/s; conditioned BLOCKED; conversation_drift WARN; general_ability BLOCKED; scenarios FAIL; stability drift 1.00; translation FAIL | output/reports/intel-win-x86/reports/qwen2.5-3b-intel-win_scenarios-20260619_20260619_002817.json |
| qwen2.5-7b-intel-win | intel-win-x86 | ollama | llm_intel_win_parameter_uplift | translation | MEASURED | model_limits:MEASURED | PP/TG 112.20/9.10 tok/s; TPS 8.25; TTFT p50/p95 4820.0/8440.7 ms; limit concurrency c16 9.54 tok/s; max context 16k | output/reports/intel-win-x86/reports/ollama_model_limits_intel_qwen25_7b.json |
| qwen3-embedding-0.6b-intel-win | intel-win-x86 | ollama | embedding_intel_win | embedding | PASS | embedding:PASS | embed hit@1 1.000, nDCG 1.000, p50 617.5 ms | output/reports/intel-win-x86/reports/qwen3-embedding-0.6b-intel-win_20260618_221933.json |
| llava-7b-intel-win | intel-win-x86 | ollama | vlm_intel_win_baseline | vlm | FAIL | accuracy:FAIL | PP/TG 1073.89/10.56 tok/s; TPS 10.02; TTFT p50/p95 703.0/703.0 ms | output/reports/intel-win-x86/reports/llava-7b-intel-win_20260618_174001.json |
| bge-reranker-base-intel-win | intel-win-x86 | local_reranker | reranker_intel_win_cross_encoder | rerank | PASS | rerank:PASS | rerank nDCG 1.000, MRR 1.000, p50 148.5 ms | output/reports/intel-win-x86/reports/bge-reranker-base-intel-win_20260619_191441.json |
| bge-reranker-v2-m3-intel-win | intel-win-x86 | local_reranker | reranker_intel_win_cross_encoder_stronger | rerank | PASS | rerank:PASS | rerank nDCG 1.000, MRR 1.000, p50 546.5 ms | output/reports/intel-win-x86/reports/bge-reranker-v2-m3-intel-win_20260619_191544.json |
| llama3.2-3b-intel-linux | intel-linux | ollama | llm_intel_linux_baseline | translation | REGISTERED | - | - | - |
| qwen2.5-7b-intel-linux | intel-linux | ollama | llm_intel_linux_primary | translation | REGISTERED | - | - | - |
| qwen3-embedding-0.6b-intel-linux | intel-linux | ollama | embedding_intel_linux | embedding | REGISTERED | - | - | - |
| qwen2.5-0.5b-k3-riscv | k3-riscv | generic |  | - | REGISTERED | - | - | - |
| llama3.2-1b-k3-riscv | k3-riscv | generic |  | - | REGISTERED | - | - | - |

## 证据规则

- 表格基于 `models.yaml` 全量注册模型生成。
- 每个模型聚合 `output/reports/**.json` 中可解析的最新/专项证据。
- `REGISTERED` 表示模型已注册但当前工作区没有可复核 JSON 结果。
- 旧生成式 reranker 代理结果不作为当前 reranker 通过依据；当前采用 `local_reranker` CrossEncoder。
