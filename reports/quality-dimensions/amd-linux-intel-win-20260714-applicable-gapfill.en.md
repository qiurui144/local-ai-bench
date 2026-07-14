# Quality Dimension Coverage Report

- run_id: `amd-linux-intel-win-20260714-applicable-gapfill`
- generated_at: `2026-07-14T11:33:39.916681+08:00`
- overall_status: `complete_with_quality_failures`
- quality_coverage_status: `complete`
- family_coverage_status: `complete`
- quality_dimensions: `accuracy, translation, embedding, rerank, asr, ocr, general_ability, conditioned, long_context, scenarios, conversation_drift`

## Family Coverage

| target | family | status | feasible_models | measured_models | primary_dim |
|---|---|---:|---:|---|---|
| amd-linux-x86 | llm | covered | 5 | phi-3.5-mini-amd-linux, qwen2.5-1.5b-amd-linux, qwen2.5-3b-amd-linux, qwen2.5-7b-amd-linux | translation |
| amd-linux-x86 | vlm | covered | 2 | llava-7b-amd-linux, minicpm-v-8b-amd-linux | accuracy |
| amd-linux-x86 | embedding | covered | 2 | bge-m3-amd-linux, qwen3-embedding-0.6b-amd-linux | embedding |
| amd-linux-x86 | reranker | covered | 2 | qwen2.5-3b-reranker-amd-linux, qwen2.5-7b-reranker-amd-linux | rerank |
| amd-linux-x86 | ocr | covered | 1 | rapidocr-amd-linux | ocr |
| amd-linux-x86 | asr | covered | 3 | sensevoice-small-amd-linux, sensevoice-small-int8-amd-linux, whisper-tiny-amd-linux-npu | asr |
| intel-win-x86 | llm | covered | 3 | qwen2.5-1.5b-igpu-intel-win, qwen2.5-1.5b-intel-win | translation |
| intel-win-x86 | vlm | covered | 1 | llava-7b-intel-win | accuracy |
| intel-win-x86 | embedding | covered | 3 | bge-base-en-v1.5-igpu-intel-win, bge-m3-intel-win, qwen3-embedding-0.6b-intel-win | embedding |
| intel-win-x86 | reranker | covered | 3 | bge-reranker-base-igpu-intel-win, bge-reranker-base-intel-win, bge-reranker-v2-m3-intel-win | rerank |
| intel-win-x86 | ocr | covered | 2 | paddleocr-openvino-intel-win, rapidocr-intel-openvino | ocr |
| intel-win-x86 | asr | covered | 3 | sensevoice-small-intel-win, whisper-base-npu-intel-win, whisper-tiny-openvino-intel-win | asr |

## Model Quality Matrix

| target | model | family | eligibility | complete | required_dims | statuses |
|---|---|---|---|---:|---|---|
| intel-win-x86 | `bge-base-en-v1.5-igpu-intel-win` | embedding | feasible | true | embedding | embedding:failed |
| amd-linux-x86 | `bge-m3-amd-linux` | embedding | feasible | true | embedding | embedding:passed |
| intel-win-x86 | `bge-m3-intel-win` | embedding | feasible | true | embedding | embedding:passed |
| intel-win-x86 | `bge-reranker-base-igpu-intel-win` | reranker | feasible | true | rerank | rerank:failed |
| intel-win-x86 | `bge-reranker-base-intel-win` | reranker | feasible | true | rerank | rerank:passed |
| intel-win-x86 | `bge-reranker-v2-m3-intel-win` | reranker | feasible | true | rerank | rerank:passed |
| amd-linux-x86 | `llama3.2-1b-amd-linux` | llm | feasible | true | accuracy, long_context | accuracy:measured, long_context:failed |
| intel-win-x86 | `llama3.2-1b-intel-win` | llm | feasible | true | accuracy | accuracy:measured |
| amd-linux-x86 | `llama3.2-3b-amd-linux` | llm | runtime_blocked | false | accuracy, translation, general_ability, long_context | accuracy:missing, translation:missing, general_ability:missing, long_context:missing |
| amd-linux-x86 | `llava-7b-amd-linux` | vlm | feasible | true | accuracy | accuracy:failed |
| intel-win-x86 | `llava-7b-intel-win` | vlm | feasible | true | accuracy | accuracy:failed |
| amd-linux-x86 | `minicpm-v-8b-amd-linux` | vlm | feasible | true | accuracy, long_context | accuracy:failed, long_context:failed |
| amd-linux-x86 | `paddleocr-amd-linux` | ocr | platform_blocked | false | ocr | ocr:skipped |
| intel-win-x86 | `paddleocr-openvino-intel-win` | ocr | feasible | true | ocr | ocr:passed |
| amd-linux-x86 | `phi-3.5-mini-amd-linux` | llm | feasible | true | accuracy, translation, long_context | accuracy:measured, translation:failed, long_context:failed |
| intel-win-x86 | `phi-3.5-mini-intel-win` | llm | cpu_only_blocked | false | accuracy, translation, long_context | accuracy:missing, translation:missing, long_context:missing |
| amd-linux-x86 | `qwen2.5-0.5b-amd-linux` | llm | runtime_blocked | false | accuracy, translation, long_context | accuracy:missing, translation:missing, long_context:missing |
| amd-linux-x86 | `qwen2.5-0.5b-amd-linux-onnx` | llm | runtime_blocked | false | accuracy, translation, long_context | accuracy:missing, translation:missing, long_context:missing |
| intel-win-x86 | `qwen2.5-0.5b-intel-win` | llm | cpu_only_blocked | false | accuracy, translation | accuracy:missing, translation:missing |
| amd-linux-x86 | `qwen2.5-1.5b-amd-linux` | llm | feasible | true | accuracy, translation, long_context | accuracy:measured, translation:failed, long_context:warning |
| intel-win-x86 | `qwen2.5-1.5b-igpu-intel-win` | llm | feasible | true | accuracy, translation, long_context | accuracy:measured, translation:failed, long_context:warning |
| intel-win-x86 | `qwen2.5-1.5b-intel-win` | llm | feasible | true | accuracy, translation | accuracy:measured, translation:failed |
| amd-linux-x86 | `qwen2.5-3b-amd-linux` | llm | feasible | true | accuracy, translation, general_ability, long_context | accuracy:measured, translation:failed, general_ability:failed, long_context:failed |
| intel-win-x86 | `qwen2.5-3b-intel-win` | llm | cpu_only_blocked | false | accuracy, translation, general_ability, long_context | accuracy:missing, translation:missing, general_ability:missing, long_context:missing |
| amd-linux-x86 | `qwen2.5-3b-reranker-amd-linux` | reranker | feasible | true | rerank | rerank:passed |
| amd-linux-x86 | `qwen2.5-7b-amd-linux` | llm | feasible | true | accuracy, translation, general_ability, long_context | accuracy:measured, translation:failed, general_ability:passed, long_context:warning |
| intel-win-x86 | `qwen2.5-7b-igpu-intel-win` | llm | runtime_blocked | false | accuracy, translation, general_ability, long_context | accuracy:missing, translation:missing, general_ability:missing, long_context:missing |
| intel-win-x86 | `qwen2.5-7b-intel-win` | llm | cpu_only_blocked | false | accuracy, translation, general_ability, long_context | accuracy:missing, translation:missing, general_ability:missing, long_context:missing |
| amd-linux-x86 | `qwen2.5-7b-reranker-amd-linux` | reranker | feasible | true | rerank | rerank:passed |
| intel-win-x86 | `qwen2.5-coder-0.5b-igpu-intel-win` | llm | runtime_blocked | false | accuracy, translation, long_context | accuracy:missing, translation:missing, long_context:missing |
| amd-linux-x86 | `qwen3-0.6b-amd-linux` | llm | runtime_blocked | false | accuracy, translation, general_ability, long_context | accuracy:missing, translation:missing, general_ability:missing, long_context:missing |
| intel-win-x86 | `qwen3-0.6b-igpu-intel-win` | llm | runtime_blocked | false | accuracy, translation, long_context | accuracy:missing, translation:missing, long_context:missing |
| intel-win-x86 | `qwen3-0.6b-intel-win` | llm | cpu_only_blocked | false | accuracy, general_ability, long_context | accuracy:missing, general_ability:missing, long_context:missing |
| amd-linux-x86 | `qwen3-1.7b-amd-linux` | llm | runtime_blocked | false | accuracy, translation, general_ability, long_context | accuracy:missing, translation:missing, general_ability:missing, long_context:missing |
| intel-win-x86 | `qwen3-1.7b-igpu-intel-win` | llm | runtime_blocked | false | accuracy, translation, long_context | accuracy:missing, translation:missing, long_context:missing |
| intel-win-x86 | `qwen3-1.7b-intel-win` | llm | cpu_only_blocked | false | accuracy, general_ability, long_context | accuracy:missing, general_ability:missing, long_context:missing |
| amd-linux-x86 | `qwen3-4b-amd-linux` | llm | runtime_blocked | false | accuracy, translation, long_context | accuracy:missing, translation:missing, long_context:missing |
| intel-win-x86 | `qwen3-4b-igpu-intel-win` | llm | runtime_blocked | false | accuracy, translation, general_ability, long_context | accuracy:missing, translation:missing, general_ability:missing, long_context:missing |
| intel-win-x86 | `qwen3-4b-intel-win` | llm | cpu_only_blocked | false | accuracy, translation, general_ability, long_context | accuracy:missing, translation:missing, general_ability:missing, long_context:missing |
| amd-linux-x86 | `qwen3-embedding-0.6b-amd-linux` | embedding | feasible | true | embedding | embedding:passed |
| intel-win-x86 | `qwen3-embedding-0.6b-intel-win` | embedding | feasible | true | embedding | embedding:passed |
| amd-linux-x86 | `rapidocr-amd-linux` | ocr | feasible | true | ocr | ocr:failed |
| amd-linux-x86 | `rapidocr-amd-linux-directml` | ocr | platform_blocked | false | ocr | ocr:skipped |
| amd-linux-x86 | `rapidocr-amd-linux-npu` | ocr | platform_blocked | false | ocr | ocr:skipped |
| intel-win-x86 | `rapidocr-intel-directml` | ocr | platform_blocked | false | ocr | ocr:skipped |
| intel-win-x86 | `rapidocr-intel-openvino` | ocr | feasible | true | ocr | ocr:warning |
| amd-linux-x86 | `sensevoice-small-amd-linux` | asr | feasible | true | asr | asr:passed |
| amd-linux-x86 | `sensevoice-small-int8-amd-linux` | asr | feasible | true | asr | asr:passed |
| intel-win-x86 | `sensevoice-small-intel-win` | asr | feasible | true | asr | asr:passed |
| intel-win-x86 | `whisper-base-npu-intel-win` | asr | feasible | true | asr | asr:failed |
| amd-linux-x86 | `whisper-tiny-amd-linux-npu` | asr | feasible | true | asr | asr:passed |
| intel-win-x86 | `whisper-tiny-openvino-intel-win` | asr | feasible | true | asr | asr:failed |

## Gaps

- feasible_model_dimension_gaps: `0`
- family_coverage_gaps: `0`
- runtime_or_cpu_blocked_models: `22`

A `failed` quality dimension is still measured evidence. `blocked`, `skipped`, and `missing` are coverage gaps for feasible models.
