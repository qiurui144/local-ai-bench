# SpacemiT Model Zoo Cache And VLM Runs

This project uses one local cache root for SpacemiT archive artifacts:

```text
drivers/spacemit-ai/model_zoo/
  llm/        GGUF LLM artifacts from archive.spacemit.com/spacemit-ai/model_zoo/llm
  vlm/        multimodal VLM tar/GGUF/mmproj artifacts from archive.spacemit.com/spacemit-ai/model_zoo/vlm
  vision/     official ONNX CV models from archive.spacemit.com/spacemit-ai/model_zoo/vision
  asr/        ASR packages from archive.spacemit.com/spacemit-ai/model_zoo/asr
  embed/      embedding artifacts
  rerank/     reranker artifacts
```

`vision/` and `vlm/` are different archive scopes, not separate cache roots.
The whole `drivers/` tree is a local cache and is ignored by git.

## Official References

- Invocation and application integration reference:
  [SpacemiT AI SDK](https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/application_tools/ai-sdk.md),
  doc tree update time `2026-06-30 20:04:26`.
- Performance baseline reference:
  [SpacemiT ModelZoo](https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/compute_stack/ai_compute_stack/modelzoo.md),
  doc tree update time `2026-06-09 18:06:37`.

The AI SDK page is treated as the application-facing call contract: component
demos, `llm_chat`, gateway HTTP/WS routes, and VLM model load/chat/unload
flows. The ModelZoo page is treated as the official performance baseline:
vision uses `onnxruntime_perf_test`, LLM uses `llama-bench -p 128 -n 128 -mmp
0 -fa 1 -ub 128`, and VLM/ASR performance rows use the SMT llama-server path.
Local K3 measurements in this repo are labeled separately from those official
baseline rows.

## Alignment And Retest Policy

Use this policy before promoting K3 numbers into platform conclusions:

- Throughput rows pass official alignment only when the same artifact,
  quantization, command, thread/core setting, and metric reach at least 95% of
  the official ModelZoo value.
- Latency and RTF rows pass official alignment only when the same artifact,
  command, input class, and metric are no worse than 105% of the official
  ModelZoo value.
- If the artifact or metric differs, mark the row `RETEST_REQUIRED` instead of
  comparing numbers. Neighboring quantizations such as Q4_K_M vs Q4_0 are not
  substitutes.

Current K3 32G alignment status:

| Area | Status | Reason |
|---|---|---|
| Vision ONNX | `ALIGNED` | 132/132 K3 ModelZoo rows have same-command local `onnxruntime_perf_test` results; none are >5% slower than official. |
| LLM | `ALIGNED` | 8/8 official rows passed the same-command `llama-bench -p 128 -n 128 -mmp 0 -fa 1 -ub 128` retest in `output/reports/k3-riscv-32g/official-modelzoo-llm-20260706_185656/alignment-summary.tsv`. The retest must leave `PRIVATE_ENV` empty so TCM is enabled and run `spacemit-tcm-smi -c` before each model. |
| VLM VisionEncoder | `ALIGNED_PROBE` | `scripts/run_k3_32g_official_modelzoo_vlm_encoder_probe.sh` measured all 10 official 4/8-core VisionEncoder rows within the 105% latency gate in `output/reports/k3-riscv-32g/official-modelzoo-vlm-encoder-20260706_193214/results.tsv`. This is an encoder ONNX probe, not an end-to-end VLM document latency claim. |
| qwen3-ASR 0.6B | `PARTIAL_RETEST_REQUIRED` | Broad rerun in `output/reports/k3-riscv-32g/nonllm-broad-20260706_190649/` measured RTF p50 0.168 and p95 0.512 with normalized CER avg 0.0192; the input mix and aggregation still differ from the official ModelZoo RTF row. |
| sensevoice | `RETEST_REQUIRED` | Official RTF exists, but current K3 result is package inspection only. |
| OCR / embedding / reranker | `LOCAL_ONLY` | The cited ModelZoo page does not publish official baseline rows for these tasks. Latest broad rerun: PP-OCRv5 line OCR CER 0.0039/p95 2985.5ms; BGE-Zh embedding Hit@1 0.9722/p95 5.85ms; BGE reranker Hit@1 1.0 through top50, with top20 p95 1333ms. |

Exact LLM official-baseline retest wrapper:

```bash
bash scripts/run_k3_32g_official_modelzoo_llm_retest.sh --describe
CACHE_ROOT=drivers/spacemit-ai/model_zoo \
  bash scripts/run_k3_32g_official_modelzoo_llm_retest.sh
```

Supply K3 connection values through the local secure environment before running
remote tests. Do not record host, account, or password values in docs, reports,
or run outputs.

The wrapper defaults to `PRIVATE_ENV=''` and `FORCE_TCM_RELEASE=1`. A control
run with `SPACEMIT_DISABLE_TCM=1` caused qwen3-0.6B to fall from roughly
PP128/TG128 `502/53` token/s to `437/37` token/s, so disabling TCM must not be
used for ModelZoo baseline claims.

## Download

Inspect the expected data source manifest without downloading:

```bash
bash scripts/cache_spacemit_model_zoo.sh --manifest
```

Download only missing or partial cache entries in parallel:

```bash
PARALLEL_DOWNLOADS=3 bash scripts/cache_spacemit_model_zoo.sh --download-missing
```

Cache all non-LLM artifacts used by the K3 32G broad retest:

```bash
SCOPE=nonllm CACHE_ROOT=drivers/spacemit-ai/model_zoo \
  bash scripts/cache_spacemit_model_zoo.sh
```

This covers:

- OCR: `vision/ppocr/PP-OCRv5_mobile_det.onnx`,
  `vision/ppocr/PP-OCRv5_mobile_rec.onnx`, `ppocrv5_dict.txt`
- ASR: `vlm/qwen3-asr-0.6B.tar.gz`,
  `asr/qwen3-asr-1.7B-dynq-q4km.tar.gz`, `asr/sensevoice.tar.gz`
- Embedding: BGE-Zh, BGE-En, Jina, Nomic, Qwen3-Embedding 0.6B GGUF
- Reranker: BGE-Reranker V2-M3 and Qwen3-Reranker 0.6B GGUF

Cache VLM artifacts:

```bash
SCOPE=vlm CACHE_ROOT=drivers/spacemit-ai/model_zoo \
  bash scripts/cache_spacemit_model_zoo.sh
```

Cache official ONNX vision artifacts:

```bash
SCOPE=vision CACHE_ROOT=drivers/spacemit-ai/model_zoo \
  bash scripts/cache_spacemit_model_zoo.sh
```

The cache script records downloaded file sizes in
`drivers/spacemit-ai/model_zoo/cache-index.tsv` and stores `.md5.actual` files
when upstream checksums are available. These files stay local with the cache.

## External HF Multimodal MoE Candidate

The SpacemiT ModelZoo high-parameter VLM options are still either slow
(`qwen30ba3b-mm-q4_1`) or not a single LLM+VLM replacement for the 35B text
models. For the external single-model experiment, cache the HF GGUF+mmproj
candidate separately from SpacemiT archive artifacts:

```text
drivers/huggingface/unsloth-Qwen3.5-35B-A3B-GGUF/
  Qwen3.5-35B-A3B-Q4_K_M.gguf  22016023168 bytes
  mmproj-F16.gguf               899283648 bytes
```

The direct Hugging Face transfer hit TLS EOF errors in this environment; the
same files were downloaded successfully through `hf-mirror.com` and then synced
to the K3 working cache:

```text
/root/models/spacemit-ai/vlm/unsloth-Qwen3.5-35B-A3B-GGUF/
```

Keep the local `drivers/huggingface/` copy as the canonical cache. The K3 copy
is a working cache and can be deleted only after the candidate is either fully
qualified or rejected.

K3 smoke result:

| Candidate | Runtime | Result | Evidence |
|---|---|---|---|
| `Qwen3.5-35B-A3B-Q4_K_M.gguf` + `mmproj-F16.gguf` | `/opt/spacemit-runtime/llama.cpp/bin/llama-server --mmproj` | loads as `capabilities=["completion","multimodal"]`, `vision_backend=mtmd`; no-think LLM/VLM smoke PASS; full VLM doc suite 29/30 case pass, field accuracy 0.9942, JSON parse 1.0, avg/p95 68.822/78.774s; 4K context run passed 1K/3K needle recall, but 3K took 297.122s | `output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-thinkingoff-official-20260707_190728/`; `output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-docfull-defaultimg-official-20260708_093315/`; `output/reports/k3-riscv-32g/qwen35-35b-a3b-hf-mmproj-context4k-official-20260708_101309/` |

Required direct-server invocation detail:

```bash
MODE=vlm-pair \
MODEL_PATH=/root/models/spacemit-ai/vlm/unsloth-Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf \
MMPROJ_PATH=/root/models/spacemit-ai/vlm/unsloth-Qwen3.5-35B-A3B-GGUF/mmproj-F16.gguf \
CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}' \
SERVER_EXTRA_ARGS='--parallel 1 --cache-ram 512' \
CTX_SIZE=2048 BATCH_SIZE=512 UBATCH_SIZE=128 \
bash /root/run_k3_32g_model_zoo_highspec.sh
```

Without `enable_thinking=false`, the model served successfully but text probes
returned reasoning-only responses with empty final `content`. Treat the
thinking-off setting as part of the candidate's required call contract until a
better template/API setting is validated.

Do not force `--image-min-tokens 1024 --image-max-tokens 1024` for this
candidate on the current K3 runtime. A control run stalled on the first image
request and logged repeated `non-consecutive token position` warnings. The
default mtmd image-token path completed the full 30-case document suite.

## Invocation Discovery

The local K3 runner scripts can describe their model invocation contract without
connecting to the board:

```bash
bash scripts/run_k3_32g_model_zoo_cached.sh --describe
bash scripts/run_k3_32g_model_zoo_nonllm_cached.sh --describe
```

The scripts intentionally require explicit local connection settings. Store
those values outside the repository and do not put real host IPs, users, or
passwords in reports, scripts, docs, or `run-config.json` outputs.

## Invocation Map

The tables below intentionally separate application-facing calls from benchmark
calls. Product integration should follow the AI SDK or gateway routes. Raw
benchmark scripts use direct llama-server and ORT calls so the same artifact can
be measured repeatedly and compared with ModelZoo rows.

| Workload | AI SDK / gateway call reference | Official performance reference | Local benchmark script |
|---|---|---|---|
| LLM | AI SDK starts `llama-server` and calls `llm_chat "<prompt>" "http://localhost:8080/v1" "<model>" ...`; gateway uses `POST /v1/chat/completions`. | ModelZoo LLM uses `llama-bench -p 128 -n 128 -mmp 0 -fa 1 -ub 128`. | `MODE=llm`, `llama-server -m <MODEL_PATH>`, `POST /v1/chat/completions`. |
| VLM tar | AI SDK VLM can use C++/Python demos or gateway `POST /v1/vlm/models/load` then `POST /v1/vlm/chat/completions` with `image_url`. | ModelZoo VLM example starts `llama-server --media-backend smt --smt-config-dir <dir>` with `SPACEMIT_EP_DENSE_ACCURACY_LEVEL=1`. | `MODE=vlm-tar`, extract tar, start SMT backend, call chat completions with `image_url`. |
| VLM VisionEncoder | Same application path as VLM tar. | Official VLM table publishes VisionEncoder latency for 4/8 core. The page does not publish a standalone encoder benchmark command; local probe uses `onnxruntime_perf_test -e spacemit` on the tar vision ONNX files. | `scripts/run_k3_32g_official_modelzoo_vlm_encoder_probe.sh`. |
| VLM GGUF+mmproj | Same chat request shape as VLM gateway or direct OpenAI-compatible server. | No separate official mmproj performance row in the cited ModelZoo page. | `MODE=vlm-pair`, `llama-server -m <MODEL_PATH> --mmproj <MMPROJ_PATH>`, chat with `image_url`. |
| Vision / OCR | AI SDK vision demos and gateway `POST /v1/vision/inference` are the official application layer. | ModelZoo vision uses `onnxruntime_perf_test ... -e spacemit ... SPACEMIT_EP_INTRA_THREAD_NUM`. PP-OCRv5 archive OCR is locally measured because the cited page does not publish OCR rows. | Broad runner uses PP-OCRv5 det+rec ONNX and records CER/NED on generated line data. |
| ASR | AI SDK uses `asr_file_demo`; gateway uses `POST /v1/asr/recognize` with uploaded audio. | ModelZoo lists qwen3-ASR 0.6B and sensevoice RTF rows. | qwen3-ASR tar packages use SMT backend and chat completions with audio input; sensevoice is package-inspected unless a K3 serving wrapper is present. |
| Embedding | AI SDK identifies Embed as a gateway-backed capability, but the cited quick verification section does not publish a concrete curl route. | The cited ModelZoo page does not publish embedding performance rows. | `MODE=embedding`, `llama-server --embedding --pooling mean`, `POST /v1/embeddings`. |
| Reranker | AI SDK identifies Rerank as a gateway-backed capability, but the cited quick verification section does not publish a concrete curl route. | The cited ModelZoo page does not publish reranker performance rows. | `MODE=rerank`, `llama-server --reranking --pooling rank`, `POST /v1/rerank`. |

## Local Data Paths

| Workload | Data path | Probe |
|---|---|---|
| LLM GGUF | `drivers/spacemit-ai/model_zoo/llm/<model>.gguf` | `POST /v1/chat/completions` |
| VLM tar | `drivers/spacemit-ai/model_zoo/vlm/<model>.tar.gz` | `POST /v1/chat/completions` with `image_url` |
| VLM GGUF+mmproj | `drivers/spacemit-ai/model_zoo/vlm/...` | `POST /v1/chat/completions` with `image_url` |
| Embedding | `drivers/spacemit-ai/model_zoo/embed/<model>.gguf` | `POST /v1/embeddings` |
| Reranker | `drivers/spacemit-ai/model_zoo/rerank/<model>.gguf` | `POST /v1/rerank` |
| OCR | `drivers/spacemit-ai/model_zoo/vision/ppocr/*` | generated OCR line dataset + CER/NED scoring |
| ASR | `drivers/spacemit-ai/model_zoo/vlm/qwen3-asr-0.6B.tar.gz` or `asr/qwen3-asr-1.7B-dynq-q4km.tar.gz` | audio recognition probe |

## VLM Full Run

Run the full K3 32G VLM suite from the local cache after loading connection
values into the local secure environment:

```bash
VLM_MAX_CASES=0 \
  bash scripts/run_k3_32g_model_zoo_vlm_full.sh
```

The script syncs the synthetic VLM document extraction dataset to
`/root/local-ai-bench` on the target and runs:

- SMT tar VLMs: `qwen30ba3b-mm-q4_1`, `Qwen3.5-4B`, `Qwen3.5-2B`,
  `Qwen3.5-0.8B`, `fastvlm-mm-0.5b-q4_1`
- GGUF+mmproj VLMs: `Qwen3VL-4B`, `SmolVLM-256M`

Set `RUN_HEAVY=0` to skip the 30B tar model, and set `VLM_MAX_CASES=N` for a
fast subset. `VLM_MAX_CASES=0` means all cases.

## Result Placement

- Raw run logs stay on the target under `/root/k3_32g_model_zoo_vlm_full/...`
  or in ignored local paths such as `output/` and `reports/runs/`.
- Curated public conclusions belong in fixed-name reports such as
  `reports/k3-riscv-32g.en.md`.
- Date-stamped reports, raw JSONL, model files, and transfer logs must not be
  promoted to tracked `reports/`.
