# K3 llama.cpp / GGUF Path

**Last updated:** 2026-07-08
**Chinese version:** [llama.zh.md](llama.zh.md)
**Sources:** [CPU/RVV/IME run report](../../runs/k3-riscv-32g/20260704/k3-riscv-32g-cpu.en.md), [legacy full report](../../k3-riscv-32g.en.md)

## Scope

This path covers private SpacemiT `llama.cpp`, source-built SpacemiT `llama.cpp`, upstream fallback where needed, GGUF+mmproj VLM, embedding, and reranker. Qwen-family no-think handling is required for reliable OpenAI-compatible content.

## LLM Results

| Model | Key metrics | Status | Decision |
|---|---:|---|---|
| `Qwen3-30B-A3B-Q4_0` | PP512 33.69 tok/s, TG128 9.80 tok/s; GA PASS | PASS with limits | Primary K3 LLM for bounded prompts |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL` | PP512 30.49 tok/s, TG128 6.75 tok/s; 1K/3K needle PASS | PASS LLM, not VLM | Large LLM candidate only |
| `Qwen3.5-35B-A3B-Q4_0` | PP512 29.31 tok/s, TG128 6.48 tok/s; 1K/3K PASS | PASS LLM, not VLM | Large LLM control |
| `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | TTFT 1.071s, decode128 29.374s; 3K needle 297.122s | PASS, async-only | Single-service LLM+VLM candidate |
| `LFM2-24B-A2B-Q4_0` | PP512 55.48 tok/s, TG128 15.36 tok/s | Perf PASS, quality mixed | Throughput candidate only |
| `Qwen3-8B-Q4_K_M` | PP512 25.97 tok/s, TG128 4.24 tok/s | Perf PASS, quality mixed | Context/perf smoke |
| `Qwen3-4B-Q4_K_M` | PP512 42.14 tok/s, TG128 7.30 tok/s | PASS smoke | Small control |
| `Qwen3-0.6B-Q4_0` | PP512 198.51 tok/s, TG128 37.53 tok/s | PASS smoke | Pipeline smoke; not quality default |

## VLM via llama.cpp

| Model | Vision path | Key metric | Status | Decision |
|---|---|---:|---|---|
| `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | mtmd | 29/30 docs, avg/p95 68.822/78.774s | PASS | Async single-service candidate |
| `Qwen3VL-4B + mmproj` | mtmd | 30/30 docs; realistic p95 78.064s | PASS, slow | Quality control / async |
| `SmolVLM-256M + mmproj` | mtmd | image route runs but document quality fails | FAIL quality | Runtime smoke only |

## Embedding and Reranker

| Role | Model | Key metric | Status | Decision |
|---|---|---:|---|---|
| Embedding | `Bge-Small-Zh-V1.5-Q4_K_M` | p95 5.85ms, overall Hit@1 0.9722 | PASS | Default |
| Embedding | `Nomic-Embed-Text-V2-Moe-Q4_0` | p95 21.21ms, overall Hit@1 0.9722 | PASS | Alternate |
| Embedding | `Qwen3-Embedding-0.6B-Q4_0` | p95 45.27ms, overall Hit@1 0.9722 | PASS | Slower alternate |
| Embedding | `Bge-Small-En-V1.5-Q4_K_M` | finite vector ratio 0.0 | FAIL | Do not use |
| Reranker | `Bge-Reranker-V2-M3-Q4_0` | top50 Hit@1 1.0; top20 p95 1333ms | PASS | Default, cap top-k |
| Reranker | `Qwen3-Reranker-0.6B-Q4_0` | top50 p95 18804ms; top50 Hit@1 0.8333 | PASS slow/offline | Not online default |

## Source Runtime Equivalence

| Runtime row | Result | Decision |
|---|---|---|
| Source-built llama.cpp vs system | 6/6 PP/TG rows pass source/system >=0.95 | Source runtime can be optimization baseline |
| Qwen3.6 private-package loader issue | Earlier private package failed on missing `blk.40.ssm_conv1d.weight`; later current system probe serves text | Keep runtime version in evidence for every retest |

## Known Limits

| Limit | Impact |
|---|---|
| Qwen3/Qwen3.5 thinking mode | Use `/no_think` or `enable_thinking=false`; otherwise final `content` can be empty. |
| Long context | 30B/35B requests above small windows are latency and memory risks. |
| VLM image support | Text-only 35B GGUF files report `vision_backend=none`; image input fails unless an mmproj/vision path is loaded. |
| Resource margin | 30B realistic 1K reached 30.488GiB RSS, so admission control is mandatory. |
