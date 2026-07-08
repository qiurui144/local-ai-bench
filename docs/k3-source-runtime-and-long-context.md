# K3 Source Runtime and Long-Context Plan

This document covers two K3 32GB follow-up tracks:

- Source-built SpacemiT `llama.cpp` / ONNX Runtime comparison against the system runtime.
- Long-context quality gating with aviation manuals.

## Source Runtime Build

Build SpacemiT A100-targeted `llama.cpp`:

```bash
BUILD_LLAMA=1 BUILD_ORT=0 bash scripts/build_spacemit_a100_sources.sh
```

Build ONNX Runtime after caching CMake dependencies:

```bash
python3 scripts/cache_spacemit_ort_deps.py --download-missing
BUILD_LLAMA=0 BUILD_ORT=1 bash scripts/build_spacemit_a100_sources.sh
```

Inputs:

- Toolchain defaults to `drivers/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.2.4` when present.
- Source repos default to `drivers/spacemit-source/llama.cpp` and `drivers/spacemit-source/onnxruntime`.
- Install tarballs land under `builds/spacemit-a100/`.

The `llama.cpp` build enables:

```text
GGML_CPU_RISCV64_SPACEMIT=ON
GGML_RVV=ON
GGML_RV_ZVFH=ON
GGML_RV_ZFH=ON
GGML_RV_ZICBOP=ON
GGML_RV_ZIHINTPAUSE=ON
GGML_RV_ZBA=ON
```

The current local `llama.cpp` build emits RISC-V attributes containing
`xsmtvdotii`, and CMake detects `RISCV64_SPACEMIT_IME1` plus
`RISCV64_SPACEMIT_IME2`.

The current local ORT build installs:

```text
builds/spacemit-a100/onnxruntime-install.tar.gz
builds/spacemit-a100/onnxruntime-install/bin/onnxruntime_perf_test
builds/spacemit-a100/onnxruntime-install/lib/libonnxruntime.so.1.24.0+spacemit.a3
```

The ORT CMake dependency mirror is cached under
`drivers/ort-cmake-deps-mirror/`. The local source patch keeps the default
RISC-V MLAS flags unchanged but allows `SPACEMIT_RISCV_MLAS_FLAGS` to include
`xsmtvdotii` for A100 feature probing.

## Runtime Compare

Run default LLM comparison after loading K3 connection values into the local
secure environment:

```bash
bash scripts/run_k3_32g_source_runtime_compare.sh
```

Run ORT comparison with an ONNX manifest:

```bash
cat > /tmp/k3-ort-models.tsv <<'TSV'
label	remote_onnx_path	core_count
resnet18	/root/models/spacemit-ai/vision/resnet/resnet18.q.onnx	8
TSV

RUN_ORT_COMPARE=1 ORT_MODEL_MANIFEST=/tmp/k3-ort-models.tsv \
bash scripts/run_k3_32g_source_runtime_compare.sh
```

Pass gates:

- `llama.cpp`: source/system TPS ratio must be `>= 0.95`.
- ORT: source/system average latency ratio must be `<= 1.05`.

Current `llama.cpp` result from `output/reports/k3-riscv-32g/source-runtime-compare-20260707_101930/summary.tsv`:

| Model | Test | System TPS | Source TPS | Source/System |
|---|---:|---:|---:|---:|
| qwen3-0.6B | pp128 | 497.11 | 499.41 | 1.005 |
| qwen3-0.6B | tg128 | 52.58 | 54.21 | 1.031 |
| qwen3-4B | pp128 | 76.81 | 76.16 | 0.992 |
| qwen3-4B | tg128 | 10.85 | 10.98 | 1.012 |
| qwen3-30B-A3B | pp128 | 55.50 | 55.23 | 0.995 |
| qwen3-30B-A3B | tg128 | 12.28 | 12.50 | 1.017 |

Current ORT result from
`output/reports/k3-riscv-32g/source-runtime-compare-20260707_111859/ort-summary.tsv`:

| Model | Core | System ms | Source ms | Source/System | Status |
|---|---:|---:|---:|---:|---|
| fastvlm-vision | 8 | 158.942 | 155.254 | 0.977 | PASS |
| qwen3vl30b-vision | 8 | 4623.870 | 4653.055 | 1.006 | PASS |

Conclusion: source-built `llama.cpp` and source-built ORT are
performance-equivalent to the system SpacemiT runtimes for the tested A100/K3
LLM and VisionEncoder rows. This is enough to use the source trees as the
baseline for K3 optimization work, while still keeping system runtime retests as
the regression gate.

## Aviation Manuals

The long-context suite now includes
`https://github.com/shiroinekotfs/airplane-manual-collection` as an aviation
manual source. The upstream repository itself warns that manuals may be
unverified for real flight use; these cases are benchmark-only.

Cache selected manuals and generate cases:

```bash
python3 scripts/cache_long_context_suites.py --skip-longbench --airplane-manuals
```

Default K3 long-context run after loading K3 connection values into the local
secure environment:

```bash
bash scripts/run_k3_32g_long_context_20b.sh
```

Controls:

```bash
AIRPLANE_MANUAL_SCOPE=core|broad|all
AIRPLANE_MANUAL_CASE_LIMIT=12
AIRPLANE_MANUAL_CONTEXT_TOKENS=3072
AIRPLANE_MANUAL_PROMPT_BUDGET_SAFETY=0.70
MAX_INPUT_TOKENS=3072
```

The aviation suite generates three case types per usable PDF:

- `span_recall`: retrieve the exact line following an anchor line.
- `keyword_recall`: list technical terms from a marked paragraph.
- `manual_needle`: locate a synthetic validation code in an aircraft manual excerpt.

Current long-term aviation material cache:

| Manual | Cases | PDF pages | Extracted text chars |
|---|---:|---:|---:|
| `Airbus/A350/FDS Briefing/a350-900-flight-deck-and-systems-briefing-for-pilots.pdf` | 3 | 389 | 273,298 |
| `Airbus/A220/QRH/a220-300-cs300-bd500-1a11-quick-reference-handbook.pdf` | 3 | 760 | 883,577 |
| `Boeing/B737/FCOM/737MAX FCOM.pdf` | 3 | 1,528 | 2,565,750 |
| `Boeing/B737/QRH/B737-700 Quick Reference Handbook (QRH).pdf` | 3 | 358 | 465,832 |
| `Boeing/B737/FCTM/B737 Flight Crew Training Manual - All.pdf` | 3 | 396 | 720,373 |

Quality requirement for K3 32GB:

- Direct full-manual ingestion is not the default product path.
- Long documents must go through retrieval/windowing first.
- Sync requests should stay within the tested context budget and must use a
  tokenizer-aware or conservative prompt budget. Character-count estimates alone
  undercounted real Qwen tokens in the first 0.6B aviation run.
- 20B+ models and VLM quality-control paths remain async-only with timeout,
  cancellation, and queue backpressure.

Current aviation-manual evidence:

| Model / window | Cases | Quality | Latency / runtime | Finding |
|---|---:|---|---|---|
| `Qwen3-4B-Q4_0`, 1K text window | 1 | PASS, score 1.0 | 175.453s E2E; prefill 1263 tokens at 7.74 tok/s; RSS about 5GB | API, scoring, and report chain are valid, but synchronous latency is too high for interactive manual QA. |
| `Qwen3-4B-Q4_0`, 3K text window | 0 completed | aborted | first request exceeded 5 minutes before completion; server log showed 1024 prompt tokens at 9.51 tok/s before cancellation | Direct 3K manual windows are not acceptable as a sync default. |
| `Qwen3-0.6B-Q4_0`, 1K naive window | 6 attempted, 4 measured | score 0.75 on measured cases; 2 HTTP 400 context overflows | mean 62.747s, p95/max 74.659s | Character-based prompt fitting was not reliable enough for K3 serving limits. |
| `Qwen3-0.6B-Q4_0`, 1K safety-budget window | 6 measured | FAIL overall, aviation score 0.5958; no context overflows | mean 57.438s, p95/max 143.945s; prompts 796-1911 tokens | Conservative clipping stabilizes API calls but loses evidence quality; retrieval must provide tighter answer-centered chunks before LLM generation. |

Recommended aviation-manual architecture on K3 32GB:

1. Extract PDF text/OCR offline and store page/chunk metadata.
2. Build embeddings with `Bge-Small-Zh-V1.5-Q4_K_M` or the language-matched
   embedding model.
3. Use `Bge-Reranker-V2-M3-Q4_0` with bounded top-k, preferably <=20 for
   interactive paths.
4. Use tokenizer-aware/adaptive clipping after reranking. Start from the page or
   chunk containing the answer anchor, add lead/lag context, and shrink with the
   serving tokenizer or a conservative safety factor.
5. Send only cited evidence windows to the LLM. Keep 4B/8B for triage and
   30B/35B/VLM quality-control paths async.
6. Require source page references and a refusal path when evidence windows do
   not contain the requested procedure or limit.
