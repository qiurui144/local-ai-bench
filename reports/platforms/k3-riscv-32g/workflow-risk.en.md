# K3 Workflow Risk

**Last updated:** 2026-07-08
**Chinese version:** [workflow-risk.zh.md](workflow-risk.zh.md)
**Sources:** [legacy full report](../../k3-riscv-32g.en.md), [docs/k3-realistic-stress-plan.md](../../../docs/k3-realistic-stress-plan.md)

## Scope

This report translates raw model results into product workflow risk for K3 32G. It covers common embedding, reranker, OCR, ASR, VLM, LLM, and long-document workflows.

## Workflow Decisions

| Workflow | Recommended path | Risk level | Required controls |
|---|---|---|---|
| Realtime RAG | BGE-Zh embedding -> BGE reranker top-k <=20 -> bounded Qwen3-30B answer | Medium | Context cap, token cap, per-request timeout |
| Document OCR | PP-OCRv5 OCR first, VLM only for visual reasoning | Low/Medium | Separate OCR and VLM queues; avoid VLM-as-OCR |
| Sync VLM upload | Qwen3.5-2B SMT | Medium | TCM preflight, timeout, visible fallback |
| Async high-quality VLM | Qwen3VL-4B, qwen30ba3b-mm, or Qwen3.5-35B+mmproj | High | Queue, TTL, cancellation, one-job admission |
| ASR | qwen3-ASR 0.6B SMT | Low/Medium | Normalize Chinese variants/numerals before scoring |
| Long aviation manuals | offline text/OCR -> embedding -> reranker -> cited evidence window -> async LLM | High | Tokenizer-aware clipping, queue isolation, memory guard |
| Model sweeps | Local `drivers/spacemit-ai/model_zoo` canonical cache, K3 working cache only | Medium | Delete non-hot K3 copies after evidence capture |

## Long-Context / Aviation Manuals

| Model/window | Result | Decision |
|---|---|---|
| `Qwen3-4B-Q4_0`, aviation 1K window | score 1.0, but 175.453s E2E | Pipeline works; sync UX fails |
| `Qwen3-4B-Q4_0`, aviation 3K window | first request exceeded 5 minutes before completion | Not sync acceptable |
| `Qwen3-0.6B-Q4_0`, naive 1K windows | context-overflow errors occurred | Character clipping is unsafe |
| `Qwen3-0.6B-Q4_0`, safety-budget windows | no overflow, score 0.5958 | API stable, quality below gate |
| `Qwen3-30B` / 35B class | short probes pass, but long text approaches memory/latency limits | Use only as async verifier with admission control |

## Resource Risks

| Resource | Observed issue | Control |
|---|---|---|
| Memory | 30B realistic 1K reached 30.488GiB RSS | Single large-model job, reject/queue long-context requests |
| TCM | Stale blocks caused ORT/SMT failures | Preflight `spacemit-tcm-smi`, release stale blocks before media runs |
| Storage | Model sweeps can fill K3 root fs | Keep canonical cache local; K3 holds only hot working copies |
| Scheduler | Raw model-server tests do not prove fairness/backpressure | Implement `/capacity`, queue wait, cancellation, TTL, and async status |
| Power | Board-level power was not sampled | Do not compare K3 TPS/W yet |

## Product Gates

| Gate | Requirement |
|---|---|
| Admission | Large LLM/VLM jobs must check free memory, TCM state, and current active jobs. |
| Timeouts | Sync paths need short hard timeouts; slow quality models must be async. |
| Cancellation | Async document and long-context jobs must support cancellation and TTL cleanup. |
| Evidence windows | Long documents must use retrieval and tokenizer-aware clipping, not whole-manual ingestion. |
| Metrics | Collect latency, RSS, TCM state, queue wait, error class, and model/runtime version per run. |
