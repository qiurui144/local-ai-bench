# Baselines

A baseline is the "is the new thing actually better than nothing"
benchmark. This document records the baselines this framework expects
to compare against, the null hypotheses they represent, and the
literature points we calibrate our default thresholds to.

## Retrieval baselines

| Baseline | What it is | Where it lives |
|---|---|---|
| Random ranker | Uniform random ordering of corpus | tests fixture |
| BM25 default | Whoosh / tantivy with default `k1=1.2, b=0.75` | Built into the test harness |
| Dense bi-encoder default | `bge-m3` cosine top-K | Out-of-box |
| Hybrid (RRF) | BM25 + dense via `reciprocal_rank_fusion` | `benchmark.rag.reranker` |

Modern strong-result references for context (no SOTA chasing in this
bench; we report against representative public benchmarks):

- BEIR (Thakur, N. et al. 2021). BM25 baseline NDCG@10 by dataset
  ranges 0.30-0.65 across 18 datasets.
- MS MARCO Passage (Nguyen, T. et al. 2016). BM25 MRR@10 baseline
  0.187; dense bi-encoder usually 0.32-0.36; cross-encoder rerank
  0.39-0.42.

Our defaults: NDCG@10 >= 0.65, MRR >= 0.55 (per `quality_gate_matrix
.yaml`). Below those, retrieval is the bottleneck, not generation.

## Generation baselines

| Baseline | What it represents |
|---|---|
| Greedy decode of the production model | "Did the change hurt?" |
| Refuse-on-any-doubt baseline | Trivial 100% precision, ~0% recall |
| Cited-evidence-only generation | Honest grounded floor |

References for refusal calibration:

- Mishra, N. et al. (2024). Calibrated Selective Classification with
  Inference-time Refusal. (over-refusal tradeoff curves)

## Quality (judge) baselines

We do not publish "Sonnet beats Haiku" without:

- judge calibration (`calibration_report`) on a published gold pair
  set,
- effect size + CI on the win-rate gap,
- multi-seed and order-swapped averaging.

Internal baseline: a single judge run on a single ordering is reported
*only* as "unaudited."

## Latency baselines

| Stage | P50 budget | P95 budget |
|---|---|---|
| Retrieval | 200 ms | 600 ms |
| Rerank | 50 ms | 200 ms |
| Generation (TTFT) | 1500 ms | 3000 ms |
| End-to-end | 3000 ms | 6000 ms |

These are *minimum* expectations for a competitive production RAG;
your domain may tighten them. See `quality_gate_matrix.yaml` for the
canonical thresholds the framework enforces.

## Null hypotheses we reject only with evidence

1. The new ranker is no better than BM25 (Wilcoxon paired NDCG@10
   on a 200-query held-out set, p < 0.01, effect size d > 0.2).
2. The new generator is no better than the current generator
   (paired groundedness; intent_satisfaction win rate after order
   swap; both p < 0.01).
3. The new judge agrees with humans (Cohen's kappa >= 0.6).
4. The candidate's online behavior matches offline (KS p > 0.10;
   spearman > 0.7).
5. The dataset is fresh (no public-set contamination per SHA256
   check against known leaks).

## What we do not claim

- That a single round of "X beats Y" generalizes beyond the gold set.
- That a strong human-judge agreement implies generalization across
  domains.
- That a fast model is a better model. Latency and quality are both
  required to be in budget.
