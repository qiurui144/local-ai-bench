# Case 03 — Retrieval recall regression hidden by NDCG averaging

## Summary

A swap from BM25 to a hybrid dense+BM25 retriever showed
+0.04 NDCG@10 in aggregate. After two weeks the legal-domain
sub-bucket dropped 0.12 NDCG. The aggregate metric had averaged
across domains where the new retriever improved by 0.10 and a
domain where it lost 0.12. Production had no per-domain alarm.

## How it was caught

A user complained that case-citation queries had stopped surfacing
the correct case. The diagnostic re-ran the same queries against
the two retrievers and per-domain reporting showed the regression
clearly.

## What we now require

- **Per-bucket reporting**: every retrieval metric must be reported
  by domain / difficulty / language as well as in aggregate (see
  `benchmark/rag/retrieval_metrics.py::bucketed_metrics`).
- **Quality gate matrix**: each subgroup carries its own
  pass/warn/fail thresholds; release blocks if any bucket fails
  even when the overall is green
  (`benchmark/rag/rubrics/quality_gate_matrix.yaml`).
- **Subgroup audit** in `benchmark/rigor/ood_assessment.py` flags
  any bucket whose gap from overall exceeds a tolerance.

## Takeaway

Aggregates lie. The 94%-overall-0%-on-Bucket-Z pathology is so
common that every benchmark must default to bucketed reporting
with per-bucket gates.
