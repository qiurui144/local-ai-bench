# Case 06 — Embedding model swap silently degraded retrieval

## Summary

An upstream embedding library auto-updated from v1.0 to v1.1
(minor version bump) and the embedding dimension changed from 768
to 1024. New documents were indexed at 1024; old documents
remained at 768. The vector store silently zero-padded the older
embeddings, destroying retrieval quality for the legacy corpus.

The aggregate offline metric was stable because new documents
dominated the post-update query set; legacy queries had degraded
~30%. The change went unnoticed for 11 days.

## How it was caught

A weekly cohort report (per
`benchmark/rag/drift_detection.py::per_week_performance`) showed
NDCG@10 declining 0.02 per week despite no apparent change in
the inputs. A reproducibility snapshot
(`benchmark/rigor/reproducibility.py`) revealed the embedding
package had been transitively upgraded.

## What we now require

- **Pin the embedding library version** in `requirements.txt` to
  an exact patch.
- **Reproducibility snapshot** stored alongside every run; diff
  against last run on every release.
- **Embedding dimension assertion** at index time: writing a
  mismatched-dim vector raises rather than silently padding.
- **Temporal performance drift** check
  (`temporal_performance_drift`) included in nightly CI; any week
  showing >0.05 drop triggers escalation.

## Takeaway

Silent dependency drift is one of the cheapest ways to lose
months of quality. The reproducibility module is not a nicety;
it is the only thing that lets you assign blame six months later.
