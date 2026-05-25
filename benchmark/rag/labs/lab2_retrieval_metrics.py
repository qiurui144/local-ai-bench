"""Lab 2: Walk through retrieval metrics on a toy 5-query dataset.

Computes NDCG / MAP / MRR / bpref / ERR / RBP on the same set so you
can see how each metric ranks the same systems differently. Useful to
internalize why no single metric is sufficient.

Run:
    python -m benchmark.rag.labs.lab2_retrieval_metrics
"""
from __future__ import annotations

from ..retrieval_metrics import (
    RetrievalQueryResult,
    average_precision,
    bpref,
    err_at_k,
    mean_average_precision,
    mean_mrr,
    mean_ndcg_at_k,
    ndcg_at_k,
    rank_biased_precision,
    reciprocal_rank,
)


def main() -> None:
    # System A: relevant doc at rank 1.
    # System B: relevant doc at rank 3 but with two also-relevant at 5/6.
    system_a = [
        RetrievalQueryResult(
            query_id=f"q{i}",
            ranked_doc_ids=["d1", "d2", "d3", "d4", "d5"],
            relevant_doc_ids=["d1"],
            relevance_grades={"d1": 3.0, "d2": 0, "d3": 0, "d4": 0, "d5": 0},
        )
        for i in range(5)
    ]
    system_b = [
        RetrievalQueryResult(
            query_id=f"q{i}",
            ranked_doc_ids=["d2", "d3", "d1", "d4", "d5"],
            relevant_doc_ids=["d1", "d5"],
            relevance_grades={"d1": 3.0, "d2": 0, "d3": 0, "d4": 0, "d5": 1.0},
        )
        for i in range(5)
    ]

    print("# Lab 2: retrieval metrics side-by-side")
    print("-" * 60)
    for name, sysX in [("A", system_a), ("B", system_b)]:
        ndcg = mean_ndcg_at_k(sysX, 5)
        mmrr = mean_mrr(sysX)
        mmap = mean_average_precision(sysX)
        first = sysX[0]
        err = err_at_k(first.ranked_doc_ids, first.relevance_grades or {}, 5)
        rbp = rank_biased_precision(first.ranked_doc_ids, first.relevant_doc_ids, 0.8)
        bp = bpref(first.ranked_doc_ids, first.relevant_doc_ids, first.ranked_doc_ids)
        print(f"\nSystem {name}:")
        print(f"  NDCG@5  = {ndcg:.4f}")
        print(f"  MRR     = {mmrr:.4f}")
        print(f"  MAP     = {mmap:.4f}")
        print(f"  ERR@5   = {err:.4f}    (cascade-style user model)")
        print(f"  RBP p=0.8 = {rbp:.4f}  (user persistence)")
        print(f"  bpref   = {bp:.4f}    (robust to incomplete judgments)")

    print(
        "\nObserve: A wins on MRR (relevant at rank 1) but B has two relevant "
        "docs total so MAP may flip. ERR + RBP show how cascade vs persistence "
        "user models reshape conclusions."
    )


if __name__ == "__main__":
    main()
