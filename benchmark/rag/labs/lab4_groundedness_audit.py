"""Lab 4: Claim-level groundedness audit.

Demonstrates the decomposition -> per-claim judgment -> report flow on
a single answer with mixed support quality.

Run:
    python -m benchmark.rag.labs.lab4_groundedness_audit
"""
from __future__ import annotations

from ..groundedness import (
    ClaimJudgment,
    decompose_claims,
    groundedness_report,
)


def main() -> None:
    answer = (
        "The Eiffel Tower was built in 1889 [tour_eiffel_history]. "
        "It is the tallest building in France [tour_eiffel_history]. "
        "It was originally intended for the Paris Exposition [paris_expo_1889]. "
        "Today it has 99 million visitors per year [made_up_doc]."
    )
    claims = decompose_claims(answer)
    print("# Lab 4: groundedness audit")
    print("-" * 60)
    print("Decomposed claims:")
    for c in claims:
        print(f"  {c.claim_id}: {c.text!r}")
        print(f"          cited={c.cited_doc_ids}")

    # Imagined human/LLM judge verdict.
    judgments = [
        ClaimJudgment(claim_id="c0", supported=True, supporting_doc_ids=["tour_eiffel_history"]),
        ClaimJudgment(
            claim_id="c1",
            supported=False,
            supporting_doc_ids=[],
            notes="Not the tallest; contradicts evidence",
        ),
        ClaimJudgment(claim_id="c2", supported=True, supporting_doc_ids=["paris_expo_1889"]),
        ClaimJudgment(
            claim_id="c3",
            supported=False,
            supporting_doc_ids=[],
            notes="Citation fabricated; doc not in corpus",
        ),
    ]

    rpt = groundedness_report(claims, judgments)
    print("\nGroundedness report:")
    print(f"  n_claims:                 {rpt.n_claims}")
    print(f"  n_supported:              {rpt.n_supported}")
    print(f"  grounded_rate:            {rpt.grounded_rate:.3f}")
    print(f"  strict faithfulness:      {rpt.faithfulness_strict:.3f}")
    print(f"  attribution precision:    {rpt.attribution_precision:.3f}")
    print(f"  attribution recall:       {rpt.attribution_recall:.3f}")
    print(f"  citation precision:       {rpt.citation_precision:.3f}")
    print(f"  citation recall:          {rpt.citation_recall:.3f}")


if __name__ == "__main__":
    main()
