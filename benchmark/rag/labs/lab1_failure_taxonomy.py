"""Lab 1: Failure-mode taxonomy walk-through.

Builds three synthetic RAG runs each exhibiting a different failure
kind, then uses the component_pipeline.FailureKind enum to classify
them and produce a per-stage attribution table.

Run:
    python -m benchmark.rag.labs.lab1_failure_taxonomy
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List

from ..component_pipeline import FailureKind


@dataclass
class SyntheticRun:
    query: str
    retrieved_docs: List[str]
    relevant_docs: List[str]
    answer: str
    citations: List[str]


def classify(run: SyntheticRun) -> List[FailureKind]:
    kinds: List[FailureKind] = []
    if not any(d in run.relevant_docs for d in run.retrieved_docs):
        kinds.append(FailureKind.RETRIEVAL_MISS)
    if run.retrieved_docs and len(set(run.retrieved_docs) & set(run.relevant_docs)) < 1 and len(run.retrieved_docs) > 3:
        kinds.append(FailureKind.RETRIEVAL_NOISE)
    if not run.citations and run.answer:
        kinds.append(FailureKind.CITATION_MISSING)
    if any(c not in run.retrieved_docs for c in run.citations):
        kinds.append(FailureKind.CITATION_FABRICATED)
    if run.answer.lower().startswith("i don't have") and run.relevant_docs:
        kinds.append(FailureKind.OVER_REFUSAL)
    if "always" in run.answer.lower() and "always" not in " ".join(run.retrieved_docs).lower():
        kinds.append(FailureKind.HALLUCINATION)
    return kinds


def main() -> None:
    runs = [
        SyntheticRun(
            query="What is the capital of France?",
            retrieved_docs=["wiki_germany", "wiki_italy", "wiki_spain"],
            relevant_docs=["wiki_france"],
            answer="The capital of France is Paris.",
            citations=["wiki_france"],  # citation_fabricated; not retrieved
        ),
        SyntheticRun(
            query="Tell me about quantum entanglement",
            retrieved_docs=["physics_quantum_1"],
            relevant_docs=["physics_quantum_1"],
            answer="I don't have enough information.",
            citations=[],
        ),
        SyntheticRun(
            query="Is coffee always bad for you?",
            retrieved_docs=["health_coffee_2024"],
            relevant_docs=["health_coffee_2024"],
            answer="Coffee is always harmful to health.",
            citations=["health_coffee_2024"],
        ),
    ]

    print("# Lab 1: failure-mode classification")
    print("-" * 60)
    summary = Counter()
    for i, r in enumerate(runs, 1):
        kinds = classify(r)
        print(f"\nRun {i}: {r.query}")
        print(f"  retrieved={r.retrieved_docs}")
        print(f"  answer={r.answer!r}")
        print(f"  failures={[k.value for k in kinds]}")
        for k in kinds:
            summary[k.value] += 1
    print("\nAggregate failure histogram:")
    for kind, n in summary.most_common():
        print(f"  {kind}: {n}")


if __name__ == "__main__":
    main()
