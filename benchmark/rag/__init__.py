"""RAG validation framework.

Implements the 12-chapter RAG evaluation methodology covering:
- Component-wise pipeline stages (Ch 1)
- Offline/online alignment (Ch 2)
- Retrieval metrics (Ch 3) + bpref/ERR/RBP academic extensions
- Reranker assessment (Ch 4) + RRF/Borda/CombSUM/CombMNZ fusion
- Answer relevance (Ch 5) + ROUGE/BLEU/chrF/embedding-sim baselines
- Groundedness (Ch 6) + RAGAS strict faithfulness
- LLM-as-judge prompts (Ch 7) with G-Eval CoT
- Judge calibration (Ch 8) with position/verbosity/self-preference biases
- Judge attack hardening (Ch 9) with adversarial perturbation suite
- Regression CI (Ch 10) with flake controller
- Canary rollback (Ch 11) with shadow runner and traffic splitter
- Drift detection (Ch 12) with PSI/JS, temporal cohorts, auto-curation

Plus appendices: schemas, capstone, interview Q&A, labs, rubrics,
case studies. See `docs/` and subpackages for the appendix content.
"""

__version__ = "0.2.0"


__all__ = [
    "component_pipeline",
    "offline_online_alignment",
    "retrieval_metrics",
    "reranker",
    "answer_relevance",
    "groundedness",
    "judge_prompts",
    "judge_calibration",
    "judge_attacks",
    "regression_ci",
    "canary",
    "drift_detection",
]
