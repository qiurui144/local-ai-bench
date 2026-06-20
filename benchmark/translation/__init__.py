"""Translation evaluation for LLM/VLM benchmark.

Adds a machine-translation quality + latency dimension on top of the
existing vLLM OpenAI-compatible harness.

Modules
-------
- ``accuracy``    : ``translate_batch`` (calls the served model) + corpus-level
  SacreBLEU / chrF (pure-Python via the ``sacrebleu`` package) and COMET
  (GPU-only, gracefully skipped when unavailable).
- ``performance`` : per-language-pair TTFT + tokens/s, reusing the project
  performance idiom (streaming for TTFT, sustained sync for throughput).
- ``datasets``    : Flores-200 loader (HF ``facebook/flores`` with an offline
  built-in fallback) and a custom JSONL product-domain loader.

Task levels (prompt templates live in ``prompts``):
- L1 single-sentence (zh->en / en->zh)
- L2 multi-sentence context consistency (pronoun / tense agreement)
- L3 terminology preservation (technical glossary, exact-match term rate)
"""

from __future__ import annotations

__all__ = [
    "accuracy",
    "performance",
    "datasets",
    "prompts",
]
