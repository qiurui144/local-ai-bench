# Case 01 — Digit-shift hallucination on a financial Q&A bot

## Summary

A customer-facing financial Q&A bot answered "What is the minimum
deposit for a 30-day fixed-term product?" with **"¥120"** when the
correct number, also present verbatim in the cited evidence, was
**"¥1200"**. Perplexity-based offline metrics had ranked the
candidate model above the production baseline; the digit shift
slipped through to production for 6 days before a customer-support
ticket flagged it.

## How it was caught

- A user reported that the deposit they actually saw on the web page
  did not match the chatbot answer.
- A retroactive scan over chat traces using a `must_not_say` clause
  derived from the support ticket surfaced 14 more affected
  conversations.

## What the existing bench measured and missed

| Metric | Value | Did it flag the bug? |
|---|---|---|
| Perplexity vs baseline | -0.02 (better) | No |
| ROUGE-L vs gold answer | 0.91 | No |
| Embedding similarity | 0.97 | No |
| Groundedness (claim-level) | NOT TRACKED at the time | N/A |
| Must-not-say violations | NOT TRACKED at the time | N/A |

## Root cause

Two contributors:

1. The new model preferred shorter responses; the tokenizer
   compressed the digit string aggressively and the model produced
   a one-digit-shorter alternative.
2. The judge prompt focused on "is the answer correct?" but the
   judge LLM, lacking access to the cited evidence in its context
   window during sampling, frequently rubber-stamped "correct."

## Fix and prevention

- **Claim-level groundedness** added: every numeric claim is
  extracted via regex, located in the cited evidence, and a
  mismatch is a critical failure (see
  `benchmark/rag/groundedness.py`).
- **Must-not-say golden set**: 60 high-risk numerical/legal
  substrings now in the test fixture.
- **Evidence-only judge prompt**: enforced by
  `benchmark/rag/judge_prompts.py::GROUNDEDNESS_SYSTEM_PROMPT`.
- **Regression CI snapshot**: per-item ratchet means a regression
  in numeric accuracy cannot land silently.

## Takeaway

A correlation-style metric (ROUGE, embedding similarity) cannot
catch a one-character substitution that materially changes meaning.
Numeric correctness needs a dedicated, deterministic check.
