# Case 02 — LLM judge position bias inflated a model swap

## Summary

A pairwise win-rate comparison of two LLM generators reported that
Candidate B beat Control A 62% to 38% on a 200-pair gold set,
prompting a sign-off to swap production. After deployment, user
thumbs-down rate increased 4 percentage points and engagement
dropped. A calibration audit revealed the judge had a 9% systemic
preference for whichever answer it saw FIRST in the prompt.

## How it was caught

The post-deployment review re-ran the same comparison with **A/B
order swapped** between calibration replays. When B was shown
first, B "won" 64% / 36%; when A was shown first, A won 53% / 47%.
The averaged win-rate was 57% / 43%, a much smaller and not
significant gap.

## Position bias detector

The audit ran:

```python
from benchmark.rag.judge_calibration import (
    GoldPair, replay_calibration_pairs, calibration_report,
)
rows = replay_calibration_pairs(
    pairs, simulated_judge, n_runs_per_pair=3, swap_order=True
)
report = calibration_report(rows, pairs, n_runs_per_pair=3)
# report.position_bias = +0.18
```

## Fix and prevention

- **Required A/B swap** in any pairwise judging. Average the two
  orderings before declaring a winner.
- **Two-sided judge prompt** that explicitly instructs the judge
  to ignore order; not sufficient on its own (only reduces bias to
  ~6%) but combined with averaging it works.
- **Position bias < 5%** added to the calibration gate matrix.

## Takeaway

LLM judges are not unbiased oracles. Any benchmark methodology that
trusts a single A-vs-B pass is incorrect by construction. Always
randomize and average; report bias along with win-rate.
