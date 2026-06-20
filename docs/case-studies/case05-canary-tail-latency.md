# Case 05 — Canary mean OK, P95 latency violation rolled out anyway

## Summary

A new rerank model passed the canary gate because the mean latency
was 220 ms (well under the 500 ms ceiling). The P95 was 1.6
seconds. After full rollout the P95 user-facing latency tripled
and 1.2% of users saw 30-second timeouts. The canary gate had been
configured with `mean` only.

## How it was caught

A timeout alarm fired in production. Post-hoc analysis of the
canary samples (which had been kept) confirmed the P95 was always
in violation; the operator just had not configured the gate
correctly.

## What we now require

- **`CanaryGate` defaults to evaluating BOTH mean and P95** for any
  latency-style metric, with separate thresholds.
- **Rolling P95 vs absolute P95**: a canary running for 10
  minutes can give a misleading P95 because the slow tail had no
  time to materialize. The window-size minimum sample requirement
  in `CanaryGate(min_samples=...)` prevents premature promotion.
- **Per-percentile failure budget**: any single 100-sample window
  whose P95 exceeds the SLO counts as a breach; three consecutive
  breaches trigger rollback (`RollbackPolicy(consecutive_breaches=3)`).

## Takeaway

Latency is a distribution, not a number. Means hide tails. Always
guard at the percentile you care about.
