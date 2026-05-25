"""Canary rollout + rollback (PDF Chapter 11).

When a new model / pipeline change ships, we don't just flip the switch.
We route a fraction of production traffic to the candidate, measure
quality signals, and either ramp up or roll back.

Provided
--------
- TrafficSplitter: deterministic % routing keyed by a request hash.
- ShadowRunner: clone-and-compare without affecting the user response.
- CanaryGate: rolling-window quality vs latency thresholds.
- RollbackPolicy: encodes the rules for automatic rollback.

This is a *library*; the actual traffic-routing integration goes through
the application's request handler.

References
----------
- Tang, D. et al. (2010). Overlapping Experiment Infrastructure (Google).
- Crowne et al. (2020). Continuous deployment of large-scale ML services.
"""
from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Traffic splitting
# ---------------------------------------------------------------------------


class TrafficSplitter:
    """Deterministic hash-based traffic splitter.

    `fraction_to_candidate` in [0, 1]: e.g. 0.1 means 10% to candidate, 90%
    to control. Routing is keyed by request_id so the same request always
    sees the same arm (no cross-arm leakage on retries).
    """

    def __init__(self, fraction_to_candidate: float, salt: str = "canary"):
        if not 0.0 <= fraction_to_candidate <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        self.fraction = fraction_to_candidate
        self.salt = salt

    def route(self, request_id: str) -> str:
        digest = hashlib.sha256(f"{self.salt}::{request_id}".encode()).hexdigest()
        # First 8 hex = 32-bit; map to [0, 1).
        bucket = int(digest[:8], 16) / 2**32
        return "candidate" if bucket < self.fraction else "control"


# ---------------------------------------------------------------------------
# Shadow runner
# ---------------------------------------------------------------------------


@dataclass
class ShadowComparison:
    request_id: str
    control_output: Dict[str, Any]
    candidate_output: Dict[str, Any]
    metric_deltas: Dict[str, float]


class ShadowRunner:
    """Send every request to BOTH control and candidate; return control to
    the user while logging candidate for offline comparison.

    Provide a `comparator_fn(control, candidate) -> {metric: delta}` to
    convert raw outputs into a metric-delta dict.
    """

    def __init__(
        self,
        control_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        candidate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        comparator_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, float]],
    ):
        self.control_fn = control_fn
        self.candidate_fn = candidate_fn
        self.comparator_fn = comparator_fn

    def serve(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(request.get("request_id", str(time.time_ns())))
        control_out = self.control_fn(request)
        try:
            candidate_out = self.candidate_fn(request)
        except Exception as e:  # noqa: BLE001
            candidate_out = {"_error": repr(e)}
        deltas = (
            self.comparator_fn(control_out, candidate_out)
            if "_error" not in candidate_out
            else {}
        )
        # In production this gets pushed to async log; here we return it inline
        # so the test harness can capture.
        return {
            "user_response": control_out,
            "shadow": ShadowComparison(
                request_id=request_id,
                control_output=control_out,
                candidate_output=candidate_out,
                metric_deltas=deltas,
            ),
        }


# ---------------------------------------------------------------------------
# Canary gate
# ---------------------------------------------------------------------------


@dataclass
class CanaryWindow:
    timestamp: float
    metric: str
    value: float


@dataclass
class CanaryThreshold:
    metric: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class CanaryStatus:
    decision: str  # "promote" | "hold" | "rollback"
    n_samples: int
    metric_summary: Dict[str, Dict[str, float]]
    violations: List[str] = field(default_factory=list)


class CanaryGate:
    """Rolling-window guardrails for a live canary.

    Push samples via `record(metric, value)`. Call `evaluate()` to get the
    current decision. The gate uses a deque per metric capped at
    `window_size`; older points fall off.
    """

    def __init__(
        self,
        thresholds: Sequence[CanaryThreshold],
        window_size: int = 500,
        min_samples: int = 50,
    ):
        self.thresholds = {t.metric: t for t in thresholds}
        self.window_size = window_size
        self.min_samples = min_samples
        self._buf: Dict[str, Deque[float]] = {t.metric: deque(maxlen=window_size) for t in thresholds}

    def record(self, metric: str, value: float) -> None:
        if metric not in self._buf:
            return  # untracked metric; ignore silently
        self._buf[metric].append(float(value))

    def evaluate(self) -> CanaryStatus:
        summary: Dict[str, Dict[str, float]] = {}
        violations: List[str] = []
        n_samples = 0
        for metric, buf in self._buf.items():
            if not buf:
                summary[metric] = {"n": 0.0}
                continue
            arr = list(buf)
            n_samples = max(n_samples, len(arr))
            mean = sum(arr) / len(arr)
            p95 = sorted(arr)[max(0, int(0.95 * len(arr)) - 1)]
            summary[metric] = {
                "n": float(len(arr)),
                "mean": float(mean),
                "p95": float(p95),
                "min": float(min(arr)),
                "max": float(max(arr)),
            }
            thr = self.thresholds[metric]
            if thr.min_value is not None and mean < thr.min_value:
                violations.append(f"{metric}.mean ({mean:.3f}) < {thr.min_value:.3f}")
            if thr.max_value is not None and mean > thr.max_value:
                violations.append(f"{metric}.mean ({mean:.3f}) > {thr.max_value:.3f}")
        if n_samples < self.min_samples:
            decision = "hold"
        elif violations:
            decision = "rollback"
        else:
            decision = "promote"
        return CanaryStatus(
            decision=decision,
            n_samples=n_samples,
            metric_summary=summary,
            violations=violations,
        )


# ---------------------------------------------------------------------------
# Rollback policy
# ---------------------------------------------------------------------------


@dataclass
class RollbackPolicy:
    """Concrete rules for automatic rollback.

    A breach is recorded each time `evaluate()` returns "rollback". After
    `consecutive_breaches` such breaches we issue the actual rollback so
    that single-window noise does not cause a flip-flop.
    """

    consecutive_breaches: int = 3
    cooldown_seconds: float = 300.0

    def __post_init__(self) -> None:
        self._breach_streak = 0
        self._last_rollback_ts: Optional[float] = None

    def step(self, status: CanaryStatus) -> Dict[str, Any]:
        action = "noop"
        now = time.time()
        if status.decision == "rollback":
            self._breach_streak += 1
            if (
                self._breach_streak >= self.consecutive_breaches
                and (
                    self._last_rollback_ts is None
                    or now - self._last_rollback_ts > self.cooldown_seconds
                )
            ):
                action = "rollback_now"
                self._last_rollback_ts = now
                self._breach_streak = 0
        elif status.decision == "promote":
            action = "promote"
            self._breach_streak = 0
        else:
            self._breach_streak = max(0, self._breach_streak - 1)
        return {
            "action": action,
            "breach_streak": self._breach_streak,
            "last_rollback_ts": self._last_rollback_ts,
        }
