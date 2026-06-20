"""Regression CI (PDF Chapter 10).

A golden-query set is checked into the repo. On every PR / nightly:
1. Run the full RAG pipeline against the golden set.
2. Compare per-query metrics to the previous best snapshot.
3. Flag regressions: any metric dropping by more than `tolerance`.
4. Flake controller: rerun flagged items N times and majority-vote.

This module is intended to be wired into `.github/workflows/ci.yml`:
    python -m benchmark.rag.regression_ci \
        --golden tests/data/golden_v1.jsonl \
        --baseline reports/baseline.json \
        --output reports/regression.json \
        --tolerance 0.02 --flake-retries 3

Caller injects the run_fn so this stays independent of any specific
LLM/embedder.

References
----------
- Liang, P. et al. (2022). Holistic Evaluation of Language Models (HELM).
  (rigor of golden-query snapshots)
- ML Test Score (Breck, E. et al. 2017). Practices for monitoring.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class GoldenItem:
    item_id: str
    query: str
    expected: Dict[str, Any]


@dataclass
class GoldenRunRow:
    item_id: str
    metrics: Dict[str, float]


@dataclass
class GoldenBaselineRow:
    item_id: str
    metrics: Dict[str, float]


@dataclass
class RegressionRow:
    item_id: str
    metric: str
    baseline: float
    current: float
    delta: float
    regressed: bool


@dataclass
class RegressionReport:
    baseline_timestamp: Optional[float]
    run_timestamp: float
    n_items: int
    regressions: List[RegressionRow]
    aggregate_deltas: Dict[str, float]
    flakey_items: List[str] = field(default_factory=list)

    def passed(self) -> bool:
        return not self.regressions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "baseline_timestamp": self.baseline_timestamp,
            "run_timestamp": self.run_timestamp,
            "n_items": self.n_items,
            "regressions": [asdict(r) for r in self.regressions],
            "aggregate_deltas": self.aggregate_deltas,
            "flakey_items": self.flakey_items,
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_golden_jsonl(path: Path) -> List[GoldenItem]:
    items: List[GoldenItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            items.append(
                GoldenItem(
                    item_id=str(rec["item_id"]),
                    query=str(rec["query"]),
                    expected=dict(rec.get("expected", {})),
                )
            )
    return items


def load_baseline(path: Path) -> Dict[str, GoldenBaselineRow]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: Dict[str, GoldenBaselineRow] = {}
    for r in data.get("rows", []):
        rows[r["item_id"]] = GoldenBaselineRow(
            item_id=r["item_id"], metrics={k: float(v) for k, v in r["metrics"].items()}
        )
    return rows


def write_snapshot(rows: Sequence[GoldenRunRow], path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "timestamp": time.time(),
        "rows": [
            {"item_id": r.item_id, "metrics": dict(r.metrics)} for r in rows
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return Path(path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_against_golden(
    items: Sequence[GoldenItem],
    run_fn: Callable[[GoldenItem], Dict[str, float]],
) -> List[GoldenRunRow]:
    """Evaluate `run_fn` over every golden item and return per-item rows."""
    rows: List[GoldenRunRow] = []
    for item in items:
        metrics = run_fn(item)
        rows.append(GoldenRunRow(item_id=item.item_id, metrics={k: float(v) for k, v in metrics.items()}))
    return rows


# ---------------------------------------------------------------------------
# Regression comparator (with flake controller)
# ---------------------------------------------------------------------------


def detect_regressions(
    current: Sequence[GoldenRunRow],
    baseline: Dict[str, GoldenBaselineRow],
    tolerance: float = 0.02,
    direction: str = "higher_is_better",
) -> RegressionReport:
    """Compare per-item metrics vs baseline.

    `tolerance` is absolute. `direction` decides regression polarity.
    """
    regs: List[RegressionRow] = []
    deltas_by_metric: Dict[str, List[float]] = {}
    for row in current:
        base = baseline.get(row.item_id)
        if base is None:
            # New item; not a regression but worth logging.
            continue
        for metric, value in row.metrics.items():
            if metric not in base.metrics:
                continue
            base_v = base.metrics[metric]
            delta = value - base_v
            deltas_by_metric.setdefault(metric, []).append(delta)
            regressed = (
                delta < -tolerance if direction == "higher_is_better" else delta > tolerance
            )
            if regressed:
                regs.append(
                    RegressionRow(
                        item_id=row.item_id,
                        metric=metric,
                        baseline=base_v,
                        current=value,
                        delta=delta,
                        regressed=True,
                    )
                )
    aggregate = {
        k: statistics.fmean(v) if v else 0.0 for k, v in deltas_by_metric.items()
    }
    return RegressionReport(
        baseline_timestamp=None,
        run_timestamp=time.time(),
        n_items=len(current),
        regressions=regs,
        aggregate_deltas=aggregate,
    )


def flake_recheck(
    flagged: Sequence[RegressionRow],
    run_fn: Callable[[str], Dict[str, float]],
    n_retries: int = 3,
    tolerance: float = 0.02,
    direction: str = "higher_is_better",
) -> List[RegressionRow]:
    """Re-run each flagged item N times; keep only items that majority-fail.

    `run_fn(item_id) -> metrics_dict`.

    Returns the still-regressing rows; items that no longer regress on
    majority vote are treated as flakes and dropped from the report.
    """
    surviving: List[RegressionRow] = []
    for row in flagged:
        votes: List[bool] = []
        for _ in range(n_retries):
            metrics = run_fn(row.item_id)
            v = metrics.get(row.metric, row.current)
            delta = v - row.baseline
            regressed = (
                delta < -tolerance if direction == "higher_is_better" else delta > tolerance
            )
            votes.append(regressed)
        if sum(votes) > n_retries / 2:
            surviving.append(row)
    return surviving


# ---------------------------------------------------------------------------
# Convenience CI entrypoint
# ---------------------------------------------------------------------------


def run_ci(
    golden_path: Path,
    baseline_path: Path,
    run_fn: Callable[[GoldenItem], Dict[str, float]],
    output_path: Path,
    tolerance: float = 0.02,
    flake_retries: int = 3,
    direction: str = "higher_is_better",
) -> RegressionReport:
    items = load_golden_jsonl(golden_path)
    rows = run_against_golden(items, run_fn)
    baseline = load_baseline(baseline_path) if Path(baseline_path).exists() else {}
    report = detect_regressions(rows, baseline, tolerance=tolerance, direction=direction)
    if report.regressions and flake_retries > 0:
        item_by_id = {it.item_id: it for it in items}

        def _retry_one(item_id: str) -> Dict[str, float]:
            return run_fn(item_by_id[item_id])

        report.regressions = flake_recheck(
            report.regressions, _retry_one, flake_retries, tolerance, direction
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report
