"""Cross-validation splitters.

We need k-fold / stratified-k-fold / leave-one-out / nested CV for two
purposes inside this bench:

1. Calibration: holdout split that does not contaminate the test set
   (the held-out portion gets re-calibrated via Platt/Isotonic and the
   remainder is used to measure the final ECE).
2. Hyperparameter selection on small golden sets: nested CV is the only
   defensible protocol against test-set leakage.

We intentionally do not depend on sklearn here so the rigor module works
in lean deployment environments. Implementations are short, documented,
and validated against sklearn fixtures in the test suite.

References
----------
- Stone, M. (1974). Cross-Validatory Choice and Assessment of Statistical
  Predictions. JRSS B.
- Varma, S. & Simon, R. (2006). Bias in error estimation when using
  cross-validation for model selection. BMC Bioinformatics. (nested CV
  motivation)
- Cawley, G. C. & Talbot, N. L. C. (2010). On Over-fitting in Model
  Selection and Subsequent Selection Bias in Performance Evaluation. JMLR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CVFold:
    train_idx: List[int]
    test_idx: List[int]


# ---------------------------------------------------------------------------
# K-fold and friends
# ---------------------------------------------------------------------------


def k_fold(n_samples: int, k: int, shuffle: bool = True, seed: Optional[int] = 0) -> List[CVFold]:
    """Plain k-fold split. Folds are roughly equal-sized."""
    if k < 2:
        raise ValueError("k must be >=2")
    if n_samples < k:
        raise ValueError("n_samples must be >= k")
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1
    folds: List[CVFold] = []
    current = 0
    for size in fold_sizes:
        test_idx = indices[current : current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size :]])
        folds.append(CVFold(train_idx=train_idx.tolist(), test_idx=test_idx.tolist()))
        current += size
    return folds


def stratified_k_fold(
    labels: Sequence[int],
    k: int,
    shuffle: bool = True,
    seed: Optional[int] = 0,
) -> List[CVFold]:
    """Stratified k-fold preserving class proportions per fold.

    Essential when classes are imbalanced (which they nearly always are
    on real benchmark golden sets).
    """
    labels_arr = np.asarray(labels)
    n = labels_arr.size
    if k < 2:
        raise ValueError("k must be >=2")
    unique = np.unique(labels_arr)
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    for cls in unique:
        cls_idx = np.where(labels_arr == cls)[0]
        if shuffle:
            rng.shuffle(cls_idx)
        # Round-robin assign.
        for i, idx in enumerate(cls_idx):
            folds[i % k].append(int(idx))
    out: List[CVFold] = []
    all_idx = set(range(n))
    for f in folds:
        test_idx = sorted(f)
        train_idx = sorted(all_idx - set(test_idx))
        out.append(CVFold(train_idx=train_idx, test_idx=test_idx))
    return out


def leave_one_out(n_samples: int) -> List[CVFold]:
    """LOO: n folds, each holding out a single sample. Use only on small
    golden sets (n < ~50); otherwise prefer k-fold for variance reasons."""
    folds: List[CVFold] = []
    for i in range(n_samples):
        train_idx = [j for j in range(n_samples) if j != i]
        folds.append(CVFold(train_idx=train_idx, test_idx=[i]))
    return folds


def group_k_fold(
    groups: Sequence,
    k: int,
    shuffle: bool = True,
    seed: Optional[int] = 0,
) -> List[CVFold]:
    """K-fold that keeps grouped samples together (no leakage across folds).

    Use when, e.g., multiple golden queries share the same source document
    and we want the test set to evaluate generalization across documents.
    """
    groups_arr = np.asarray(groups)
    unique_groups = np.unique(groups_arr)
    if len(unique_groups) < k:
        raise ValueError("need at least k unique groups")
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(unique_groups)
    fold_groups: List[List] = [[] for _ in range(k)]
    for i, g in enumerate(unique_groups):
        fold_groups[i % k].append(g)
    folds: List[CVFold] = []
    n = groups_arr.size
    all_idx = set(range(n))
    for fg in fold_groups:
        mask = np.isin(groups_arr, fg)
        test_idx = np.where(mask)[0].tolist()
        train_idx = sorted(all_idx - set(test_idx))
        folds.append(CVFold(train_idx=train_idx, test_idx=test_idx))
    return folds


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NestedCVFold:
    outer: CVFold
    inner: List[CVFold]  # k_inner folds over the outer training set


def nested_k_fold(
    n_samples: int,
    k_outer: int,
    k_inner: int,
    shuffle: bool = True,
    seed: Optional[int] = 0,
) -> List[NestedCVFold]:
    """Nested CV for unbiased hyperparameter selection.

    The outer loop estimates generalization; the inner loop picks
    hyperparameters within each outer training set. Total fits = k_outer
    * k_inner; budget accordingly.
    """
    outer = k_fold(n_samples, k_outer, shuffle=shuffle, seed=seed)
    out: List[NestedCVFold] = []
    for of in outer:
        # Renumber the outer train as positions 0..len(of.train_idx) for
        # the inner splitter; but we keep the original indices so the
        # caller can slice their actual data array.
        inner_relative = k_fold(
            n_samples=len(of.train_idx),
            k=k_inner,
            shuffle=shuffle,
            seed=(seed or 0) + 1,
        )
        # Translate inner indices back to global.
        translated: List[CVFold] = []
        for inf in inner_relative:
            tr = [of.train_idx[i] for i in inf.train_idx]
            te = [of.train_idx[i] for i in inf.test_idx]
            translated.append(CVFold(train_idx=tr, test_idx=te))
        out.append(NestedCVFold(outer=of, inner=translated))
    return out


# ---------------------------------------------------------------------------
# Iteration helpers used in run loops
# ---------------------------------------------------------------------------


def iter_folds(folds: Sequence[CVFold]) -> Iterator[Tuple[int, CVFold]]:
    """Convenience iterator yielding (fold_id, fold) pairs."""
    for i, f in enumerate(folds):
        yield i, f


def repeated_k_fold(
    n_samples: int,
    k: int,
    n_repeats: int,
    seed: Optional[int] = 0,
) -> List[List[CVFold]]:
    """`n_repeats` independent k-fold splits with different seeds.

    Reduces variance from a single random split when only one shot at the
    full CV budget is available.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    rng = np.random.default_rng(seed)
    all_repeats: List[List[CVFold]] = []
    for _ in range(n_repeats):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        all_repeats.append(k_fold(n_samples, k, shuffle=True, seed=sub_seed))
    return all_repeats
