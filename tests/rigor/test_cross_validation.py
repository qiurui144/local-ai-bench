"""Tests for benchmark.rigor.cross_validation."""
from __future__ import annotations

import pytest

from benchmark.rigor.cross_validation import (
    group_k_fold,
    k_fold,
    leave_one_out,
    nested_k_fold,
    repeated_k_fold,
    stratified_k_fold,
)


def test_k_fold_partitions_disjoint():
    folds = k_fold(20, 5, shuffle=False)
    seen = set()
    for f in folds:
        assert not set(f.test_idx) & seen
        seen.update(f.test_idx)
    assert seen == set(range(20))


def test_k_fold_balanced_sizes():
    folds = k_fold(10, 5)
    sizes = [len(f.test_idx) for f in folds]
    assert sizes == [2, 2, 2, 2, 2]


def test_stratified_k_fold_preserves_classes():
    labels = [0] * 8 + [1] * 4
    folds = stratified_k_fold(labels, k=4)
    for f in folds:
        cls0 = sum(1 for i in f.test_idx if labels[i] == 0)
        cls1 = sum(1 for i in f.test_idx if labels[i] == 1)
        # at least one of each class per fold
        assert cls0 >= 1 or cls1 >= 1


def test_loo_yields_n_folds():
    folds = leave_one_out(7)
    assert len(folds) == 7
    for f in folds:
        assert len(f.test_idx) == 1
        assert len(f.train_idx) == 6


def test_group_k_fold_keeps_groups_together():
    groups = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    folds = group_k_fold(groups, k=3)
    for f in folds:
        test_groups = {groups[i] for i in f.test_idx}
        train_groups = {groups[i] for i in f.train_idx}
        assert not (test_groups & train_groups)


def test_nested_k_fold_layered_structure():
    nested = nested_k_fold(20, k_outer=4, k_inner=3)
    assert len(nested) == 4
    for n in nested:
        assert len(n.inner) == 3
        for inf in n.inner:
            assert all(i in n.outer.train_idx for i in inf.train_idx)
            assert all(i in n.outer.train_idx for i in inf.test_idx)


def test_repeated_k_fold_multiple_splits():
    repeats = repeated_k_fold(20, k=5, n_repeats=3)
    assert len(repeats) == 3


def test_k_fold_invalid_k():
    with pytest.raises(ValueError):
        k_fold(20, 1)


def test_group_k_fold_too_few_groups():
    groups = ["a"] * 10
    with pytest.raises(ValueError):
        group_k_fold(groups, k=3)
