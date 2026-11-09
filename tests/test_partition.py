from __future__ import annotations

import numpy as np
import pytest

from p2p_peft_fl.data.partition import dirichlet_partition

N_CLASSES = 10
LABELS = np.repeat(np.arange(N_CLASSES), 100)  # 1000 samples, balanced


def label_skew(labels: np.ndarray, parts) -> float:
    """Size-weighted mean total-variation distance between client and global label distributions."""
    glob = np.bincount(labels, minlength=N_CLASSES) / len(labels)
    total = 0.0
    for p in parts:
        if len(p) == 0:
            continue
        local = np.bincount(labels[p], minlength=N_CLASSES) / len(p)
        total += len(p) / len(labels) * 0.5 * np.abs(local - glob).sum()
    return total


@pytest.mark.parametrize("n_clients", [1, 3, 10])
@pytest.mark.parametrize("alpha", [0.05, 0.5, 100.0])
@pytest.mark.parametrize("seed", [0, 123])
def test_every_sample_is_assigned_exactly_once(n_clients, alpha, seed):
    parts = dirichlet_partition(LABELS, n_clients=n_clients, alpha=alpha, seed=seed)
    assert len(parts) == n_clients
    allocated = np.concatenate(parts)
    assert allocated.dtype.kind == "i"
    assert np.array_equal(np.sort(allocated), np.arange(len(LABELS)))


def test_partition_is_deterministic_per_seed():
    a = dirichlet_partition(LABELS, n_clients=5, alpha=0.5, seed=42)
    b = dirichlet_partition(LABELS, n_clients=5, alpha=0.5, seed=42)
    c = dirichlet_partition(LABELS, n_clients=5, alpha=0.5, seed=43)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_label_skew_increases_as_alpha_decreases():
    alphas = [100.0, 10.0, 1.0, 0.1]
    skews = [
        np.mean([label_skew(LABELS, dirichlet_partition(LABELS, 10, alpha, seed)) for seed in range(10)])
        for alpha in alphas
    ]
    assert all(lo < hi for lo, hi in zip(skews, skews[1:])), skews
    assert skews[0] < 0.1  # large alpha: close to IID
    assert skews[-1] > 0.5  # small alpha: strongly non-IID


def test_works_with_non_contiguous_label_values():
    labels = np.array([3, 7, 7, 3, 9, 9, 9, 3])
    parts = dirichlet_partition(labels.tolist(), n_clients=2, alpha=1.0, seed=0)
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(labels)))
