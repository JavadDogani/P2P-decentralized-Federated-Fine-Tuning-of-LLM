from __future__ import annotations
from typing import List, Sequence
import numpy as np

def dirichlet_partition(labels: Sequence[int], n_clients: int, alpha: float, seed: int) -> List[np.ndarray]:
    """Label-skew split via Dirichlet. Returns index arrays per client."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    idx_by_class = {c: np.where(labels == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx = idx_by_class[c]
        # draw proportions per client
        props = rng.dirichlet(alpha=np.full(n_clients, alpha))
        # split
        splits = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        parts = np.split(idx, splits)
        for i, part in enumerate(parts):
            client_indices[i].extend(part.tolist())

    for i in range(n_clients):
        rng.shuffle(client_indices[i])
    return [np.asarray(ci, dtype=int) for ci in client_indices]
