"""Reconstruction-target masks.

``True`` means masked: that token is a reconstruction target and is hidden
from the encoder. ``structure_mask`` draws masked indices with probability
proportional to damped-canvas variances, so informative regions are the
targets. All-zero variances fall back to ``uniform_mask``.
"""

from __future__ import annotations

import numpy as np


def mask_count(n, ratio):
    """Number of masked tokens. About ``ratio`` of ``n``, with both roles present."""
    n = int(n)
    if n <= 0:
        return 0
    if n == 1:
        return 1
    k = int(np.round(float(ratio) * n))
    return int(max(1, min(n - 1, k)))


def uniform_mask(n, ratio=0.75, seed=0):
    """Boolean mask with about ``ratio`` tokens marked True (masked)."""
    n = int(n)
    k = mask_count(n, ratio)
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(n, size=k, replace=False)
    mask[chosen] = True
    return mask


def structure_mask(variances, ratio=0.75, seed=0):
    """Sample masked indices proportional to ``variances``.

    Zero variance tokens are less likely (weight 0 while any positive weight
    remains). If every variance is 0, fall back to ``uniform_mask``.
    """
    variances = np.asarray(variances, dtype=np.float64).reshape(-1)
    n = int(variances.shape[0])
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not np.any(variances > 0):
        return uniform_mask(n, ratio=ratio, seed=seed)

    k = mask_count(n, ratio)
    rng = np.random.default_rng(int(seed))
    weights = np.clip(variances, 0.0, None)
    available = np.ones(n, dtype=bool)
    chosen = []
    for _ in range(k):
        current = np.where(available, weights, 0.0)
        total = float(current.sum())
        if total <= 0.0:
            candidates = np.flatnonzero(available)
            pick = int(rng.choice(candidates))
        else:
            probs = current / total
            pick = int(rng.choice(n, p=probs))
        chosen.append(pick)
        available[pick] = False
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(chosen, dtype=int)] = True
    return mask
