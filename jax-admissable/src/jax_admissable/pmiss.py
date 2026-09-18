"""Adversary missingness distribution p_miss (paper Eq. 8, Table 1)."""

from __future__ import annotations

import numpy as np

# Table 1
ZETA = 5000
EPS = 1e-8
L_BOUND = 0.0


def init_pmiss(d):
    """Uniform distribution over ``d`` features."""
    d = int(d)
    if d <= 0:
        raise ValueError("d must be positive")
    return np.ones(d, dtype=np.float64) / d


def sample_missing_index(p_miss, rng):
    """Draw a single missing feature index (k=1)."""
    p = np.asarray(p_miss, dtype=np.float64)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def update_pmiss(returns, l=L_BOUND, eps=EPS):
    """Inverse-score update of p_miss (Eq. 8).

    ``p_miss(i) = (1 / (a_i - l + eps)) / sum_u (1 / (a_u - l + eps))``

    Features whose missingness yields a *lower* return get a *higher*
    probability. Returns are clipped to be at least ``l`` so the
    denominator cannot be zero or negative.
    """
    a = np.asarray(returns, dtype=np.float64)
    a = np.maximum(a, l)
    inv = 1.0 / (a - l + eps)
    p = inv / inv.sum()
    return p


def update_pmiss_baseline(p_miss, returns=None, l=L_BOUND, eps=EPS):
    """Uniform baseline: leave p_miss unchanged."""
    return np.asarray(p_miss, dtype=np.float64).copy()
