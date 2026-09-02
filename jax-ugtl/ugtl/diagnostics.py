"""The two trajectory-quality measures of Section 5.3.1.

    R(tau) = sum_{k=1}^{H} (1 - s_k)
    S(tau) = (1 / (H - 2)) sum_{k=1}^{H-2} || x_{k+2} - 2 x_{k+1} + x_k ||^2

s_k is the normalized objective value, so regret is small when a trajectory stays
in high-value regions. The second difference is the change between consecutive
displacements, so a small S means successive moves agree with each other. The
paper min-max scales the design coordinates before computing S; pass designs that
are already scaled, or hand in the box with ``lo`` and ``hi``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cumulative_regret", "smoothness"]


def cumulative_regret(scores: np.ndarray) -> np.ndarray:
    """R(tau) per trajectory, from normalized scores of shape (N, H)."""
    return np.sum(1.0 - np.asarray(scores), axis=-1)


def smoothness(designs: np.ndarray, lo=None, hi=None) -> np.ndarray:
    """S(tau) per trajectory, from designs of shape (N, H, d)."""
    X = np.asarray(designs, dtype=np.float64)
    if lo is not None and hi is not None:
        X = (X - np.asarray(lo)) / (np.asarray(hi) - np.asarray(lo))
    second = np.diff(X, n=2, axis=1)
    return np.mean((second**2).sum(axis=-1), axis=-1)
