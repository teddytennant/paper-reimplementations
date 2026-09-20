"""Estimation by adaptively weighted mid-summaries (Wang & Gao, arXiv:2609.20749).

Algorithm 1 in §3. Observations are split into a training half D_m of length
m = floor(n/2) and a holdout half D'_m. If n is odd, the leftover last point
is dropped from the holdout so both halves have length m.

Order statistics use 1-based ranks from the paper: X(r) is the r-th order
statistic, implemented as sorted[r - 1].

Zero-gap scales: if the holdout spacing ℓ̂_j is zero or tinier than
1e-15 times a crude scale of the sample, that scale is skipped (weight 0).
If every scale is skipped (e.g. all observations equal), equal weights are
used so the estimate is the common mid-summary value rather than NaN.

J_p is floor(log2(1/(4p))) with p = k/(m+1). The loop uses max(J_p, 0) so a
legal (n, k) with p > 1/4 still produces the j = 0 mid-summary.
"""

from __future__ import annotations

import math
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_TINY_REL = 1e-15


def jp(n: int, k: int) -> int:
    """J_p = floor(log2(1/(4p))) with p = k/(m+1) and m = floor(n/2)."""
    n = int(n)
    k = int(k)
    m = n // 2
    if m < 1 or k < 1:
        raise ValueError(f"cannot form J_p for n={n}, k={k}")
    p = k / (m + 1.0)
    return int(math.floor(math.log2(1.0 / (4.0 * p))))


def mid_summary(sorted_x: Any, r: int) -> float:
    """Mid-summary M_r = (X(r) + X(m+1-r)) / 2 for 1-based rank r."""
    xs = np.asarray(sorted_x, dtype=np.float64).reshape(-1)
    m = int(xs.shape[0])
    r = int(r)
    if r < 1 or r > m:
        raise ValueError(f"rank r={r} is outside 1..{m}")
    return float((xs[r - 1] + xs[m - r]) / 2.0)


def _validate(n: int, k: int) -> None:
    if n < 8:
        raise ValueError(f"n={n} is too small; need n >= 8")
    if k < 1 or k > n / 4.0:
        raise ValueError(f"k={k} is out of range; need 1 <= k <= n/4 (n={n})")


def _as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n = int(x.shape[0])
    m = n // 2
    train = x[:m]
    holdout = x[m : 2 * m]
    return train, holdout, m


def weights_and_mids(
    x: Any, k: int = 2
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (λ̂, mid-summaries, ℓ̂) for Algorithm 1.

    λ̂, mids, and ℓ̂ each have length J_p+1 (or 1 if J_p < 0). Weights are
    computed from the holdout half; mid-summaries from the training half.
    """
    x = _as_1d(x)
    n = int(x.shape[0])
    k = int(k)
    _validate(n, k)
    train, holdout, m = _split(x)
    xs = np.sort(train)
    xp = np.sort(holdout)
    J = max(jp(n, k), 0)
    scale = float(np.max(np.abs(xp)) + np.max(np.abs(xs)) + 1.0)
    tiny = _TINY_REL * scale

    ells = np.empty(J + 1, dtype=np.float64)
    mids = np.empty(J + 1, dtype=np.float64)
    ws = np.empty(J + 1, dtype=np.float64)
    for j in range(J + 1):
        r = (1 << j) * k
        r2 = (1 << (j + 1)) * k
        if r2 > m or r < 1:
            ell = 0.0
            mid = mid_summary(xs, min(max(r, 1), m))
        else:
            ell_minus = float(xp[r2 - 1] - xp[r - 1])
            ell_plus = float(xp[m - r] - xp[m - r2])
            ell = 0.5 * (ell_minus + ell_plus)
            mid = 0.5 * (float(xs[r - 1]) + float(xs[m - r]))
        ells[j] = ell
        mids[j] = mid
        if not math.isfinite(ell) or abs(ell) <= tiny:
            ws[j] = 0.0
        else:
            w = (2.0**j) / (ell * ell)
            ws[j] = w if math.isfinite(w) else np.inf

    if np.any(np.isinf(ws)):
        ws = np.where(np.isinf(ws), 1.0, 0.0)
    if not np.any(ws > 0.0):
        ws = np.ones_like(ws)
    lam = ws / ws.sum()
    return (
        jnp.asarray(lam, dtype=jnp.float64),
        jnp.asarray(mids, dtype=jnp.float64),
        jnp.asarray(ells, dtype=jnp.float64),
    )


def estimate(x: Any, k: int = 2) -> float:
    """Algorithm 1: adaptively weighted mid-summaries on a single split."""
    lam, mids, _ = weights_and_mids(x, k)
    return float(
        np.dot(
            np.asarray(lam, dtype=np.float64),
            np.asarray(mids, dtype=np.float64),
        )
    )


def estimate_crossfit(x: Any, k: int = 2) -> float:
    """Cross-fit Algorithm 1: average the two split directions.

    The numerical studies in the paper use k=2 plus this swap-and-average.
    """
    x = _as_1d(x)
    n = int(x.shape[0])
    k = int(k)
    _validate(n, k)
    m = n // 2
    leftover = x[2 * m :]
    a = x[:m]
    b = x[m : 2 * m]
    x_ab = np.concatenate([a, b, leftover])
    x_ba = np.concatenate([b, a, leftover])
    return 0.5 * (estimate(x_ab, k) + estimate(x_ba, k))
