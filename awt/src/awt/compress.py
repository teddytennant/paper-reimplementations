"""Column-scale a weight, then factor the preconditioned matrix with TT-SVD."""

import jax.numpy as jnp

from awt.importance import (
    DEFAULT_ALPHA,
    DEFAULT_EPS,
    DEFAULT_MIN_SCALE,
    column_scales,
    importance_mean_abs,
    importance_std,
)
from awt.tt import interleave, tt_matvec, tt_svd

_STATISTICS = {
    "mean_abs": importance_mean_abs,
    "meanabs": importance_mean_abs,
    "mean_absolute": importance_mean_abs,
    "abs": importance_mean_abs,
    "importance_mean_abs": importance_mean_abs,
    "std": importance_std,
    "stdev": importance_std,
    "stddev": importance_std,
    "importance_std": importance_std,
}


def _resolve_statistic(statistic):
    if callable(statistic):
        return statistic
    key = str(statistic).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _STATISTICS:
        known = "mean_abs, std"
        raise ValueError(f"unknown statistic {statistic!r}; expected {known} or a callable")
    return _STATISTICS[key]


def _prod(factors):
    total = 1
    for factor in factors:
        total *= int(factor)
    return total


def compress(
    W,
    C,
    statistic="mean_abs",
    alpha=DEFAULT_ALPHA,
    eps=DEFAULT_EPS,
    min_scale=DEFAULT_MIN_SCALE,
    m_factors=None,
    n_factors=None,
    ranks=None,
):
    """Scale columns of W by activation importance, then TT-SVD.

    Returns (scales, cores). W_prime = W @ diag(s) scales columns, not rows.
    statistic is "mean_abs", "std", or a callable (C, eps) -> (n,).
    """
    if m_factors is None or n_factors is None:
        raise ValueError("m_factors and n_factors are required")
    W = jnp.asarray(W)
    C = jnp.asarray(C)
    if W.ndim != 2:
        raise ValueError("W must have shape (m, n)")
    if C.ndim != 2:
        raise ValueError("C must have shape (T, n)")
    m_factors = tuple(int(v) for v in m_factors)
    n_factors = tuple(int(v) for v in n_factors)
    if len(m_factors) != len(n_factors):
        raise ValueError("m_factors and n_factors must have the same length")
    if _prod(m_factors) != int(W.shape[0]) or _prod(n_factors) != int(W.shape[1]):
        raise ValueError("factor products must equal W.shape")
    if int(C.shape[1]) != int(W.shape[1]):
        raise ValueError("C must have one column per input channel of W")

    score = _resolve_statistic(statistic)
    importance = score(C, eps)
    scales = column_scales(importance, alpha=alpha, eps=eps, min_scale=min_scale)
    # Column scaling only. W_prime[:, i] = s_i * W[:, i].
    w_prime = W * scales[None, :]
    cores = tt_svd(interleave(w_prime, m_factors, n_factors), ranks)
    return scales, cores


def deploy(cores, scales, x):
    """Deployment identity: y = W_hat'(D^{-1} x) = tt_matvec(cores, x / scales)."""
    x = jnp.asarray(x)
    scales = jnp.asarray(scales)
    return tt_matvec(cores, x / scales)
