"""Centered weighted affine ridge, the closed-form transfer interface of section 3.1.

Weights are the mean-one convention of eq 8, so sum_i w_i = n and the weighted
means of eq 9 carry the n^-1 factor.
"""

from __future__ import annotations

import jax.numpy as jnp

LAMBDA_DEFAULT = 0.01  # section 4.1, inherited from the reference implementation


def weighted_mean(X, w):
    """xbar_w = n^-1 sum_i w_i x_i, eq 9. X is (..., n, p) and w is (..., n)."""
    n = X.shape[-2]
    return jnp.einsum("...n,...np->...p", w, X) / n


def sufficient_statistics(X, Y, w, mu_x=None, mu_y=None):
    """A_h = Xc' Dw Xc and B_h = Xc' Dw Yc, eqs 20-21.

    X is (..., n, p), Y is (..., n, dt), w is (..., n). Means default to eq 9.
    """
    if mu_x is None:
        mu_x = weighted_mean(X, w)
    if mu_y is None:
        mu_y = weighted_mean(Y, w)
    Xc = X - mu_x[..., None, :]
    Yc = Y - mu_y[..., None, :]
    A = jnp.einsum("...n,...np,...nq->...pq", w, Xc, Xc)
    B = jnp.einsum("...n,...np,...nd->...pd", w, Xc, Yc)
    return A, B, mu_x, mu_y


def solve_from_stats(A, B, mu_x, mu_y, lam: float = LAMBDA_DEFAULT):
    """W_h = (A_h + lambda I)^-1 B_h and b_h = mu_y - mu_x W_h, eq 23."""
    p = A.shape[-1]
    W = jnp.linalg.solve(A + lam * jnp.eye(p, dtype=A.dtype), B)
    b = mu_y - jnp.einsum("...p,...pd->...d", mu_x, W)
    return W, b


def fit_affine(X, Y, w=None, lam: float = LAMBDA_DEFAULT):
    """One-shot centered weighted affine ridge, eq 10. Returns (W, b)."""
    if w is None:
        w = jnp.ones(X.shape[:-1], dtype=X.dtype)
    A, B, mu_x, mu_y = sufficient_statistics(X, Y, w)
    return solve_from_stats(A, B, mu_x, mu_y, lam)


def apply_affine(X, W, b):
    """The deployed online map, y = x W + b."""
    return jnp.einsum("...np,...pd->...nd", X, W) + b[..., None, :]


def ridge_objective(W, b, X, Y, w, lam: float = LAMBDA_DEFAULT):
    """The eq 8 objective evaluated at an arbitrary (W, b), for gradient checks.

    Eq 8 fixes the intercept to eq 9, so this writes the residual with an
    explicit b and leaves b out of the penalty.
    """
    resid = Y - (jnp.einsum("np,pd->nd", X, W) + b)
    return jnp.sum(w * jnp.sum(resid**2, axis=-1)) + lam * jnp.sum(W**2)
