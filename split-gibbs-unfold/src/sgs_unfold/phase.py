"""Toy nonlinear phase retrieval: Y = |A S| + ε, ε ~ N(0, σ_y² I).

Paper experiments use Y = |FFT(B ⊙ X)| + ε; here A is a small real matrix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_A(key, m, n):
    """Dense real sensing matrix with columns of expected unit norm."""
    return jax.random.normal(key, (m, n)) / jnp.sqrt(n)


def forward(A, s):
    """|A s|."""
    return jnp.abs(A @ s)


def observe(key, A, s, meas_std):
    """y = |A s| + ε."""
    eps = meas_std * jax.random.normal(key, (A.shape[0],), dtype=s.dtype)
    return forward(A, s) + eps


def fy(s, y, A, meas_std):
    """f_y(s') ≜ -log p_{Y|S}(y|s') up to a constant.
    For Gaussian noise: (1/(2 σ_y²)) || y - |A s'| ||².
    """
    resid = forward(A, s) - y
    return 0.5 * jnp.sum(resid**2) / (meas_std**2)


def grad_fy(s, y, A, meas_std):
    """∇ f_y(s') = Aᵀ ((|A s'| - y) ⊙ sign(A s')) / σ_y²."""
    r = A @ s
    return (A.T @ ((jnp.abs(r) - y) * jnp.sign(r))) / (meas_std**2)


def measurement_residual(s, y, A):
    """|| y - |A s| ||₂."""
    return jnp.linalg.norm(forward(A, s) - y)
