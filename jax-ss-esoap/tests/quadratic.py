"""Shared ill-conditioned matrix quadratic, Section 5.2.

``f(X) = 0.5 ||A X B - C||_F^2`` with ``X in R^{m x n}``, gradient
``A^T (A X B - C) B^T`` and Hessian ``(B B^T) kron (A^T A)``, exactly Kronecker.
Dimensions are much smaller than the paper's (500, 100, 1000, 250) so the suite
stays fast on a CPU.
"""

import jax.numpy as jnp
import numpy as np

M, N, P, Q = 24, 16, 32, 20


def _factor(rng, rows, cols, kappa):
    k = min(rows, cols)
    u, _ = np.linalg.qr(rng.standard_normal((rows, k)))
    v, _ = np.linalg.qr(rng.standard_normal((cols, k)))
    return (u * np.geomspace(1.0, 1.0 / kappa, k)) @ v.T


def make(seed, kappa=1e3, m=M, n=N, p=P, q=Q):
    """Returns ``(A, B, C, X0, X_star)`` in float64."""
    rng = np.random.default_rng(seed)
    a = _factor(rng, p, m, kappa)
    b = _factor(rng, n, q, kappa)
    x_star = rng.standard_normal((m, n))
    return (
        jnp.asarray(a),
        jnp.asarray(b),
        jnp.asarray(a @ x_star @ b),
        jnp.asarray(rng.standard_normal((m, n))),
        jnp.asarray(x_star),
    )


def loss(x, a, b, c):
    return 0.5 * jnp.sum((a @ x @ b - c) ** 2)


def dense_hessian(a, b):
    """``(B B^T) kron (A^T A)`` under column-stacking vec, matching Section 5.2."""
    return jnp.kron(b @ b.T, a.T @ a)


def vec(x):
    """Column-stacking vec, so that ``(M kron N) vec(X) = vec(N X M^T)``."""
    return x.T.reshape(-1)
