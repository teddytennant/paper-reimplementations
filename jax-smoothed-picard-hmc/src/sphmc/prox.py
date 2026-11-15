"""Proximal maps: closed-form quadratic prox and gradient-only Algorithm D.2."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def prox_quadratic(y: Array, A: Array, eta: float) -> Array:
    """prox_{η V}(y) for V(x) = (1/2) x^T A x, A SPD: (I + η A)^{-1} y.

    Works for y of shape (d,) or (..., d).
    """
    y = jnp.asarray(y)
    A = jnp.asarray(A)
    d = A.shape[0]
    eye = jnp.eye(d, dtype=y.dtype)
    system = eye + eta * A
    if y.ndim == 1:
        return jnp.linalg.solve(system, y)
    return jnp.linalg.solve(system, y.T).T


def prox_gradient_descent(
    y: Array,
    grad_fn: Callable[[Array], Array],
    eta: float,
    eps_prox: float = 1e-6,
    max_iter: int = 1024,
) -> Array:
    """Algorithm D.2: gradient-only approximate prox.

    x(0) = y,  x(k+1) = y - η ∇V(x(k)) until ||x(k+1) - x(k)|| ≤ (1-η) ε_prox.
    Returns x(k). Requires 0 < η < 1 when ∇V is 1-Lipschitz (paper scaling β = 1).
    """
    y = jnp.asarray(y)
    thresh = (1.0 - eta) * eps_prox

    def cond(state):
        _x, k, converged = state
        return (k < max_iter) & (~converged)

    def body(state):
        x, k, _converged = state
        x_next = y - eta * grad_fn(x)
        converged = jnp.linalg.norm(x_next - x) <= thresh
        x_out = jnp.where(converged, x, x_next)
        return x_out, k + 1, converged

    x0 = y
    x_final, _, _ = jax.lax.while_loop(
        cond,
        body,
        (x0, jnp.asarray(0), jnp.asarray(False)),
    )
    return x_final
