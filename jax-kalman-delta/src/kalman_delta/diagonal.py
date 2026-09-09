"""Diagonal KDN: sequential step, variational p, and Möbius scan (eq 23-38)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def predict_variance(p: jax.Array, alpha: jax.Array, omega: jax.Array) -> jax.Array:
    """p̂ = α² ⊙ p + ω (eq 23)."""
    return alpha * alpha * p + omega


def diagonal_gain(p_hat: jax.Array, k: jax.Array, r: jax.Array | float) -> jax.Array:
    """κ = (p̂ ⊙ k) / (r + sum_i p̂_i k_i²) (eq 24 / 38)."""
    denom = r + jnp.dot(p_hat, k * k)
    return (p_hat * k) / denom


def variational_diagonal_p(
    p_hat: jax.Array,
    k: jax.Array,
    r: jax.Array | float,
    mu: jax.Array | float = 1.0,
) -> jax.Array:
    """Diagonal variational posterior p = p̂ / (1 + u ⊙ p̂), u = (μ/r) k⊙k.

    With μ = 1 this is (p̂^{-1} + k²/r)^{-1} coordinatewise (eq 30-31, 34).
    """
    u = (mu / r) * (k * k)
    return p_hat / (1.0 + u * p_hat)


def information_scale(k: jax.Array, r: jax.Array | float, mu: jax.Array | float = 1.0) -> jax.Array:
    """u = (μ / r) k ⊙ k."""
    return (mu / r) * (k * k)


def diagonal_step(
    S: jax.Array,
    p: jax.Array,
    alpha: jax.Array,
    omega: jax.Array,
    k: jax.Array,
    v: jax.Array,
    r: jax.Array | float,
    mu: jax.Array | float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One Diagonal KDN step: mean matches exact Kalman given diagonal P̂.

    Returns S, p, kappa, p_hat. Gain uses p̂, not the projected p (eq 38).
    """
    p_hat = predict_variance(p, alpha, omega)
    kappa = diagonal_gain(p_hat, k, r)
    S_hat = alpha[:, None] * S
    innov = v - jnp.dot(jnp.swapaxes(S_hat, -1, -2), k)
    S_new = S_hat + jnp.outer(kappa, innov)
    p_new = variational_diagonal_p(p_hat, k, r, mu=mu)
    return S_new, p_new, kappa, p_hat


def mobius_matrices(alpha: jax.Array, omega: jax.Array, u: jax.Array) -> jax.Array:
    """Per-channel 2x2 Möbius matrices M_t (eq 33).

    Args:
        alpha, omega, u: (T, d_k)

    Returns:
        M: (T, d_k, 2, 2) with [[α², ω], [u α², 1 + u ω]].
    """
    a2 = alpha * alpha
    m00 = a2
    m01 = omega
    m10 = u * a2
    m11 = 1.0 + u * omega
    row0 = jnp.stack([m00, m01], axis=-1)
    row1 = jnp.stack([m10, m11], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def _assoc_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
    """Associative operator so prefix_t = M_t ... M_0."""
    return jnp.matmul(right, left)


def mobius_scan(
    alpha: jax.Array,
    omega: jax.Array,
    u: jax.Array,
    n0: jax.Array | float = 1.0,
    d0: jax.Array | float = 1.0,
) -> jax.Array:
    """Associative scan of Möbius maps; returns p_t = n_t / d_t (eq 32-33).

    Args:
        alpha, omega, u: (T, d_k)
        n0, d0: homogeneous prior, default [1, 1] so p_0 = 1.

    Returns:
        p: (T, d_k) posterior diagonal variances after each token.
    """
    M = mobius_matrices(alpha, omega, u)
    prefix = lax.associative_scan(_assoc_matmul, M, axis=0)
    nd0 = jnp.stack([jnp.asarray(n0, dtype=M.dtype), jnp.asarray(d0, dtype=M.dtype)])
    nd = prefix @ nd0
    return nd[..., 0] / nd[..., 1]


def sequential_p(
    alpha: jax.Array,
    omega: jax.Array,
    u: jax.Array,
    p0: jax.Array | float = 1.0,
) -> jax.Array:
    """Sequential Möbius recurrence (eq 32), same inputs as ``mobius_scan``."""

    def body(p, inputs):
        a_t, w_t, u_t = inputs
        a2 = a_t * a_t
        p_new = (a2 * p + w_t) / (u_t * a2 * p + 1.0 + u_t * w_t)
        return p_new, p_new

    p0 = jnp.broadcast_to(jnp.asarray(p0, dtype=alpha.dtype), alpha.shape[-1])
    _, p = lax.scan(body, p0, (alpha, omega, u))
    return p
