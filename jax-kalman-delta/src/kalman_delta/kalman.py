"""Exact Kalman associative memory (Proposition 3.1 / eq 17)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def read(S: jax.Array, q: jax.Array) -> jax.Array:
    """Query readout o = S^T q (eq 16)."""
    return jnp.dot(jnp.swapaxes(S, -1, -2), q)


def kalman_predict(
    S: jax.Array,
    P: jax.Array,
    D: jax.Array,
    Omega: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Predict Ŝ = D S and P̂ = D P D^T + Ω."""
    S_hat = D @ S
    P_hat = (D @ P) @ jnp.swapaxes(D, -1, -2) + Omega
    return S_hat, P_hat


def kalman_gain(P_hat: jax.Array, k: jax.Array, r: jax.Array | float) -> jax.Array:
    """κ = P̂ k / (r + k^T P̂ k) via explicit dots."""
    Pk = jnp.dot(P_hat, k)
    denom = r + jnp.dot(k, Pk)
    return Pk / denom


def kalman_update(
    S_hat: jax.Array,
    P_hat: jax.Array,
    k: jax.Array,
    v: jax.Array,
    r: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Posterior mean, covariance, and gain given a predictive state."""
    kappa = kalman_gain(P_hat, k, r)
    innov = v - jnp.dot(jnp.swapaxes(S_hat, -1, -2), k)
    S = S_hat + jnp.outer(kappa, innov)
    eye = jnp.eye(k.shape[-1], dtype=S_hat.dtype)
    P = (eye - jnp.outer(kappa, k)) @ P_hat
    return S, P, kappa


def kalman_step(
    S: jax.Array,
    P: jax.Array,
    D: jax.Array,
    Omega: jax.Array,
    k: jax.Array,
    v: jax.Array,
    r: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One exact Kalman associative-memory step (eq 17)."""
    S_hat, P_hat = kalman_predict(S, P, D, Omega)
    return kalman_update(S_hat, P_hat, k, v, r)


def kalman_filter(
    S0: jax.Array,
    P0: jax.Array,
    D: jax.Array,
    Omega: jax.Array,
    k: jax.Array,
    v: jax.Array,
    r: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sequential exact filter over a length-T sequence.

    Args:
        S0: (d_k, d_v) prior mean.
        P0: (d_k, d_k) prior covariance.
        D: (T, d_k, d_k) transitions.
        Omega: (T, d_k, d_k) process covariances.
        k: (T, d_k) keys.
        v: (T, d_v) values.
        r: (T,) observation noise.

    Returns:
        S: (T, d_k, d_v), P: (T, d_k, d_k), kappa: (T, d_k).
    """

    def body(carry, inputs):
        S, P = carry
        D_t, Omega_t, k_t, v_t, r_t = inputs
        S, P, kappa = kalman_step(S, P, D_t, Omega_t, k_t, v_t, r_t)
        return (S, P), (S, P, kappa)

    _, (S, P, kappa) = lax.scan(body, (S0, P0), (D, Omega, k, v, r))
    return S, P, kappa
