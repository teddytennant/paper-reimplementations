"""Closed-form Kalman predict / update over a denoiser-output window.

Paper eqs 10–16 in the task restatement; main-text eqs 19, 21, 23 and
Algorithm 1. Process-residual mean `μ` is applied as `G μ` rather than being
folded into `b` (see README).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from quake.transition import observation_row


def predict(mean, cov, A, b, G, mu, Q) -> tuple[Array, Array]:
    """Predictive mean and covariance.

    `m_{i|i-1} = A m_{i-1|i-1} + b + G μ`
    `P_{i|i-1} = A P A^⊤ + G Q G^⊤`
    """
    mean = jnp.asarray(mean)
    cov = jnp.asarray(cov)
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    G = jnp.asarray(G)
    mu = jnp.asarray(mu)
    Q = jnp.asarray(Q)
    mean_pred = (A @ mean[..., None]).squeeze(-1) + b + G * mu[..., None]
    GGT = G[..., None] * G[..., None, :]
    cov_pred = A @ cov @ jnp.swapaxes(A, -1, -2) + Q[..., None, None] * GGT
    return mean_pred, cov_pred


def update(mean, cov, observation, H_row, xi, R) -> tuple[Array, Array, Array]:
    """Observation update (current-slot row `H = γ e_1^⊤` or a general row).

    `K = P H^⊤ (H P H^⊤ + R)^{-1}`
    `m = m + K (f̂ − H m − ξ)`
    `P = (I − K H) P`
    """
    mean = jnp.asarray(mean)
    cov = jnp.asarray(cov)
    observation = jnp.asarray(observation)
    H_row = jnp.broadcast_to(jnp.asarray(H_row), mean.shape)
    xi = jnp.asarray(xi)
    R = jnp.asarray(R)
    window = mean.shape[-1]
    Hm = jnp.sum(H_row * mean, axis=-1)
    HP = jnp.einsum("...i,...ij->...j", H_row, cov)
    S = jnp.sum(HP * H_row, axis=-1) + R
    K = HP / S[..., None]
    innov = observation - Hm - xi
    mean_post = mean + K * innov[..., None]
    eye = jnp.eye(window, dtype=mean.dtype)
    KH = K[..., None] * H_row[..., None, :]
    cov_post = (eye - KH) @ cov
    return mean_post, cov_post, K


def step(
    mean,
    cov,
    f_hat,
    A,
    b,
    G,
    mu,
    Q,
    gamma,
    xi,
    R,
    *,
    do_predict: bool = True,
) -> tuple[Array, Array, Array]:
    """One predict (optional) + update; returns posterior and current-slot mean."""
    if do_predict:
        mean, cov = predict(mean, cov, A, b, G, mu, Q)
    window = mean.shape[-1]
    H_row = observation_row(gamma, window)
    mean, cov, _K = update(mean, cov, f_hat, H_row, xi, R)
    return mean, cov, mean[..., 0]
