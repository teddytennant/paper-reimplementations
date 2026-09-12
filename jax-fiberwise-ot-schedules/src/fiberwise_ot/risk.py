"""Midpoint fiberwise-risk estimators for DDPM and flow matching."""

from __future__ import annotations

import jax.numpy as jnp

from fiberwise_ot.paths import alpha_sigma


def _midpoints(edges):
    edges = jnp.asarray(edges)
    return 0.5 * (edges[:-1] + edges[1:])


def ddpm_midpoint_risk(predict_eps, x0, eps, t_edges, soff: float = 0.008):
    """r_k = (1 − ᾱ(τ°)) E‖ε_θ(x_{τ°}, τ°) − ε‖² at interval midpoints (D.8).

    ``predict_eps(x, t)`` maps noisy states and times to ε-predictions.
    ``x0`` and ``eps`` are batched 1D samples, broadcast against the midpoint grid.
    """
    t_mid = _midpoints(t_edges)
    alpha, sigma = alpha_sigma(t_mid, soff=soff)
    x0_b = x0[:, None]
    eps_b = eps[:, None]
    t_b = t_mid[None, :]
    x = alpha[None, :] * x0_b + sigma[None, :] * eps_b
    eps_hat = predict_eps(x, jnp.broadcast_to(t_b, x.shape))
    mse = jnp.mean(jnp.square(eps_hat - eps_b), axis=0)
    return jnp.square(sigma) * mse


def fm_midpoint_risk(predict_v, x0, eps, tau_edges):
    """r_k = (τ(1 − τ))² E‖v_θ − (ε − x0)‖² at interval midpoints."""
    tau_mid = _midpoints(tau_edges)
    x0_b = x0[:, None]
    eps_b = eps[:, None]
    tau_b = tau_mid[None, :]
    x = (1.0 - tau_mid[None, :]) * x0_b + tau_mid[None, :] * eps_b
    target = eps_b - x0_b
    v_hat = predict_v(x, jnp.broadcast_to(tau_b, x.shape))
    mse = jnp.mean(jnp.square(v_hat - target), axis=0)
    return jnp.square(tau_mid * (1.0 - tau_mid)) * mse
