"""Discrete fiberwise-OT interval allocation and piecewise-linear Φ."""

from __future__ import annotations

import jax.numpy as jnp

LAMBDA_DDPM = 220.0
LAMBDA_FM = 450.0


def allocate_weights(L, r, lam):
    """w_k = L_k √(λ r_k) / Z, Z = Σ_j L_j √(λ r_j). If Z = 0, w = L / Σ L."""
    L = jnp.asarray(L)
    r = jnp.asarray(r)
    lam = jnp.asarray(lam)
    scaled = L * jnp.sqrt(lam * jnp.clip(r, 0.0, None))
    Z = jnp.sum(scaled)
    L_sum = jnp.sum(L)
    w_fallback = L / jnp.where(L_sum == 0.0, 1.0, L_sum)
    w = jnp.where(Z == 0.0, w_fallback, scaled / Z)
    return w


def discrete_action(L, w, r, lam):
    """J(w) = Σ_k (L_k² / w_k + λ w_k r_k)."""
    L = jnp.asarray(L)
    w = jnp.asarray(w)
    r = jnp.asarray(r)
    lam = jnp.asarray(lam)
    return jnp.sum(jnp.square(L) / w + lam * w * r)


def phi_from_weights(w):
    """Piecewise-linear Φ, Φ⁻¹ from w on a uniform grid of the reference coordinate.

    Returns (tau_edges, t_edges) of length K+1 for w of length K, with
    tau uniform on [0, 1] and t the cumulative schedule-time allocation.
    """
    w = jnp.asarray(w)
    K = w.shape[0]
    tau_edges = jnp.linspace(0.0, 1.0, K + 1)
    t_edges = jnp.concatenate([jnp.zeros((1,), dtype=w.dtype), jnp.cumsum(w)])
    return tau_edges, t_edges


def apply_phi(tau, tau_edges, t_edges):
    """Φ(τ): reference coordinate → schedule time."""
    return jnp.interp(jnp.asarray(tau), tau_edges, t_edges)


def apply_phi_inv(t, tau_edges, t_edges):
    """Φ⁻¹(t): schedule time → reference coordinate."""
    return jnp.interp(jnp.asarray(t), t_edges, tau_edges)
