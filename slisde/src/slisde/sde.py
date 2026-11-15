"""Structured linear SDE layer: affine Itô dynamics and one-step maps (Eqs. 2, 6, 77, 78)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg


def init_layer(key, d, m, *, structure="dense", noise="additive", scale=0.1):
    """Trainable coefficients (A, b, C^j, D) of Eq. 6.

    structure: "dense" | "diagonal"
    noise: "additive" (C=0) | "multiplicative" (D=0) | "affine"
    """
    kA, kb, kC, kD = jax.random.split(key, 4)
    if structure == "diagonal":
        A = -0.5 * jnp.eye(d) + scale * jnp.diag(jax.random.normal(kA, (d,)))
        C = scale * jnp.zeros((m, d, d))
        if noise != "additive":
            C = C + scale * jax.vmap(jnp.diag)(jax.random.normal(kC, (m, d)))
    else:
        A = -0.5 * jnp.eye(d) + scale * jax.random.normal(kA, (d, d))
        C = scale * jax.random.normal(kC, (m, d, d))
        if noise == "additive":
            C = jnp.zeros_like(C)
    b = scale * jax.random.normal(kb, (d,))
    D = scale * jax.random.normal(kD, (d, m))
    if noise == "multiplicative":
        D = jnp.zeros_like(D)
    return {"A": A, "b": b, "C": C, "D": D}


def _sum_C_dW(C, dW):
    """sum_j C^j ΔW^j, shape (B, N, d, d)."""
    return jnp.einsum("mij,bnm->bnij", C, dW)


def _D_dW(D, dW):
    """D ΔW, shape (B, N, d)."""
    return jnp.einsum("dm,bnm->bnd", D, dW)


def _C_squared_sum(C):
    """sum_j (C^j)^2 = sum_j C^j C^j."""
    return jnp.einsum("mij,mjk->ik", C, C)


def euler_maruyama_pairs(A, b, C, D, dW, dt):
    """Euler–Maruyama affine pair (Eq. 78).

    F_k = I + A Δt + sum_j C^j ΔW^j_k
    g_k = b Δt + D ΔW_k
    """
    d = A.shape[0]
    eye = jnp.eye(d, dtype=A.dtype)
    F = eye + A * dt + _sum_C_dW(C, dW)
    g = b * dt + _D_dW(D, dW)
    return F, g


def exponential_pairs(A, b, C, D, dW, dt):
    """Matrix-exponential affine pair (Eq. 77).

    F_k = exp((A - 1/2 sum_j (C^j)^2) Δt + sum_j C^j ΔW^j_k)
    g_k = b Δt + D ΔW_k
    """
    M = (A - 0.5 * _C_squared_sum(C)) * dt + _sum_C_dW(C, dW)
    F = jax.vmap(jax.vmap(jax.scipy.linalg.expm))(M)
    g = b * dt + _D_dW(D, dW)
    return F, g


def transition_pairs(layer, dW, dt, method="euler"):
    """Build (F_k, g_k) from layer coefficients and Brownian increments."""
    A, b, C, D = layer["A"], layer["b"], layer["C"], layer["D"]
    if method == "euler":
        return euler_maruyama_pairs(A, b, C, D, dW, dt)
    if method == "expm":
        return exponential_pairs(A, b, C, D, dW, dt)
    raise ValueError(f"unknown method {method}")


def sample_increments(key, batch, n_steps, m, dt):
    """ΔW_k ~ N(0, Δt I_m)."""
    return jax.random.normal(key, (batch, n_steps, m)) * jnp.sqrt(dt)


def ou_mean(z0, kappa, t):
    """Exact mean of dX = -κ X dt + σ dW."""
    return z0 * jnp.exp(-kappa * t)


def ou_var(sigma, kappa, t):
    """Exact variance of dX = -κ X dt + σ dW from X_0 deterministic."""
    return (sigma**2) / (2.0 * kappa) * (1.0 - jnp.exp(-2.0 * kappa * t))


def affine_additive_mean(z0, a, b, t):
    """Exact mean of dX = (a X + b) dt + σ dW."""
    ea = jnp.exp(a * t)
    return jnp.where(jnp.abs(a) < 1e-12, z0 + b * t, ea * z0 + (b / a) * (ea - 1.0))


def affine_additive_var(sigma, a, t, v0=0.0):
    """Exact variance of dX = (a X + b) dt + σ dW."""
    e2 = jnp.exp(2.0 * a * t)
    integral = jnp.where(jnp.abs(a) < 1e-12, t, (e2 - 1.0) / (2.0 * a))
    return e2 * v0 + (sigma**2) * integral


def em_moments_1d(z0, a, b, sigma, dt, n_steps):
    """Exact mean and variance of the 1D additive Euler–Maruyama chain."""
    m = z0
    v = 0.0
    F = 1.0 + a * dt
    for _ in range(n_steps):
        m = F * m + b * dt
        v = (F**2) * v + (sigma**2) * dt
    return m, v
