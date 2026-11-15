"""Adaptive node allocation (paper Sec. 3.3 / Eq. 6).

I(x) = σ(η_θ(Concat(P, Q̃))(x))
o_i  = tanh(ψ_θ(P)_i)
w_i  = (Imin + s_I I_i) / (sum_k (Imin + s_I I_k) + ε)
Δx_i = δ_max * w_i * o_i
x^a_i = Π_Ω (x^r_i + Δx_i)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def importance_from_logits(logits: Array) -> Array:
    """I = σ(·)."""
    return jax.nn.sigmoid(logits)


def direction_from_logits(logits: Array) -> Array:
    """o = tanh(·)."""
    return jnp.tanh(logits)


def importance_weights(
    importance: Array,
    i_min: float = 0.05,
    s_i: float = 1.0,
    eps: float = 1e-8,
) -> Array:
    """w_i = (Imin + s_I I_i) / (sum_k (Imin + s_I I_k) + ε)."""
    num = i_min + s_i * importance
    return num / (jnp.sum(num) + eps)


def displacements(weights: Array, direction: Array, delta_max: float) -> Array:
    """Δx_i = δ_max * w_i * o_i (broadcast w over the spatial dim of o)."""
    w = weights if weights.ndim == direction.ndim else weights[..., None]
    return delta_max * w * direction


def project_domain(
    x: Array,
    omega_min: float = 0.0,
    omega_max: float = 1.0,
) -> Array:
    """Π_Ω: axis-aligned box projection by clipping."""
    return jnp.clip(x, omega_min, omega_max)


def allocate(
    x_r: Array,
    importance: Array,
    direction: Array,
    *,
    i_min: float = 0.05,
    s_i: float = 1.0,
    delta_max: float = 0.08,
    eps: float = 1e-8,
    omega_min: float = 0.0,
    omega_max: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Return (x^a, w, Δx) from Eq. 6 given I and o."""
    w = importance_weights(importance, i_min=i_min, s_i=s_i, eps=eps)
    dx = displacements(w, direction, delta_max)
    x_a = project_domain(x_r + dx, omega_min=omega_min, omega_max=omega_max)
    return x_a, w, dx


def physical_indicators(field: Array) -> Array:
    """Q̃ channels used by the kernel: gradient magnitude and |Laplacian|.

    ``field`` is (H, W) or (H, W, C). Returns (H, W, 2), min-max normalised
    per channel (the paper's N(·)).
    """
    if field.ndim == 2:
        field = field[..., None]
    gy = jnp.gradient(field, axis=0)
    gx = jnp.gradient(field, axis=1)
    grad_mag = jnp.sqrt(jnp.mean(gx * gx + gy * gy, axis=-1, keepdims=True))
    lap = jnp.gradient(gy, axis=0) + jnp.gradient(gx, axis=1)
    lap_mag = jnp.mean(jnp.abs(lap), axis=-1, keepdims=True)
    q = jnp.concatenate([grad_mag, lap_mag], axis=-1)
    qmin = jnp.min(q, axis=(0, 1), keepdims=True)
    qmax = jnp.max(q, axis=(0, 1), keepdims=True)
    return (q - qmin) / (qmax - qmin + 1e-8)
