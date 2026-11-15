"""Three-signal fusion, g_A gating, and C–α decoupling (Sec. 3.3–3.4)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from mdl.constants import ALPHA_FLOOR, ALPHA_WEIGHT, BIAS, NMAX, NMIN
from mdl.encoding import value_encode
from mdl.subspaces import Subspaces, subspace_weights

jax.config.update("jax_enable_x64", True)


def risk_gate(va) -> jnp.ndarray:
    """g_A = 0.5 + 0.5 * mean(v_a)  (dimension-wise mean of the encoded risk vector)."""
    return 0.5 + 0.5 * jnp.mean(va)


def cosine(a, b, eps: float = 1e-12) -> jnp.ndarray:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    na = jnp.linalg.norm(a)
    nb = jnp.linalg.norm(b)
    return jnp.where((na < eps) | (nb < eps), 0.0, jnp.dot(a, b) / (na * nb))


def gated_confidence(C, alpha) -> jnp.ndarray:
    """C_final = C * (0.3 + 0.7 α)."""
    C = jnp.asarray(C, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    return C * (ALPHA_FLOOR + ALPHA_WEIGHT * alpha)


def confidence_consistency(vmeta, vwm, nmin: float = NMIN, nmax: float = NMAX):
    """Return (C, α, C_final) from fused and relevance vectors."""
    vmeta = jnp.asarray(vmeta, dtype=jnp.float64)
    vwm = jnp.asarray(vwm, dtype=jnp.float64)
    n = jnp.linalg.norm(vmeta)
    denom = jnp.asarray(nmax - nmin, dtype=jnp.float64)
    C = jnp.clip((n - nmin) / denom, 0.0, 1.0)
    alpha = jnp.clip(cosine(vwm, vmeta), -1.0, 1.0)
    Cfinal = gated_confidence(C, alpha)
    return C, alpha, Cfinal


def fuse(
    M: float,
    R: float,
    A: float,
    sub: Subspaces,
    bias: float = BIAS,
):
    """Encode (M, R, A), apply W_k, risk-gate, and tanh-fuse to v_meta."""
    vwm = value_encode(M)
    vr = value_encode(R)
    va = value_encode(1.0 - A)  # risk inversion s_a = 1 - A
    gA = risk_gate(va)
    W = subspace_weights(sub)
    b = jnp.asarray(bias, dtype=jnp.float64)
    z = gA * (W["wm"] @ vwm) + (W["r"] @ vr) + (W["a"] @ va) + b
    vmeta = jnp.tanh(z)
    return {
        "vwm": vwm,
        "vr": vr,
        "va": va,
        "gA": gA,
        "vmeta": vmeta,
        "va_norm": jnp.linalg.norm(va),
    }
