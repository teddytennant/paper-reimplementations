"""Primitive maps from the HeRo method (eqs. 2, 3, 10-12, 16, 18)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def linear(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Apply W x with W in R^{out x in} on a trailing feature axis."""
    return jnp.einsum("...i,oi->...o", x, weight)


def phi(x: jax.Array) -> jax.Array:
    """Positive feature map phi(x) = ELU(x) + 1 (eq. 11).

    ELU(x)+1 equals x+1 for x>=0 and exp(x) for x<0. The branched form
    avoids float32 cancellation of (exp(x)-1)+1 for large negative x.
    """
    return jnp.where(x >= 0.0, x + 1.0, jnp.exp(x))


def cosine_similarity(x: jax.Array, y: jax.Array, eps: float) -> jax.Array:
    """Cosine along the last axis, with eps added after the product of norms (eq. 8)."""
    dot = jnp.sum(x * y, axis=-1)
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    return dot / (x_norm * y_norm + eps)


def memory_read(
    phi_q: jax.Array,
    s_prev: jax.Array,
    zeta_prev: jax.Array,
    eps: float,
) -> jax.Array:
    """Read-before-write context (eq. 11).

    c = (phi(q)^T S_{j-1}) / (phi(q)^T zeta_{j-1} + eps)
    """
    numer = jnp.einsum("...i,...ij->...j", phi_q, s_prev)
    denom = jnp.einsum("...i,...i->...", phi_q, zeta_prev) + eps
    return numer / denom[..., None]


def memory_write(
    s_prev: jax.Array,
    zeta_prev: jax.Array,
    phi_k: jax.Array,
    v: jax.Array,
    rho: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Kernelized linear-attention write with forgetting (eq. 12).

    S_j = rho S_{j-1} + phi(k) v^T
    zeta_j = rho zeta_{j-1} + phi(k)
    """
    rho = jnp.asarray(rho)
    while rho.ndim < s_prev.ndim - 1:
        rho = rho[..., None]
    outer = phi_k[..., :, None] * v[..., None, :]
    s_new = rho[..., None] * s_prev + outer
    zeta_new = rho * zeta_prev + phi_k
    return s_new, zeta_new


def routed_residual(
    hbar: jax.Array,
    gate: jax.Array,
    ffn_out: jax.Array,
    adapter_out: jax.Array,
    tau_exec: float,
) -> jax.Array:
    """Thresholded single-branch residual (eq. 3).

    if g >= tau_exec: h = hbar + g * FFN(u)
    else:             h = hbar + (1-g) * Adapter(u)

    The comparison is not differentiated. Scaling the chosen branch by g or
    (1-g) still carries gradients to the router.
    """
    gate = gate[..., None]
    use_ffn = jax.lax.stop_gradient(gate >= tau_exec)
    chosen = jnp.where(use_ffn, gate * ffn_out, (1.0 - gate) * adapter_out)
    return hbar + chosen


def skip_loss(gates: jax.Array) -> jax.Array:
    """Layer-usage penalty (eq. 18).

    L_skip = 1/(B*(T-1)) sum_b sum_t (sum_j g_{b,lj,t})^2

    ``gates`` has shape (B, T_pred, L_R) where T_pred is the number of
    next-token positions (T-1 in a length-T sequence).
    """
    per_token_sum = jnp.sum(gates, axis=-1)
    return jnp.mean(jnp.square(per_token_sum))


def rms_norm(x: jax.Array, scale: jax.Array, eps: float) -> jax.Array:
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


def layer_norm(
    x: jax.Array, scale: jax.Array, bias: jax.Array, eps: float
) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias
