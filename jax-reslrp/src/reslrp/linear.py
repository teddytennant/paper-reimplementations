"""Dense LRP-ε (z-rule) so a residual MLP can be explained."""

from __future__ import annotations

import jax.numpy as jnp


def lrp_eps(x, w, b, r_out, eps: float = 1e-9, conserve: bool = True):
    """Standard dense LRP-ε.

    y = x @ w + b, with w shaped (in, out). Relevance of output neurons is
    redistributed as R_i = sum_j (x_i w_ij) / (y_j + ε sign(y_j)) R_j.
    If ``conserve`` is True, leftover mass from bias and ε is redistributed
    over the input so sum(R_in) = sum(R_out) per batch row.
    ``b`` may be None.
    """
    x = jnp.asarray(x)
    w = jnp.asarray(w)
    r_out = jnp.asarray(r_out)
    y = x @ w
    if b is not None:
        y = y + jnp.asarray(b)
    sign = jnp.sign(y)
    sign = jnp.where(sign == 0, jnp.ones_like(sign), sign)
    denom = y + jnp.asarray(eps, dtype=x.dtype) * sign
    contrib = x[..., :, None] * w
    r_in = jnp.sum(contrib / denom[..., None, :] * r_out[..., None, :], axis=-1)
    if conserve:
        r_out_sum = jnp.sum(r_out, axis=-1, keepdims=True)
        r_in_sum = jnp.sum(r_in, axis=-1, keepdims=True)
        leftover = r_out_sum - r_in_sum
        mass = jnp.sum(jnp.abs(r_in), axis=-1, keepdims=True)
        n = jnp.asarray(r_in.shape[-1], dtype=x.dtype)
        share = jnp.where(mass > 0, jnp.abs(r_in) / mass, 1.0 / n)
        r_in = r_in + leftover * share
    return r_in
