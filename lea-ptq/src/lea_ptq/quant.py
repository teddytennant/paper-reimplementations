"""Symmetric per-row quantization."""

import jax.numpy as jnp


def symmetric_quantize(W, bits):
    """Quantize W with a symmetric uniform grid and dequantize.

    W is shape (out, in) or (n,). bits must be an integer >= 2.
    qmax = 2**(bits-1) - 1. A 2d matrix is scaled per row (axis 0).
    A 1d vector uses one scale for the whole vector.
    scale = max(abs(row)) / qmax, or 1.0 when the row max is 0.
    Integer codes use jnp.round (JAX default). For the half value -1.5,
    jnp.round returns -2.0. Codes are clipped to [-qmax, qmax] and
    multiplied by the scale. A zero row stays zero.
    """
    if isinstance(bits, bool) or not isinstance(bits, (int, float)):
        raise ValueError("bits must be a real number >= 2")
    b = int(bits)
    if b < 2 or float(bits) != float(b):
        raise ValueError("bits must be an integer >= 2")

    W = jnp.asarray(W)
    if W.ndim not in (1, 2):
        raise ValueError("W must have shape (out, in) or (n,)")

    qmax = 2 ** (b - 1) - 1
    abs_w = jnp.abs(W)
    if W.ndim == 2:
        row_max = jnp.max(abs_w, axis=-1, keepdims=True)
    else:
        row_max = jnp.max(abs_w)

    scale = jnp.where(row_max == 0, jnp.asarray(1.0, dtype=W.dtype), row_max / qmax)
    codes = jnp.clip(jnp.round(W / scale), -qmax, qmax)
    return codes * scale
