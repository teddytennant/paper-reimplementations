"""Range remedies: percentile clip scale and symmetric integer quantization."""

from __future__ import annotations

import jax.numpy as jnp


def percentile_scale(x, p=99.9):
    """Clip magnitude: the ``p``-th percentile of ``|x|`` (default 99.9)."""
    return jnp.percentile(jnp.abs(x), p)


def quantize(x, bits, alpha=None):
    """Symmetric signed quantization to ``bits`` bits.

    The integer grid is ``[-(2^{bits-1}-1), 2^{bits-1}-1]`` so the range is
    symmetric about zero (int8 uses 127, not 128). ``alpha`` is the full-scale
    magnitude; it defaults to ``max(|x|)``. Values outside ``[-alpha, alpha]``
    are clipped, then uniformly quantized.

    Returns ``(codes, step)`` with reconstruction ``codes * step`` and
    ``step = alpha / qmax``.
    """
    x = jnp.asarray(x)
    qmax = (1 << (bits - 1)) - 1
    if alpha is None:
        alpha = jnp.max(jnp.abs(x))
    alpha = jnp.asarray(alpha, dtype=jnp.result_type(x, jnp.float32))
    step = alpha / jnp.asarray(qmax, dtype=alpha.dtype)
    step = jnp.where(step == 0, jnp.ones_like(step), step)
    codes = jnp.round(x / step)
    codes = jnp.clip(codes, -qmax, qmax).astype(jnp.int32)
    return codes, step


def dequantize(codes, step, dtype=None):
    """Map symmetric integer codes back to reals."""
    out_dtype = jnp.float32 if dtype is None else dtype
    return jnp.asarray(codes, dtype=out_dtype) * jnp.asarray(step, dtype=out_dtype)
