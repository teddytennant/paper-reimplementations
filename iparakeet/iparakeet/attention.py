"""Relative shift and integer score fusion (arXiv 2609.30846, eq. 10 and eq. 11).

Eq. 10 applies ``floor`` to the real-valued fused sum of scaled integer content
and position scores. Eq. 11 replaces those real coefficients by dyadic integers
and an arithmetic right shift.

The shift uses round-half-away-from-zero (not round-half-up, and not the
positive-only ``add 2^(n-1)`` bias, which is wrong for negatives). Multiplier
rounding is half-to-even. The accumulator is int32.
Softmax is left in float; the paper defers that kernel to I-BERT and this code
does not rederive it.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from iparakeet.range import quantize


def rel_shift(x):
    """Transformer-XL relative shift Phi.

    ``x`` has shape ``(..., qlen, 2*klen-1)``. The result has shape
    ``(..., qlen, klen)``. The map only permutes entries and inserts zeros, so
    ``Phi(S * x) == S * Phi(x)`` for a scalar ``S``. Zeros keep ``x.dtype``, so
    an integer tensor stays integer (Phi is applied before dequantization).
    """
    x = jnp.asarray(x)
    qlen = x.shape[-2]
    klen = (x.shape[-1] + 1) // 2
    zero = jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    xp = jnp.concatenate([zero, x], axis=-1)  # (..., qlen, 2*klen)
    xp = jnp.reshape(xp, x.shape[:-2] + (2 * klen, qlen))
    xp = xp[..., 1:, :]
    xp = jnp.reshape(xp, x.shape[:-2] + (qlen, 2 * klen - 1))
    return xp[..., :klen]


phi = rel_shift


def _rounding_right_shift(x, n):
    """Arithmetic right shift by ``n`` with round-half-away-from-zero.

    For a positive value this is the usual ``(x + 2^(n-1)) >> n``. Negatives
    use the same bias on the magnitude, then restore the sign, so halves move
    away from zero rather than toward +infinity. The accumulator is int32,
    which is JAX's default integer width and holds the n=16 products of int8
    scores used here.
    """
    x = jnp.asarray(x, dtype=jnp.int32)
    bias = jnp.int32(1 << (n - 1))
    magnitude = jnp.right_shift(jnp.abs(x) + bias, n)
    return jnp.where(x >= 0, magnitude, -magnitude)


def fuse_scores_reference(qc, qp, Sc, Sp, Ss, dk):
    """Eq. 10. Floor of the real-valued fused sum.

    ``qc`` is ``(..., qlen, klen)`` and ``qp`` is the unshifted relative score
    ``(..., qlen, 2*klen-1)``. Phi is applied to ``qp`` while it is still an
    integer tensor. The paper writes a floor on that fused real sum; this
    function uses ``jnp.floor``.
    """
    qc_f = jnp.asarray(qc, dtype=jnp.float32)
    qp_f = jnp.asarray(rel_shift(qp), dtype=jnp.float32)
    inv = jnp.asarray(Ss, dtype=jnp.float32) * jnp.sqrt(jnp.asarray(dk, dtype=jnp.float32))
    fused = (jnp.asarray(Sc, dtype=jnp.float32) / inv) * qc_f
    fused = fused + (jnp.asarray(Sp, dtype=jnp.float32) / inv) * qp_f
    return jnp.floor(fused)


def fuse_scores_dyadic(qc, qp, Sc, Sp, Ss, dk, n=16):
    """Eq. 11. Dyadic fusion with a rounding arithmetic right shift.

    ``mc = round(2^n * Sc / (Ss * sqrt(dk)))`` and likewise for ``mp``.
    Multipliers are rounded half-to-even in host float64 (Python ``round``),
    then the int32 shift of ``mc * qc + mp * Phi(qp)`` by ``n`` is
    round-half-away-from-zero.
    """
    inv = float(jnp.asarray(Ss)) * math.sqrt(float(jnp.asarray(dk)))
    two_n = 2.0 ** n
    mc = int(round(two_n * float(jnp.asarray(Sc)) / inv))
    mp = int(round(two_n * float(jnp.asarray(Sp)) / inv))
    shifted = rel_shift(qp)
    acc = jnp.int32(mc) * jnp.asarray(qc, dtype=jnp.int32)
    acc = acc + jnp.int32(mp) * jnp.asarray(shifted, dtype=jnp.int32)
    return _rounding_right_shift(acc, n)


def _content_and_position(Q, K, P, u, v):
    content = (Q + u) @ jnp.swapaxes(K, -1, -2)
    position = (Q + v) @ jnp.swapaxes(P, -1, -2)
    return content, position


def _unpack_scales(scales):
    if isinstance(scales, dict):
        return scales["Sc"], scales["Sp"], scales["Ss"]
    Sc, Sp, Ss = scales
    return Sc, Sp, Ss


def float_attention(Q, K, V, P, u, v):
    """Float relative attention. ``P`` is ``(..., 2*klen-1, dk)``."""
    dk = Q.shape[-1]
    content, position = _content_and_position(Q, K, P, u, v)
    logits = (content + rel_shift(position)) / jnp.sqrt(jnp.asarray(dk, dtype=Q.dtype))
    return jax.nn.softmax(logits, axis=-1) @ V


def activation_scales(Q, K, V, P, u, v, bits=8):
    """Symmetric full-scale steps for content, position, and fused logits.

    ``V`` is unused; it is accepted so the helper matches the attention
    signature. Steps are ``max(abs) / (2^{bits-1}-1)``.
    """
    del V
    qmax = (1 << (bits - 1)) - 1
    content, position = _content_and_position(Q, K, P, u, v)
    dk = Q.shape[-1]
    logits = (content + rel_shift(position)) / jnp.sqrt(jnp.asarray(dk, dtype=Q.dtype))

    def step(t):
        magnitude = jnp.max(jnp.abs(t))
        fallback = jnp.ones((), dtype=t.dtype)
        return jnp.where(magnitude > 0, magnitude / qmax, fallback)

    return {"Sc": step(content), "Sp": step(position), "Ss": step(logits)}


def integer_attention(Q, K, V, P, u, v, scales):
    """Int8 score quant, eq. 10 fusion, float dequant, float softmax, times ``V``.

    ``qc`` and ``qp`` are symmetric int8 codes of the content and unshifted
    position scores, with steps ``Sc`` and ``Sp``. Eq. 10 fuses those integers
    (Phi runs on ``qp``, not on a dequantized tensor). Logits are ``qs * Ss``.
    Softmax stays in float.
    """
    Sc, Sp, Ss = _unpack_scales(scales)
    dk = Q.shape[-1]
    content, position = _content_and_position(Q, K, P, u, v)
    qmax = 127
    qc, _ = quantize(content, 8, alpha=jnp.asarray(Sc, dtype=content.dtype) * qmax)
    qp, _ = quantize(position, 8, alpha=jnp.asarray(Sp, dtype=position.dtype) * qmax)
    qs = fuse_scores_reference(qc, qp, Sc, Sp, Ss, dk)
    logits = qs.astype(Q.dtype) * jnp.asarray(Ss, dtype=Q.dtype)
    return jax.nn.softmax(logits, axis=-1) @ V
