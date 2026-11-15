"""EFQ-Softmax score-to-E2M1 map (Algorithms 1 and II-D)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# Nonnegative E2M1 code set, paper II-B.
C8 = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# 16-to-8 folding table, paper II-D.
C16 = jnp.asarray([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7], dtype=jnp.int32)

# Paper IV-A operating points.
TAU_MMLU = -2.90
H_MMLU = 2.00
TAU_MEAN = -3.06
H_MEAN = 2.30
TAU_BALANCE = -2.10
H_BALANCE = 2.70

OPERATING_POINTS = {
    "mmlu": (TAU_MMLU, H_MMLU),
    "mean": (TAU_MEAN, H_MEAN),
    "balance": (TAU_BALANCE, H_BALANCE),
}

DEFAULT_BLOCK_SIZE = 32


class EFQOperand(NamedTuple):
    """Block-scaled E2M1 probability operand."""

    ptilde: Array
    codes: Array
    scale: Array
    k: Array
    z: Array
    j16: Array | None = None


def _partition_last(
    x: Array, block_size: int, pad_value: float = -jnp.inf
) -> tuple[Array, int]:
    """Reshape the last axis into microscaling blocks."""
    t = x.shape[-1]
    pad = (block_size - t % block_size) % block_size
    if pad:
        pads = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        x = jnp.pad(x, pads, constant_values=pad_value)
    n_blocks = x.shape[-1] // block_size
    return x.reshape(*x.shape[:-1], n_blocks, block_size), t


def _unpartition_last(x_b: Array, t: int) -> Array:
    flat = x_b.reshape(*x_b.shape[:-2], x_b.shape[-2] * x_b.shape[-1])
    return flat[..., :t]


def block_scale_and_residual(x: Array, block_size: int = DEFAULT_BLOCK_SIZE):
    """Exponent-only scale (eq 8-9) and residual z (eq 10) per microscaling block.

    ``x`` is a shifted-score tile with ``x <= 0`` on the last axis.
    """
    x_b, t = _partition_last(x, block_size)
    m_b = jnp.max(x_b, axis=-1)
    log2 = jnp.log(2.0)
    k_b = jnp.floor((m_b + jnp.log(2.0 / 9.0)) / log2)
    k_b = jnp.where(jnp.isfinite(m_b), k_b, jnp.zeros_like(k_b))
    s_b = jnp.exp2(k_b)
    z_b = x_b - k_b[..., None] * log2 - jnp.log(6.0)
    z = _unpartition_last(z_b, t)
    return x_b, m_b, k_b, s_b, z, t


def affine_codes(z: Array, tau: float, h: float) -> Array:
    """Integer E2M1 codes in {0..7} from residual z (eq 11)."""
    raw = jnp.floor((z - tau) * h) + 1.0
    return jnp.clip(raw, 0.0, 7.0).astype(jnp.int32)


def lut_index(z: Array) -> Array:
    """16-level intermediate index (paper II-D)."""
    tau_16 = jnp.log(1.0 / 24.0)
    h_16 = 14.0 / jnp.log(24.0)
    raw = jnp.floor((z - tau_16) * h_16) + 1.0
    return jnp.clip(raw, 0.0, 15.0).astype(jnp.int32)


def fold_lut(j16: Array) -> Array:
    """Monotone 16-to-8 folding ``C16[j16]``."""
    return C16[j16]


def _reconstruct(codes: Array, s_b: Array, block_size: int, t: int) -> Array:
    q = C8[codes]
    q_b, _ = _partition_last(q, block_size, pad_value=0.0)
    p_b = s_b[..., None] * q_b
    return _unpartition_last(p_b, t)


def efq_quantize(
    x: Array,
    tau: float = TAU_MMLU,
    h: float = H_MMLU,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> EFQOperand:
    """Algorithm 1 over every microscaling block of a shifted-score tile."""
    _, _, k_b, s_b, z, t = block_scale_and_residual(x, block_size)
    codes = affine_codes(z, tau, h)
    ptilde = _reconstruct(codes, s_b, block_size, t)
    return EFQOperand(ptilde=ptilde, codes=codes, scale=s_b, k=k_b, z=z, j16=None)


def efq_lut(x: Array, block_size: int = DEFAULT_BLOCK_SIZE) -> EFQOperand:
    """EFQ-Softmax-LUT (section II-D): same scale rule, 16-level fold to C8."""
    _, _, k_b, s_b, z, t = block_scale_and_residual(x, block_size)
    j16 = lut_index(z)
    codes = fold_lut(j16)
    ptilde = _reconstruct(codes, s_b, block_size, t)
    return EFQOperand(ptilde=ptilde, codes=codes, scale=s_b, k=k_b, z=z, j16=j16)


def apply_operand(ptilde: Array, v: Array) -> tuple[Array, Array]:
    """Shared-operand numerator ``Ptilde @ V`` and denominator ``Ptilde @ 1``."""
    num = ptilde @ v
    den = jnp.sum(ptilde, axis=-1, keepdims=True)
    return num, den
