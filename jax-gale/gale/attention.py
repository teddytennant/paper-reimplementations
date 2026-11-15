"""GaLe attention, Section 4.1.

Standard attention (Equation 2) needs O(N^2) to hold the map A = Q K^T. The
paper models A as a tiling of b x b blocks and approximates each block as
(Equation 3)

    A_ij ~= alpha * L_ij + (1 - alpha) * G_ij,
    L_ij = v_ij * I_b        (diagonal only: fine grained local correlation)
    G_ij = c_ij * J_b        (all ones: coarse global context)

I_b keeps, inside block (i, j), only the pairs (i*b + r, j*b + r). Those are the
token pairs with the same index modulo b, which is exactly one strided subset.
J_b is constant over the whole block, so it only depends on the block indices,
which is exactly attention on tokens downsampled by b. That is where Equation 4
comes from:

    Attn~(Q,K,V) = alpha * sum_{k=1..b} F(Q^(k), K^(k), V^(k))
                   + (1 - alpha) * F(Q_ds, K_ds, V_ds)

with F the ordinary attention of Equation 2, Q^(k) the rows k, k+b, k+2b, ... and
Q_ds the input downsampled by b. The b strided terms write to disjoint rows, so
their sum is an interleave; the downsampled term produces one row per block and
is broadcast back over the block.

Section 4.1.2: b + 1 maps of size (N/b) x (N/b) instead of one N x N map, and
since the terms are independent they can run one at a time, so peak RAM is set
by a single subset. ``peak_map_elems`` reports what was actually allocated.

At b = 1 there is one strided head, it is the whole sequence, and the downsample
is the identity, so alpha * F + (1 - alpha) * F = F. The approximation contains
exact attention as a special case.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .fusion import ALPHA_DEFAULT

HIGHEST = lax.Precision.HIGHEST


def attention(q, k, v):
    """Equation 2, softmax(Q K^T / sqrt(d)) V, at HIGHEST matmul precision."""
    d = q.shape[-1]
    scores = jnp.einsum("...id,...jd->...ij", q, k, precision=HIGHEST) / jnp.sqrt(
        jnp.asarray(d, q.dtype)
    )
    return jnp.einsum("...ij,...jd->...id", jax.nn.softmax(scores, axis=-1), v, precision=HIGHEST)


def attention_map(q, k):
    d = q.shape[-1]
    scores = jnp.einsum("...id,...jd->...ij", q, k, precision=HIGHEST) / jnp.sqrt(
        jnp.asarray(d, q.dtype)
    )
    return jax.nn.softmax(scores, axis=-1)


def strided(x, b: int, phase: int):
    """Q^(k) of Equation 4: rows phase, phase + b, phase + 2b, ..."""
    return x[..., phase::b, :]


def block_mean(x, b: int):
    """Q_ds of Equation 4: mean pool over each run of b consecutive tokens.

    Mean pooling is the downsample that matches G_ij = c_ij J_b, because a block
    constant weight applied to V sums the whole block of value rows.
    """
    *lead, n, d = x.shape
    return x.reshape(*lead, n // b, b, d).mean(axis=-2)


def gale_attention(q, k, v, b: int = 2, alpha: float = ALPHA_DEFAULT, return_peak: bool = False):
    """Equation 4. Returns the N x d output, optionally the peak map size."""
    n = q.shape[-2]
    if n % b:
        raise ValueError(f"sequence length {n} is not divisible by b={b}")
    peak = 0
    out = jnp.zeros(q.shape[:-1] + (v.shape[-1],), dtype=v.dtype)
    for phase in range(b):
        qs, ks, vs = strided(q, b, phase), strided(k, b, phase), strided(v, b, phase)
        peak = max(peak, attention_map(qs, ks).size)
        out = out.at[..., phase::b, :].set(attention(qs, ks, vs))
    le = out
    if alpha >= 1.0:
        return (le, peak) if return_peak else le
    qd, kd, vd = block_mean(q, b), block_mean(k, b), block_mean(v, b)
    peak = max(peak, attention_map(qd, kd).size)
    ga = jnp.repeat(attention(qd, kd, vd), b, axis=-2)
    res = alpha * le + (1.0 - alpha) * ga
    return (res, peak) if return_peak else res


def effective_attention_matrix(q, k, b: int = 2, alpha: float = ALPHA_DEFAULT):
    """Assemble the N x N map Equation 4 implies, block by block.

    Block (i, j) comes out as alpha * v_ij I_b + (1 - alpha) * c_ij J_b whenever
    the b strided heads agree on v_ij, which happens when the scores depend only
    on the block index. In general the local part is still diagonal inside every
    block but carries a different value per phase, so it is diag over phases
    rather than a multiple of I_b. Built for inspection and for tests; it costs
    the N x N memory the rest of the module exists to avoid.
    """
    n = q.shape[-2]
    local = jnp.zeros((n, n), dtype=q.dtype)
    for phase in range(b):
        a = attention_map(strided(q, b, phase), strided(k, b, phase))
        idx = jnp.arange(phase, n, b)
        local = local.at[idx[:, None], idx[None, :]].set(a)
    coarse = attention_map(block_mean(q, b), block_mean(k, b))
    glob = jnp.kron(coarse, jnp.ones((b, b), dtype=q.dtype)) / b
    return alpha * local + (1.0 - alpha) * glob


def full_map_elems(n: int) -> int:
    return n * n
