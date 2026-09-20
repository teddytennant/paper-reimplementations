"""Parity curriculum distributions D_r (paper §3 and §5.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_parity(key, n: int, phase: int, batch_size: int, subset=None):
    """Draw a batch from curriculum phase ``r = phase``.

    First ``r`` bits are i.i.d. uniform ±1; bits ``r+1..n`` are fixed at +1.
    Target is the product of the first ``r`` bits (prefix parity), or of
    ``S ∩ {1..r}`` when ``subset`` (0-based positions) is given.
    """
    if phase < 1 or phase > n:
        raise ValueError(f"phase must be in 1..n, got {phase} with n={n}")
    bits = jnp.ones((batch_size, n), dtype=jnp.float32)
    varying = jnp.where(
        jax.random.bernoulli(key, 0.5, (batch_size, phase)),
        1.0,
        -1.0,
    )
    bits = bits.at[:, :phase].set(varying)
    y = parity_target(bits, phase, subset)
    return bits, y


def parity_target(bits, phase: int, subset=None):
    """Prefix product, or subset-parity over ``S ∩ {1..phase}``."""
    bits = jnp.asarray(bits, dtype=jnp.float32)
    n = bits.shape[-1]
    prefix_mask = jnp.arange(n) < phase
    if subset is None:
        take = prefix_mask
    else:
        member = jnp.zeros((n,), dtype=jnp.float32).at[jnp.asarray(subset)].set(1.0)
        take = prefix_mask & (member > 0)
    selected = jnp.where(take, bits, 1.0)
    return jnp.prod(selected, axis=-1)


def sample_subset(key, n: int, k: int):
    """Fixed random subset of ``k`` bit positions in ``0..n-1`` (§5.1)."""
    return jax.random.choice(key, n, (k,), replace=False)
