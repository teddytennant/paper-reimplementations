"""Mapper support: FULL-HEAD MAPPING eq 7 against HEAD-LOCAL eqs 11-13.

Source traces are always laid out as (Ls, n, Hs, ds) in content space, and
target traces as (Lt, n, Ht, dt).
"""

from __future__ import annotations

import jax.numpy as jnp


def identity_assignment(n_heads: int):
    """a(h) = h, the natural prior when source and target expose aligned KV groups."""
    return jnp.arange(n_heads)


def cyclic_assignment(n_heads: int, shift: int):
    """The capacity-matched shifted correspondence of the figure 5 ablation."""
    return (jnp.arange(n_heads) + shift) % n_heads


def group_assignment(n_src_heads: int, n_tgt_heads: int):
    """a(h) for mismatched KV group counts: proportional index, identity when equal."""
    h = jnp.arange(n_tgt_heads)
    return (h * n_src_heads) // n_tgt_heads


def full_head_features(src_z, layers):
    """x_full = concat over j in S_l and g in [Hs] of Z^S_{j,t,g}, eq 7. Width k*Hs*ds.

    Returns (n, k*Hs*ds). Every target head in the layer shares this design matrix.
    """
    sel = jnp.take(src_z, layers, axis=0)  # (k, n, Hs, ds)
    n = sel.shape[1]
    return jnp.transpose(sel, (1, 0, 2, 3)).reshape(n, -1)


def head_local_features(src_z, layers, assign):
    """x_local = concat over j in S_l of Z^S_{j,t,a(h)}, eq 11. Width k*ds per head.

    Returns (Ht, n, k*ds), one design matrix per target KV head.
    """
    sel = jnp.take(src_z, layers, axis=0)  # (k, n, Hs, ds)
    blocks = jnp.take(sel, assign, axis=2)  # (k, n, Ht, ds)
    k, n, ht, ds = blocks.shape
    return jnp.transpose(blocks, (2, 1, 0, 3)).reshape(ht, n, k * ds)


def coefficient_count(n_tgt_layers, n_tgt_heads, k, fan_in, ds, dt):
    """C(f) = 2 Lt Ht (k f ds) dt, eq 6. Eq 12 is f = Hs, eq 13 is f = 1."""
    return 2 * n_tgt_layers * n_tgt_heads * (k * fan_in * ds) * dt
