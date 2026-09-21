"""Laplacian Point Sampler (LPS).

Deviation (specified eq 1):
    Dev(p_i) = || MLP(p_i) - (1/K) sum_{p_j in N(p_i)} MLP(p_j) ||_2
Neighbors N(p_i) come from the transient grid, not kNN.

Doubly Sorted Sampling (specified eq 3):
    Indices = Argsort(Dev)[:M]  union  Argsort(Random)[:M]
Unique, then pad/crop to exactly M. Argsort on Dev is descending (high
deviation first). Random scores come from a JAX PRNG.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pointlam.grid import neighbor_mask
from pointlam.nn import Linear, apply_linear, init_linear


class MLPParams(NamedTuple):
    lin1: Linear
    lin2: Linear


def init_mlp(key: jax.Array, in_dim: int, hidden: int, out_dim: int) -> MLPParams:
    k1, k2 = jax.random.split(key)
    return MLPParams(
        lin1=init_linear(k1, in_dim, hidden, bias=True),
        lin2=init_linear(k2, hidden, out_dim, bias=True),
    )


def mlp_apply(params: MLPParams | Linear | None, x: jax.Array) -> jax.Array:
    """Identity if params is None. Linear or two-layer GELU MLP otherwise."""
    if params is None:
        return x
    if isinstance(params, Linear):
        return apply_linear(params, x)
    h = jax.nn.gelu(apply_linear(params.lin1, x))
    return apply_linear(params.lin2, h)


def deviation(h: jax.Array, mask: jax.Array) -> jax.Array:
    """Dev_i = || h_i - mean_{j in N_i} h_j ||_2. mask is (N, N) bool."""
    w = mask.astype(h.dtype)
    k = jnp.maximum(w.sum(axis=-1, keepdims=True), 1.0)
    h_mean = (w @ h) / k
    return jnp.linalg.norm(h - h_mean, axis=-1)


def dss_indices(dev: jax.Array, rand: jax.Array, m: int) -> jax.Array:
    """Unique crop/pad of top-m Dev (desc) union top-m Random (desc) to length m.

    Dev-first order so Argsort(Dev)[:M] is always a subset when N >= M.
    Random fills only if the unique set is still short of M (N < M).
    """
    n = dev.shape[0]
    if m > n:
        raise ValueError(f"DSS needs M <= N, got M={m} N={n}")
    dev_sorted = jnp.argsort(-dev, stable=True)
    rand_sorted = jnp.argsort(-rand, stable=True)
    candidates = jnp.concatenate([dev_sorted[:m], rand_sorted[:m]])
    return _unique_first_m(candidates, n, m)


def _unique_first_m(candidates: jax.Array, n: int, m: int) -> jax.Array:
    def body(carry, idx):
        out, nfill, seen = carry
        is_new = ~seen[idx]
        take = is_new & (nfill < m)
        slot = jnp.minimum(nfill, m - 1)
        out = jnp.where(take, out.at[slot].set(idx), out)
        seen = jnp.where(take, seen.at[idx].set(True), seen)
        nfill = nfill + take.astype(jnp.int32)
        return (out, nfill, seen), None

    out0 = jnp.zeros((m,), dtype=jnp.int32)
    seen0 = jnp.zeros((n,), dtype=bool)
    (out, nfill, seen), _ = jax.lax.scan(
        body, (out0, jnp.int32(0), seen0), candidates
    )

    def pad(carry, i):
        out, nfill, seen = carry
        take = (~seen[i]) & (nfill < m)
        slot = jnp.minimum(nfill, m - 1)
        out = jnp.where(take, out.at[slot].set(i), out)
        seen = jnp.where(take, seen.at[i].set(True), seen)
        nfill = nfill + take.astype(jnp.int32)
        return (out, nfill, seen), None

    (out, _, _), _ = jax.lax.scan(
        pad, (out, nfill, seen), jnp.arange(n, dtype=jnp.int32)
    )
    return out


def apply_lps(
    points: jax.Array,
    features: jax.Array,
    key: jax.Array,
    m: int,
    cell_size: float,
    mlp_params: MLPParams | Linear | None = None,
    origin: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Downsample to M points. Returns (points_m, features_m, indices, dev)."""
    h = mlp_apply(mlp_params, features)
    mask = neighbor_mask(points, cell_size, origin=origin)
    dev = deviation(h, mask)
    rand = jax.random.uniform(key, (points.shape[0],), dtype=dev.dtype)
    idx = dss_indices(dev, rand, m)
    return points[idx], features[idx], idx, dev
