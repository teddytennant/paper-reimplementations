"""Hamming slice X_k^d = {x in {±1}^d : number of +1 coordinates = k}.

Paper Section 2.2: magnetization 1^T x = -d + 2k, set(x) locates the +1s,
vec(S) is the corresponding spin vector.
"""

from __future__ import annotations

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


def n_plus(x: Array) -> Array:
    """Number of +1 coordinates."""
    return jnp.sum(x > 0, axis=-1)


def magnetization(x: Array) -> Array:
    """Sum of spins, 1^T x = -d + 2k on the slice."""
    return jnp.sum(x, axis=-1)


def spins_from_index_set(indices, d: int, dtype=jnp.float32) -> Array:
    """vec(S): +1 on ``indices``, -1 elsewhere. Paper Section 2.2."""
    x = np.full((d,), -1.0, dtype=np.float32)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if idx.size:
        x[idx] = 1.0
    return jnp.asarray(x, dtype=dtype)


def index_set_from_spins(x) -> np.ndarray:
    """set(x): indices i with x_i = +1. Paper Section 2.2. Host-side helper."""
    x = np.asarray(x)
    return np.flatnonzero(x > 0)


def enumerate_slice(d: int, k: int, dtype=jnp.float32) -> Array:
    """All vectors in X_k^d, shape (C(d, k), d). Tiny d only."""
    if not (0 <= k <= d):
        raise ValueError(f"need 0 <= k <= d, got k={k}, d={d}")
    combos = list(combinations(range(d), k))
    rows = np.full((len(combos), d), -1.0, dtype=np.float32)
    for i, combo in enumerate(combos):
        rows[i, list(combo)] = 1.0
    return jnp.asarray(rows, dtype=dtype)


def random_slice_spin(key: Array, d: int, k: int, dtype=jnp.float32) -> Array:
    """Uniform random vector on X_k^d."""
    perm = jax.random.permutation(key, d)
    x = jnp.full((d,), -1.0, dtype=dtype)
    return x.at[perm[:k]].set(jnp.asarray(1.0, dtype=dtype))
