"""Lemma 15 / Eqs. (6)–(9): heavy-coordinate truncation of a nonnegative diagonal.

Sampling-and-query access SQ(D) (problem setting): query d_i, obtain
||D||_F, and sample i with probability d_i^2 / ||D||_F^2.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def frobenius_norm_diag(d: Array) -> Array:
    """||D||_F for D = diag(d). Equals ||d||_2."""
    return jnp.linalg.norm(d)


def sample_index(key: jax.Array, d: Array) -> Array:
    """Sample i ~ d_i^2 / ||D||_F^2 (SQ(D) sampling oracle)."""
    weights = jnp.square(d)
    total = jnp.sum(weights)
    p = jnp.where(total > 0, weights / total, jnp.full_like(weights, 1.0 / d.shape[0]))
    return jax.random.choice(key, d.shape[0], p=p)


def sample_count_R(frobenius: float, eps: float, delta: float) -> int:
    """Number of SQ(D) samples in Lemma 15 / Eq. (8).

    R = ceil( (||D||_F^2 / ϵ^2) ln(||D||_F^2 / (ϵ^2 δ)) ).
    """
    if frobenius <= 0.0 or eps <= 0.0:
        return 0
    ratio = (frobenius * frobenius) / (eps * eps)
    inner = ratio / delta
    if inner <= 1.0:
        return int(max(1, round(ratio)))
    return int(jnp.ceil(ratio * jnp.log(inner)))


def heavy_set(d: Array, eps: float) -> Array:
    """Exact heavy coordinates H_ϵ = {i : d_i ≥ ϵ} (Eq. (6))."""
    return jnp.asarray(jnp.nonzero(d >= eps)[0], dtype=jnp.int32)


def truncate_diagonal(
    key: jax.Array,
    d: Array,
    eps: float,
    delta: float,
    *,
    exact: bool = False,
) -> Tuple[Array, Array]:
    """Lemma 15: return (S, d̃_S) for the truncated diagonal D̃.

    On the successful event, Supp(D̃) = H_ϵ, rank(D̃) ≤ ||D||_F^2 / ϵ^2,
    and ||D − D̃||_op ≤ ϵ. Light coordinates are never retained.

    If ``exact=True``, return H_ϵ without sampling (the successful event
    of Lemma 15). Used in deterministic tests of later steps.
    """
    d = jnp.asarray(d)
    if exact:
        idx = heavy_set(d, eps)
        return idx, d[idx]

    fro = float(frobenius_norm_diag(d))
    # Lemma 15 edge cases: D = 0 or ϵ ≥ ||D||_F ⇒ D̃ = 0 already satisfies
    # ||D − D̃||_op ≤ ϵ.
    if fro == 0.0 or eps >= fro:
        empty_i = jnp.zeros((0,), dtype=jnp.int32)
        empty_v = jnp.zeros((0,), dtype=d.dtype)
        return empty_i, empty_v

    R = sample_count_R(fro, eps, delta)
    keys = jax.random.split(key, max(R, 1))
    retained: dict[int, float] = {}
    for r in range(R):
        i = int(sample_index(keys[r], d))
        di = float(d[i])
        if i not in retained and di >= eps:
            retained[i] = di
    if not retained:
        empty_i = jnp.zeros((0,), dtype=jnp.int32)
        empty_v = jnp.zeros((0,), dtype=d.dtype)
        return empty_i, empty_v
    idx = jnp.asarray(sorted(retained.keys()), dtype=jnp.int32)
    vals = jnp.asarray([retained[int(i)] for i in idx], dtype=d.dtype)
    return idx, vals
