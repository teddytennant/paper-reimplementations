"""LoRP two-stage redundancy allocation on the WRP similarity matrix."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

NEG_INF = -1.0e30


def _block_redundancy(similarity: Array, members: Array, index: int) -> Array:
    others = members[members != index]
    if others.size == 0:
        return jnp.array(-jnp.inf, dtype=similarity.dtype)
    return jnp.mean(similarity[index, others])


def _mean_pairwise(similarity: Array, members: Array) -> Array:
    n_remain = int(members.size)
    if n_remain < 2:
        return jnp.array(NEG_INF, dtype=similarity.dtype)
    sub = similarity[jnp.ix_(members, members)]
    total = jnp.sum(sub) - jnp.trace(sub)
    return total / (n_remain * (n_remain - 1))


def _eligible_in(members: Array, remaining: set[int], protected: set[int]) -> list[int]:
    return [int(i) for i in members if int(i) in remaining and int(i) not in protected]


def _most_redundant(
    similarity: Array,
    cluster_members: Array,
    candidates: list[int],
) -> int:
    scores = [
        (float(_block_redundancy(similarity, cluster_members, idx)), idx)
        for idx in candidates
    ]
    # Highest r; ties -> lower block index.
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1]


def allocate_removals(similarity: Array, labels: Array, p: int) -> Array:
    """Two-stage LoRP allocation. Protects blocks 0 and L-1. Returns D, |D|=P."""
    n_layers = int(similarity.shape[0])
    protected = {0, n_layers - 1}
    max_p = max(0, n_layers - 2)
    p = min(int(p), max_p)
    if p <= 0:
        return jnp.array([], dtype=jnp.int32)

    remaining = set(range(n_layers))
    removed: list[int] = []
    n_clusters = int(jnp.max(labels)) + 1
    clusters = [jnp.where(labels == k)[0] for k in range(n_clusters)]

    # Stage 1: one pass over clusters in order, most-redundant eligible block each.
    for members in clusters:
        if len(removed) >= p:
            break
        eligible = _eligible_in(members, remaining, protected)
        if not eligible:
            continue
        choice = _most_redundant(similarity, members, eligible)
        remaining.remove(choice)
        removed.append(choice)

    # Stage 2: repeatedly prune from the cluster with highest remaining rbar.
    while len(removed) < p:
        best_k = None
        best_rbar = None
        for k, members in enumerate(clusters):
            remain_k = jnp.array(
                [int(i) for i in members if int(i) in remaining],
                dtype=jnp.int32,
            )
            eligible = _eligible_in(members, remaining, protected)
            if not eligible:
                continue
            rbar = float(_mean_pairwise(similarity, remain_k))
            if (
                best_rbar is None
                or rbar > best_rbar + 1e-15
                or (abs(rbar - best_rbar) <= 1e-15 and k < best_k)
            ):
                best_rbar = rbar
                best_k = k
        if best_k is None:
            leftover = sorted(i for i in remaining if i not in protected)
            need = p - len(removed)
            removed.extend(leftover[:need])
            break
        members = clusters[best_k]
        eligible = _eligible_in(members, remaining, protected)
        choice = _most_redundant(similarity, members, eligible)
        remaining.remove(choice)
        removed.append(choice)

    return jnp.array(sorted(removed), dtype=jnp.int32)
