"""Top-level WRP entry point on synthetic (or checkpoint) layer weight dicts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jax import Array

from wrp.allocate import allocate_removals
from wrp.cluster import select_num_clusters, spectral_cluster
from wrp.constants import EPS, KMEANS_SEED
from wrp.similarity import pairwise_similarity


def prune_blocks(
    weights: Sequence[Mapping[str, Array]],
    p: int,
    *,
    seed: int = KMEANS_SEED,
    eps: float = EPS,
) -> Array:
    """Return the removal set D of P block indices. Remaining weights are untouched."""
    similarity = pairwise_similarity(weights, eps=eps)
    k, _rho = select_num_clusters(similarity, eps=eps)
    labels = spectral_cluster(similarity, k, seed=seed, eps=eps)
    return allocate_removals(similarity, labels, p)
