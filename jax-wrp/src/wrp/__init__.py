"""Weight-Redundancy Pruning (WRP) in JAX. arXiv:2609.09883."""

from wrp.allocate import allocate_removals
from wrp.cluster import select_num_clusters, spectral_cluster
from wrp.constants import EPS, PROJ_KEYS, SCALE_KEYS
from wrp.prune import prune_blocks
from wrp.similarity import (
    center_output_channels,
    depth_binned_similarity,
    gram,
    linear_cka,
    pairwise_similarity,
    projection_scale_vectors,
)

__all__ = [
    "EPS",
    "PROJ_KEYS",
    "SCALE_KEYS",
    "allocate_removals",
    "center_output_channels",
    "depth_binned_similarity",
    "gram",
    "linear_cka",
    "pairwise_similarity",
    "projection_scale_vectors",
    "prune_blocks",
    "select_num_clusters",
    "spectral_cluster",
]
