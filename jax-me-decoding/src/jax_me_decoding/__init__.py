"""Mahalanobis-Ensemble Decoding (ME-Decoding) kernel in JAX."""

from jax_me_decoding.kernel import adaptive_bandwidth, pairwise_euclidean, rbf_kernel
from jax_me_decoding.select import (
    greedy_select,
    mahalanobis_energy,
    me_sample,
    me_select,
    me_truncate_logits,
)

__version__ = "0.1.0"

__all__ = [
    "adaptive_bandwidth",
    "greedy_select",
    "mahalanobis_energy",
    "me_sample",
    "me_select",
    "me_truncate_logits",
    "pairwise_euclidean",
    "rbf_kernel",
]
