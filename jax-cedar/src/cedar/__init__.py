"""JAX implementation of CEDAR residual summary routing."""

from cedar.core import (
    chunk_summaries,
    dense_attention,
    forced_refinement_mask,
    greedy_refine,
    hard_sparse_attention,
    hybrid_attention,
    reconstruction_bound,
    refinement_priority,
    scale_tau,
)

__all__ = [
    "chunk_summaries",
    "dense_attention",
    "forced_refinement_mask",
    "greedy_refine",
    "hard_sparse_attention",
    "hybrid_attention",
    "reconstruction_bound",
    "refinement_priority",
    "scale_tau",
]
