"""Semantic-Geometric Decoupled Routing (arXiv 2609.22884)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from sgdr.attention import dense_causal_attention, sparse_block_attention
from sgdr.rope import apply_rope, rope_frequencies, to_complex, to_real
from sgdr.route import top_p_mask
from sgdr.spectrum import (
    Spectrum,
    block_means,
    estimate_spectrum,
    exact_block_score,
    macroscopic_only_score,
    microscopic_interference,
    online_block_score,
    score_from_ratio,
    traditional_pooled_score,
)

__all__ = [
    "Spectrum",
    "apply_rope",
    "block_means",
    "dense_causal_attention",
    "estimate_spectrum",
    "exact_block_score",
    "macroscopic_only_score",
    "microscopic_interference",
    "online_block_score",
    "rope_frequencies",
    "score_from_ratio",
    "sparse_block_attention",
    "to_complex",
    "to_real",
    "top_p_mask",
    "traditional_pooled_score",
]
