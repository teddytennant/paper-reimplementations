"""Facade for the block router.

Scores follow eq 4, eq 12, and the causal top-p rule. They do not include
1/sqrt(d). See sgdr.attention for the scaled sparse output.
"""

from sgdr.attention import dense_causal_attention, sparse_block_attention
from sgdr.rope import apply_rope, rope_frequencies, to_complex, to_real
from sgdr.route import causal_softmax, top_p_mask
from sgdr.spectrum import (
    Spectrum,
    block_means,
    estimate_spectrum,
    exact_block_score,
    high_frequency_mask,
    macroscopic_only_score,
    microscopic_interference,
    online_block_score,
    pair_ratio,
    score_from_ratio,
    traditional_pooled_score,
    within_block_kernel,
)

__all__ = [
    "Spectrum",
    "apply_rope",
    "block_means",
    "causal_softmax",
    "dense_causal_attention",
    "estimate_spectrum",
    "exact_block_score",
    "high_frequency_mask",
    "macroscopic_only_score",
    "microscopic_interference",
    "online_block_score",
    "pair_ratio",
    "rope_frequencies",
    "score_from_ratio",
    "sparse_block_attention",
    "to_complex",
    "to_real",
    "top_p_mask",
    "traditional_pooled_score",
    "within_block_kernel",
]
