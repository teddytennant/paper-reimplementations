"""DIPrune: training-free dual importance pruning of visual tokens.

Scoring is the inference algorithm in arXiv:2610.08341. The bound module is
the theorem. Inference does not use those gradient weights.
"""

from diprune.bound import inner_bound, relaxed_bound, token_weights, topk_mask
from diprune.prune import prune_tokens, select_topk
from diprune.scoring import (
    aggregate_attention,
    alignment_score,
    decoupled_attention,
    fuse_scores,
    ranks_from_scores,
    semantic_alignment,
    semantic_evolution,
    window_length,
)

__all__ = [
    "aggregate_attention",
    "alignment_score",
    "decoupled_attention",
    "fuse_scores",
    "inner_bound",
    "prune_tokens",
    "ranks_from_scores",
    "relaxed_bound",
    "select_topk",
    "semantic_alignment",
    "semantic_evolution",
    "token_weights",
    "topk_mask",
    "window_length",
]
